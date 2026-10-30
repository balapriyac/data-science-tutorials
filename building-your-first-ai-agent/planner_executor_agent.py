#!/usr/bin/env python3
"""
planner_executor_agent.py

A Planner-Executor agent that RE-PLANS when a step fails.

The ReAct script (react_loop.py) decides one step at a time. This agent works differently:

    1. PLAN       The model writes the whole plan up front, as JSON.
    2. EXECUTE    Our code runs the steps in order. No model call per step.
    3. RE-PLAN    If a step fails, the model sees what was done and what went
                  wrong, and writes a new plan for the remaining work.
    4. SYNTHESIZE Once every step has run, the model writes the final report
                  from the collected results.

Planning up front costs fewer model calls and gives you a plan to inspect
(or approve) before anything runs. The price is that the plan is made with
incomplete knowledge, which is exactly why re-planning matters.

The tools analyze a CSV file inside a workspace folder. They are read-only
apart from the final report.md, and paths cannot escape the workspace.

Requires (real mode only):
    pip install anthropic
    export ANTHROPIC_API_KEY=...

Usage:
    python planner_executor_agent.py sales.csv "Which region sells the most?"
    python planner_executor_agent.py --mock      # offline demo (the first plan is wrong on purpose)
"""

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path

MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
MAX_STEPS = 6
MAX_REPLANS = 2


class ToolError(Exception):
    """A step failed in a way the planner may be able to fix."""


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative: str) -> Path:
        candidate = (self.root / str(relative)).resolve()
        if not candidate.is_relative_to(self.root):
            raise ToolError("path escapes the workspace")
        if not candidate.is_file():
            raise ToolError(f"file not found: {relative}")
        return candidate


def read_rows(ws: Workspace, path: str):
    with open(ws.path(path), "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def need_column(columns, name):
    if name not in columns:
        raise ToolError(f"no column named '{name}'. Available columns: {', '.join(columns)}")


def to_number(value, column):
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        raise ToolError(f"column '{column}' contains a non-numeric value: {value!r}")


def describe_csv(ws, path):
    columns, rows = read_rows(ws, path)
    sample = "\n".join(" | ".join(r[c] for c in columns) for r in rows[:3])
    return f"{len(rows)} rows. Columns: {', '.join(columns)}\nFirst rows:\n{sample}"


def group_sum(ws, path, group_by, value):
    columns, rows = read_rows(ws, path)
    need_column(columns, group_by)
    need_column(columns, value)
    totals = {}
    for r in rows:
        totals[r[group_by]] = totals.get(r[group_by], 0) + to_number(r[value], value)
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    return "\n".join(f"{k}: {v:,.2f}" for k, v in ranked)


def column_stats(ws, path, column):
    columns, rows = read_rows(ws, path)
    need_column(columns, column)
    nums = [to_number(r[column], column) for r in rows]
    if not nums:
        raise ToolError("the file has no data rows")
    return (f"count={len(nums)} min={min(nums):,.2f} max={max(nums):,.2f} "
            f"mean={sum(nums) / len(nums):,.2f} total={sum(nums):,.2f}")


TOOLS = {
    "describe_csv": (describe_csv, ["path"], "Show the columns, row count and first rows of a CSV."),
    "group_sum": (group_sum, ["path", "group_by", "value"],
                  "Total a numeric column for each value of another column, largest first."),
    "column_stats": (column_stats, ["path", "column"],
                     "Count, min, max, mean and total of a numeric column."),
}


def tool_help() -> str:
    return "\n".join(f"- {name}({', '.join(args)}): {desc}" for name, (_, args, desc) in TOOLS.items())


# --------------------------------------------------------------------------
# Plans: JSON in, validated steps out
# --------------------------------------------------------------------------

PLAN_SYSTEM = """You are a planner for a data-analysis agent. Reply with ONLY a JSON array of steps.
Each step is an object: {{"tool": "<tool name>", "args": {{...}}, "why": "<short reason>"}}
Available tools:
{tools}
Use at most {max_steps} steps. Do not include a step that writes the final report; that happens automatically."""

REPLAN_SYSTEM = PLAN_SYSTEM + """
A step failed. Using the completed results and the error, write a NEW plan containing only the
remaining work, including a corrected version of the failed step."""

REPORT_SYSTEM = """You write short, factual reports (under 150 words) from analysis results.
Only use numbers that appear in the results."""


def extract_json(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON array found in the model's reply")
    return json.loads(match.group(0))


def validate_plan(raw) -> list:
    if not isinstance(raw, list) or not raw:
        raise ValueError("a plan must be a non-empty JSON array")
    if len(raw) > MAX_STEPS:
        raise ValueError(f"plan has {len(raw)} steps; the maximum is {MAX_STEPS}")
    steps = []
    for i, step in enumerate(raw, 1):
        if not isinstance(step, dict) or step.get("tool") not in TOOLS:
            raise ValueError(f"step {i}: unknown or missing tool")
        args = step.get("args")
        expected = TOOLS[step["tool"]][1]
        if not isinstance(args, dict) or set(args) != set(expected):
            raise ValueError(f"step {i}: {step['tool']} needs exactly these args: {expected}")
        steps.append({"tool": step["tool"], "args": args, "why": str(step.get("why", ""))})
    return steps


def get_plan(llm, system: str, user: str, attempts: int = 2) -> list:
    """Ask the model for a plan; if it is malformed, tell it why and ask once more."""
    prompt = user
    for attempt in range(attempts):
        reply = llm(system, prompt)
        try:
            return validate_plan(extract_json(reply))
        except (ValueError, json.JSONDecodeError) as exc:
            prompt = f"{user}\n\nYour previous reply was invalid ({exc}). Reply with only the JSON array."
    raise RuntimeError("the model could not produce a valid plan")


def show_plan(title: str, plan: list):
    print(f"\n{title}")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step['tool']}({json.dumps(step['args'])})  # {step['why']}")


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------

def describe_progress(done: list) -> str:
    return "\n".join(f"- {d['tool']}({json.dumps(d['args'])}) -> {d['result']}" for d in done) or "(nothing yet)"


def run_agent(goal: str, filename: str, llm, ws: Workspace) -> str:
    system = PLAN_SYSTEM.format(tools=tool_help(), max_steps=MAX_STEPS)
    plan = get_plan(llm, system, f"File: {filename}\nGoal: {goal}")
    show_plan("INITIAL PLAN", plan)

    done, replans, executed = [], 0, 0
    while plan:
        step = plan.pop(0)
        executed += 1
        if executed > MAX_STEPS * (MAX_REPLANS + 1):  # hard stop, whatever the model does
            raise RuntimeError("too many steps executed; stopping")
        func = TOOLS[step["tool"]][0]
        print(f"\n> running {step['tool']}({json.dumps(step['args'])})")
        try:
            result = func(ws, **step["args"])
        except ToolError as exc:
            print(f"  FAILED: {exc}")
            if replans >= MAX_REPLANS:
                raise RuntimeError(f"giving up after {replans} re-plans; last error: {exc}")
            replans += 1
            user = (f"File: {filename}\nGoal: {goal}\n\nCompleted so far:\n{describe_progress(done)}\n\n"
                    f"Failed step: {step['tool']}({json.dumps(step['args'])})\nError: {exc}")
            plan = get_plan(llm, REPLAN_SYSTEM.format(tools=tool_help(), max_steps=MAX_STEPS), user)
            show_plan(f"RE-PLAN #{replans}", plan)
            continue
        print(f"  ok: {result.splitlines()[0][:100]}")
        done.append({**step, "result": result})

    print("\nSYNTHESIZING REPORT")
    return llm(REPORT_SYSTEM, f"Goal: {goal}\n\nResults:\n{describe_progress(done)}").strip()


# --------------------------------------------------------------------------
# Model backends
# --------------------------------------------------------------------------

def make_anthropic_llm(model: str):
    import anthropic

    client = anthropic.Anthropic()

    def llm(system: str, user: str) -> str:
        response = client.messages.create(model=model, max_tokens=1200, system=system,
                                          messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in response.content if b.type == "text")

    return llm


SAMPLE_CSV = """region,product,units,sales
North,Widget,10,500
North,Gadget,5,300
South,Widget,20,1000
South,Gadget,8,480
East,Widget,7,350
West,Gadget,4,240
"""

MOCK_GOAL = "Which region has the highest total sales, and how do sales look overall?"


def make_mock_llm():
    """Scripted model. Its FIRST plan guesses a column name ('revenue') that doesn't exist."""
    def llm(system: str, user: str) -> str:
        if system.startswith("You write short"):
            return ("South has the highest total sales at 1,480, well ahead of North (800), East (350) "
                    "and West (240). Across all 6 rows, sales average 478.33 per row and total 2,870.")
        if "A step failed" in system:
            return json.dumps([
                {"tool": "group_sum", "args": {"path": "sales.csv", "group_by": "region", "value": "sales"},
                 "why": "The column is called 'sales', not 'revenue'"},
                {"tool": "column_stats", "args": {"path": "sales.csv", "column": "sales"},
                 "why": "Overall picture of sales"},
            ])
        return json.dumps([
            {"tool": "describe_csv", "args": {"path": "sales.csv"}, "why": "Look at the data first"},
            {"tool": "group_sum", "args": {"path": "sales.csv", "group_by": "region", "value": "revenue"},
             "why": "Total revenue per region"},
            {"tool": "column_stats", "args": {"path": "sales.csv", "column": "sales"},
             "why": "Overall picture of sales"},
        ])
    return llm


def main():
    parser = argparse.ArgumentParser(description="Planner-Executor agent with re-planning.")
    parser.add_argument("csv_file", nargs="?", help="CSV file to analyze")
    parser.add_argument("goal", nargs="?", default="Summarize this data and highlight the top performer.")
    parser.add_argument("--mock", action="store_true", help="Offline demo with a scripted model and sample data")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if args.mock:
        ws = Workspace(Path(tempfile.mkdtemp()) / "workspace")
        (ws.root / "sales.csv").write_text(SAMPLE_CSV, encoding="utf-8")
        print("MOCK MODE: scripted model; the first plan is wrong on purpose so you can watch it recover.")
        filename, goal, llm = "sales.csv", MOCK_GOAL, make_mock_llm()
    else:
        if not args.csv_file:
            parser.error("give a CSV file, or use --mock")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first, or run with --mock.")
        source = Path(args.csv_file).resolve()
        if not source.is_file():
            sys.exit(f"File not found: {args.csv_file}")
        ws = Workspace(source.parent)  # the agent can only touch the folder holding the CSV
        filename, goal, llm = source.name, args.goal, make_anthropic_llm(args.model)

    print(f"Goal: {goal}")
    try:
        report = run_agent(goal, filename, llm, ws)
    except RuntimeError as exc:
        sys.exit(f"Agent stopped: {exc}")

    out = ws.root / "report.md"
    out.write_text(f"# Report\n\n**Goal:** {goal}\n\n{report}\n", encoding="utf-8")
    print(f"\n=== REPORT ===\n{report}\n\nSaved to {out}")


if __name__ == "__main__":
    main()

