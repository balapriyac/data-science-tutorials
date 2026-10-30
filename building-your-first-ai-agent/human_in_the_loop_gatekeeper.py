#!/usr/bin/env python3
"""
human_in_the_loop_gatekeeper.py

An agent that can take real actions (write files, send email, delete files)
but must get a human's approval in the terminal first.

Every tool is registered with a risk level:

    safe     runs immediately               (list_files, read_file)
    confirm  pauses for y / n / a           (write_file, send_email)
    high     pauses; you must type 'yes'    (delete_file)
    blocked  never runs, no prompt          (run_shell)

Design choices worth copying:
    - FAIL CLOSED: if there is no answer (stdin closed), the action is denied.
    - A denial is sent back to the model as a tool result, with your optional
      note ("n: use a different filename"), so the agent can adapt.
    - Every decision is appended to an audit log (audit.jsonl).
    - File tools are sandboxed to one workspace folder.
    - send_email is SIMULATED: it writes to outbox.jsonl in the workspace.

Requires (real mode only):
    pip install anthropic
    export ANTHROPIC_API_KEY=...

Usage:
    python human_in_the_loop_gatekeeper.py "Summarize notes.txt into summary.txt and email it to sam@example.com"
    python human_in_the_loop_gatekeeper.py --mock       # offline demo
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are an assistant that works with files in a workspace folder.
Some actions need a human's approval. If an action is denied, do not retry it
unchanged: read the reviewer's note, adapt, or explain what you could not do.
Keep the final reply brief."""


# --------------------------------------------------------------------------
# Tools (all file paths are confined to the workspace)
# --------------------------------------------------------------------------

class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("path escapes the workspace")
        return candidate


def build_registry(ws: Workspace) -> dict:
    def list_files(path="."):
        target = ws.path(path)
        return "\n".join(sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())) or "(empty)"

    def read_file(path):
        return ws.path(path).read_text(encoding="utf-8")[:4000]

    def write_file(path, content):
        target = ws.path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"

    def send_email(to, subject, body):
        record = {"to": to, "subject": subject, "body": body, "at": datetime.now().isoformat()}
        with open(ws.path("outbox.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return f"(simulated) email to {to} saved to outbox.jsonl"

    def delete_file(path):
        ws.path(path).unlink()
        return f"Deleted {path}"

    def run_shell(command):  # registered as 'blocked', so this body never runs
        raise RuntimeError("blocked by policy")

    def schema(props, required):
        return {"type": "object", "properties": {k: {"type": "string"} for k in props}, "required": required}

    return {
        "list_files": dict(func=list_files, risk="safe", description="List files in a workspace folder.",
                           schema=schema(["path"], [])),
        "read_file": dict(func=read_file, risk="safe", description="Read a text file.",
                          schema=schema(["path"], ["path"])),
        "write_file": dict(func=write_file, risk="confirm", description="Create or overwrite a text file.",
                           schema=schema(["path", "content"], ["path", "content"])),
        "send_email": dict(func=send_email, risk="confirm", description="Send an email.",
                           schema=schema(["to", "subject", "body"], ["to", "subject", "body"])),
        "delete_file": dict(func=delete_file, risk="high", description="Permanently delete a file.",
                            schema=schema(["path"], ["path"])),
        "run_shell": dict(func=run_shell, risk="blocked", description="Run a shell command.",
                          schema=schema(["command"], ["command"])),
    }


def tool_schemas(registry: dict) -> list:
    return [{"name": name, "description": f"{spec['description']} [risk: {spec['risk']}]",
             "input_schema": spec["schema"]} for name, spec in registry.items()]


# --------------------------------------------------------------------------
# The gatekeeper
# --------------------------------------------------------------------------

def short(value, limit=100) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text)} chars)"


class Gatekeeper:
    def __init__(self, audit_path: Path, ask=input):
        self.audit_path = audit_path
        self.ask = ask
        self.always_allowed: set = set()  # tools approved with 'a' for this session

    def _log(self, tool, args, risk, decision, note=""):
        entry = {"at": datetime.now().isoformat(timespec="seconds"), "tool": tool,
                 "args": args, "risk": risk, "decision": decision, "note": note}
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def review(self, tool: str, args: dict, risk: str):
        """Return (approved, note_for_the_model)."""
        if risk == "safe":
            self._log(tool, args, risk, "auto-approved")
            return True, ""
        if risk == "blocked":
            self._log(tool, args, risk, "blocked")
            return False, "Blocked by policy: this tool is never allowed."
        if tool in self.always_allowed:
            self._log(tool, args, risk, "approved (always)")
            return True, ""

        print(f"\n  The agent wants to run: {tool}")
        for key, value in args.items():
            print(f"    {key}: {short(value)}")
        if risk == "high":
            print("    ** This action is hard to undo. **")
            prompt = "  Type 'yes' to approve, anything else denies (add a note after 'n:'): "
        else:
            prompt = "  [y] approve  [n] deny (note after 'n:')  [a] always approve this tool: "

        try:
            answer = self.ask(prompt).strip()
        except EOFError:
            answer = "n: no human available to approve"  # fail closed

        lowered = answer.lower()
        if risk == "high" and lowered == "yes":
            self._log(tool, args, risk, "approved")
            return True, ""
        if risk == "confirm" and lowered in ("y", "yes"):
            self._log(tool, args, risk, "approved")
            return True, ""
        if risk == "confirm" and lowered == "a":
            self.always_allowed.add(tool)
            self._log(tool, args, risk, "approved (always)")
            return True, ""

        note = answer.split(":", 1)[1].strip() if answer.lower().startswith("n:") else ""
        self._log(tool, args, risk, "denied", note)
        return False, "Denied by the human reviewer." + (f" Their note: {note}" if note else "")


# --------------------------------------------------------------------------
# The agent loop
# --------------------------------------------------------------------------

def run_agent(task, client, model, registry, gate, max_turns=12) -> str:
    messages = [{"role": "user", "content": task}]
    schemas = tool_schemas(registry)

    for _ in range(max_turns):
        response = client.messages.create(model=model, max_tokens=1000, system=SYSTEM_PROMPT,
                                          tools=schemas, messages=messages)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            spec = registry.get(block.name)
            if spec is None:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"Unknown tool {block.name}", "is_error": True})
                continue

            approved, note = gate.review(block.name, block.input, spec["risk"])
            if not approved:
                print(f"  -> not run: {note}")
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": note, "is_error": True})
                continue
            try:
                output = spec["func"](**block.input)
                print(f"  -> ran {block.name}: {short(output, 80)}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
            except Exception as exc:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"Error: {exc}", "is_error": True})
        messages.append({"role": "user", "content": results})

    return "Stopped: turn limit reached."


# --------------------------------------------------------------------------
# Offline demo
# --------------------------------------------------------------------------

DEMO_TASK = "Write a short status summary to summary.txt, email it to the team, then clean up old_draft.txt."


def _tool_use(index, name, tool_input):
    block = SimpleNamespace(type="tool_use", id=f"toolu_mock_{index}", name=name, input=tool_input)
    return SimpleNamespace(stop_reason="tool_use", content=[block])


class MockClient:
    """A scripted 'model' that tries one action of each risk level."""

    def __init__(self):
        self.messages = self
        self.calls = 0
        self.script = [
            lambda: _tool_use(1, "list_files", {"path": "."}),
            lambda: _tool_use(2, "write_file", {"path": "summary.txt",
                              "content": "Weekly status: all milestones on track."}),
            lambda: _tool_use(3, "send_email", {"to": "team@example.com", "subject": "Weekly status",
                              "body": "Hi all, status is on track. Details in summary.txt."}),
            lambda: _tool_use(4, "run_shell", {"command": "pip install requests"}),
            lambda: _tool_use(5, "delete_file", {"path": "old_draft.txt"}),
        ]

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= len(self.script):
            return self.script[self.calls - 1]()
        outcomes = []
        for message in kwargs["messages"]:
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        outcomes.append("- " + item["content"].splitlines()[0][:80])
        text = "Finished. Outcome of each action:\n" + "\n".join(outcomes)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def main():
    parser = argparse.ArgumentParser(description="Agent with a human approval gate.")
    parser.add_argument("task", nargs="?", default=DEMO_TASK)
    parser.add_argument("--workspace", default="agent_workspace",
                        help="Folder the agent may read and write (default: ./agent_workspace)")
    parser.add_argument("--mock", action="store_true", help="Offline demo with a scripted model")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if args.mock:
        workspace = Workspace(Path(tempfile.mkdtemp()) / "workspace")
        (workspace.root / "old_draft.txt").write_text("outdated notes", encoding="utf-8")
        print(f"MOCK MODE: scripted model, temporary workspace at {workspace.root}")
        client, task = MockClient(), DEMO_TASK
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first, or run with --mock.")
        import anthropic
        workspace = Workspace(Path(args.workspace))
        client, task = anthropic.Anthropic(), args.task

    registry = build_registry(workspace)
    gate = Gatekeeper(workspace.root / "audit.jsonl")

    print(f"Task: {task}")
    answer = run_agent(task, client, args.model, registry, gate)
    print(f"\n=== AGENT REPLY ===\n{answer}")
    print(f"\nAudit log: {gate.audit_path}")


if __name__ == "__main__":
    main()

