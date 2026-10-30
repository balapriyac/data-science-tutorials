#!/usr/bin/env python3
"""
react_loop.py

A ReAct ("Reason + Act") agent in plain Python, with no agent framework.

The loop is the whole idea:

    1. Ask the model what to do next.
    2. If it asks to use a tool, run the tool.
    3. Feed the result back as an "Observation" and go to step 1.
    4. Stop when the model gives a "Final Answer" (or we run out of steps).

The model is told to answer in a strict text format:

    Thought: <reasoning>
    Action: <tool name>
    Action Input: <tool input>

We pass stop_sequences=["Observation:"] so the model cannot invent its own
tool results. We run the tool, then append the real Observation ourselves.

Requires (real mode only):
    pip install anthropic
    export ANTHROPIC_API_KEY=...

Usage:
    python react_loop.py "What is 15% of 2480, plus 37?"
    python react_loop.py --mock          # offline demo, no API key needed
"""

import argparse
import ast
import datetime
import operator
import os
import re
import sys

MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")

# --------------------------------------------------------------------------
# Tools: plain functions that take one string and return one string.
# --------------------------------------------------------------------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node):
    """Evaluate a parsed arithmetic expression. Anything else is rejected."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    # We parse the expression instead of calling eval(), so the model can
    # never run arbitrary code through this tool.
    try:
        tree = ast.parse(expression.strip().replace(",", ""), mode="eval")
        return str(_eval_node(tree.body))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:
        return f"Error: {exc}"


def word_count(text: str) -> str:
    return str(len(text.split()))


def current_date(_: str) -> str:
    return datetime.date.today().isoformat()


TOOLS = {
    "calculator": (calculator, "Evaluate an arithmetic expression, e.g. '2480 * 0.15'."),
    "word_count": (word_count, "Count the words in a piece of text."),
    "current_date": (current_date, "Get today's date. Input is ignored."),
}

SYSTEM_PROMPT = """You are a careful assistant that solves problems step by step.

You can use these tools:
{tool_list}

Use EXACTLY this format when you need a tool:
Thought: <your reasoning about what to do next>
Action: <one tool name from the list>
Action Input: <a single-line input for the tool>

After you write an Action, stop. The system will reply with an Observation.

When you have enough information, finish with:
Thought: I now know the answer
Final Answer: <your answer>
"""


def build_system_prompt() -> str:
    tool_list = "\n".join(f"- {name}: {desc}" for name, (_, desc) in TOOLS.items())
    return SYSTEM_PROMPT.format(tool_list=tool_list)


# --------------------------------------------------------------------------
# Parsing the model's reply
# --------------------------------------------------------------------------

ACTION_RE = re.compile(r"Action:\s*(.+?)\s*\n\s*Action Input:\s*(.+)")
FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


def run_tool(name: str, tool_input: str) -> str:
    name = name.strip()
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(TOOLS)}"
    func, _ = TOOLS[name]
    try:
        return func(tool_input.strip().strip("\"'"))
    except Exception as exc:  # a broken tool should not crash the agent
        return f"Error while running {name}: {exc}"


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def react_loop(question: str, llm, max_steps: int = 6) -> str:
    """Run the ReAct loop. `llm` is any function: list[message] -> str."""
    messages = [{"role": "user", "content": question}]

    for step in range(1, max_steps + 1):
        reply = llm(messages).strip() or "(empty reply)"
        print(f"\n--- Step {step} ---\n{reply}")
        messages.append({"role": "assistant", "content": reply})

        action = ACTION_RE.search(reply)
        if action:
            tool_name, tool_input = action.group(1), action.group(2)
            observation = run_tool(tool_name, tool_input)
            print(f"Observation: {observation}")
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            continue

        final = FINAL_RE.search(reply)
        if final:
            return final.group(1).strip()

        # The model broke the format. Tell it, and let it try again.
        messages.append({
            "role": "user",
            "content": "Observation: Invalid format. Reply with Thought/Action/Action Input, "
                       "or give a Final Answer.",
        })

    return "Stopped: reached the step limit without a final answer."


# --------------------------------------------------------------------------
# LLM backends: a real one, and a scripted one for offline demos/tests
# --------------------------------------------------------------------------

def make_anthropic_llm(model: str):
    import anthropic  # imported here so --mock works without the package

    client = anthropic.Anthropic()
    system = build_system_prompt()

    def llm(messages):
        response = client.messages.create(
            model=model,
            max_tokens=500,
            system=system,
            messages=messages,
            stop_sequences=["Observation:"],  # the model must not invent observations
        )
        return "".join(block.text for block in response.content if block.type == "text")

    return llm


DEMO_QUESTION = "What is 15% of 2,480, plus 37?"

MOCK_REPLIES = [
    "Thought: I need 15% of 2480 first. I will use the calculator.\n"
    "Action: calculator\nAction Input: 2480 * 0.15",
    "Thought: Now I add 37 to that result.\n"
    "Action: calculator\nAction Input: 372 + 37",
    "Thought: I now know the answer\nFinal Answer: 15% of 2,480 is 372, and adding 37 gives 409.",
]


def make_mock_llm():
    replies = iter(MOCK_REPLIES)
    return lambda messages: next(replies)


def main():
    parser = argparse.ArgumentParser(description="A vanilla-Python ReAct agent.")
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--mock", action="store_true",
                        help="Use a scripted fake model (offline, no API key needed)")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if args.mock:
        print("MOCK MODE: replies are scripted, so the question is fixed.")
        question, llm = DEMO_QUESTION, make_mock_llm()
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first, or run with --mock.")
        question, llm = args.question, make_anthropic_llm(args.model)

    print(f"Question: {question}")
    answer = react_loop(question, llm, max_steps=args.max_steps)
    print(f"\nFINAL ANSWER: {answer}")


if __name__ == "__main__":
    main()

