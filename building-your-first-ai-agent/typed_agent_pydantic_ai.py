#!/usr/bin/env python3
"""
typed_agent_pydantic_ai.py

A type-safe agent built with Pydantic AI.

Instead of parsing free text, we declare the exact shape we want back as a
Pydantic model. Pydantic AI hands that schema to the model and validates the
reply. If the reply is invalid, the validation error is sent back to the
model and it gets another try (up to `retries` times).

Two layers of checking are shown:

    1. Schema validation: types, allowed values, length limits (the model).
    2. Business rules: an @output_validator that can raise ModelRetry.

Requires:
    pip install pydantic-ai
    export ANTHROPIC_API_KEY=...        (real mode only)

Usage:
    python typed_agent_pydantic_ai.py --ticket "Checkout returns a 500 error for everyone"
    python typed_agent_pydantic_ai.py --ticket-file ticket.txt --json
    python typed_agent_pydantic_ai.py --mock     # offline demo, no API key needed
"""

import argparse
import itertools
import os
import sys
from typing import Literal

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # hide the startup banner

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

DEFAULT_MODEL = "anthropic:" + os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")

SAMPLE_TICKET = (
    "Since this morning every customer who tries to pay gets a 500 error at checkout. "
    "We are losing orders by the minute. Please fix this ASAP!"
)


# --------------------------------------------------------------------------
# The output contract
# --------------------------------------------------------------------------

class TicketTriage(BaseModel):
    category: Literal["billing", "bug", "feature_request", "account", "other"]
    priority: Literal["low", "medium", "high", "critical"]
    summary: str = Field(
        min_length=10, max_length=140,
        description="One-sentence summary of the problem",
    )
    customer_sentiment: Literal["positive", "neutral", "negative"]
    needs_escalation: bool = Field(
        description="True if a human on-call engineer must be paged"
    )
    suggested_tags: list[str] = Field(default_factory=list, max_length=5)


INSTRUCTIONS = """You triage customer support tickets.
Priority guide:
- critical: outage, data loss, security issue, or payments failing for many users
- high: a major feature is broken for some users
- medium: a bug with a workaround
- low: questions, cosmetic issues, feature ideas
Critical tickets always need escalation."""


def build_agent(model) -> Agent:
    agent = Agent(
        model,
        output_type=TicketTriage,   # <- the schema the reply must satisfy
        instructions=INSTRUCTIONS,
        retries=2,                  # how many times the model may retry after a bad reply
    )

    @agent.output_validator
    def enforce_business_rules(output: TicketTriage) -> TicketTriage:
        # Schema checks can't express rules that span several fields. This can.
        if output.priority == "critical" and not output.needs_escalation:
            raise ModelRetry("Critical tickets must have needs_escalation set to true.")
        return output

    return agent


# --------------------------------------------------------------------------
# A scripted model for offline demos: the first two replies are *bad on
# purpose*, so you can watch the validation and retry machinery work.
# --------------------------------------------------------------------------

BAD_THEN_GOOD = [
    # 1) Invalid: "urgent" is not an allowed priority
    {"category": "bug", "priority": "urgent", "summary": "Checkout fails for every customer",
     "customer_sentiment": "negative", "needs_escalation": True, "suggested_tags": ["checkout"]},
    # 2) Valid schema, but breaks the business rule (critical without escalation)
    {"category": "bug", "priority": "critical", "summary": "Checkout fails for every customer",
     "customer_sentiment": "negative", "needs_escalation": False, "suggested_tags": ["checkout"]},
    # 3) Fully valid
    {"category": "bug", "priority": "critical", "summary": "Checkout returns 500 errors for all customers",
     "customer_sentiment": "negative", "needs_escalation": True,
     "suggested_tags": ["checkout", "payments", "outage"]},
]


def make_mock_model() -> FunctionModel:
    attempts = itertools.count()

    def scripted(messages, info: AgentInfo) -> ModelResponse:
        reply = BAD_THEN_GOOD[min(next(attempts), len(BAD_THEN_GOOD) - 1)]
        output_tool = info.output_tools[0].name  # the tool Pydantic AI uses to receive the result
        return ModelResponse(parts=[ToolCallPart(tool_name=output_tool, args=reply)])

    return FunctionModel(scripted)


def count_retries(result) -> int:
    return sum(
        1
        for message in result.all_messages()
        for part in getattr(message, "parts", [])
        if isinstance(part, RetryPromptPart)
    )


def main():
    parser = argparse.ArgumentParser(description="Type-safe ticket triage agent (Pydantic AI).")
    parser.add_argument("--ticket", default=None, help="Ticket text to triage")
    parser.add_argument("--ticket-file", default=None, help="Read the ticket text from a file")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Pydantic AI model string (default: {DEFAULT_MODEL})")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON only")
    parser.add_argument("--mock", action="store_true",
                        help="Use a scripted model that fails twice, then succeeds (offline)")
    args = parser.parse_args()

    if args.ticket_file:
        with open(args.ticket_file, "r", encoding="utf-8") as f:
            ticket = f.read()
    else:
        ticket = args.ticket or SAMPLE_TICKET

    if args.mock:
        agent = build_agent(make_mock_model())
        if not args.json:
            print("MOCK MODE: the model's first two replies are deliberately invalid.\n")
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first, or run with --mock.")
        agent = build_agent(args.model)

    result = agent.run_sync(ticket)
    triage: TicketTriage = result.output  # a real TicketTriage object, not a string

    if args.json:
        print(triage.model_dump_json(indent=2))
        return

    print(f"Ticket: {ticket}\n")
    print(triage.model_dump_json(indent=2))
    print(f"\nValidation retries needed: {count_retries(result)}")
    if triage.needs_escalation:
        print(f"-> Paging on-call: [{triage.priority.upper()}] {triage.summary}")
    else:
        print(f"-> Queued normally: [{triage.priority}] {triage.summary}")


if __name__ == "__main__":
    main()

