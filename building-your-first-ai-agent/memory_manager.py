#!/usr/bin/env python3
"""
memory_manager.py

Conversation memory for an agent, using nothing but local state (a JSON file).

LLMs are stateless: every call must include whatever the model should
"remember". Sending the full transcript forever gets expensive and eventually
overflows the context window. This script keeps three kinds of memory:

    1. Recent window   - the last few messages, verbatim.
    2. Running summary - older messages get folded into a short summary
                         ("compaction") once the window grows past a budget.
    3. Facts           - things you explicitly told it to remember. They are
                         retrieved by keyword overlap and injected only when
                         relevant to the current message.

Everything is saved to disk, so memory survives restarts.

Requires (interactive chat only):
    pip install anthropic
    export ANTHROPIC_API_KEY=...

Usage:
    python memory_manager.py --demo                  # offline walkthrough
    python memory_manager.py --chat                  # real chat with memory
    python memory_manager.py --chat --memory-file me.json

In chat: /remember <fact>   /facts   /forget <number>   /memory   /quit
"""

import argparse
import json
import math
import os
import re
import sys
import tempfile

MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "that", "the", "to", "was", "what", "with", "you", "am",
    "do", "find", "any", "some", "should",
}


def tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP_WORDS}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough rule of thumb: ~4 characters per token


# --------------------------------------------------------------------------
# Summarizers: take (previous summary, old messages) and return a new summary
# --------------------------------------------------------------------------

def extractive_summarizer(previous: str, old_messages: list) -> str:
    """No-LLM fallback: keep a clipped line per old message."""
    lines = [f"{m['role']}: {m['content'][:90]}" for m in old_messages]
    combined = (previous + "\n" if previous else "") + "\n".join(lines)
    if len(combined) > 800:
        combined = combined[-800:]
        combined = combined[combined.find("\n") + 1:]  # don't start mid-line
    return combined


def make_llm_summarizer(client, model: str):
    def summarize(previous: str, old_messages: list) -> str:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old_messages)
        prompt = (
            "Update the running summary of a conversation. Keep names, preferences, decisions "
            "and open tasks. Maximum 150 words. Reply with the summary only.\n\n"
            f"Current summary:\n{previous or '(none yet)'}\n\n"
            f"New messages to fold in:\n{transcript}"
        )
        response = client.messages.create(
            model=model, max_tokens=400, messages=[{"role": "user", "content": prompt}]
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()

    return summarize


# --------------------------------------------------------------------------
# The memory manager
# --------------------------------------------------------------------------

class MemoryManager:
    def __init__(self, path, max_recent_tokens=1500, keep_last=6, summarizer=extractive_summarizer):
        self.path = path
        self.max_recent_tokens = max_recent_tokens
        self.keep_last = keep_last
        self.summarizer = summarizer
        self.messages: list = []   # recent window
        self.summary: str = ""     # compacted older history
        self.facts: list = []      # long-term facts
        self.load()

    # ---- persistence ----
    def load(self):
        if self.path and os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.messages = data.get("messages", [])
            self.summary = data.get("summary", "")
            self.facts = data.get("facts", [])

    def save(self):
        if not self.path:
            return
        data = {"messages": self.messages, "summary": self.summary, "facts": self.facts}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)  # atomic: never leaves a half-written file

    # ---- short-term memory ----
    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._compact_if_needed()
        self.save()

    def recent_tokens(self) -> int:
        return sum(estimate_tokens(m["content"]) for m in self.messages)

    def _compact_if_needed(self):
        if self.recent_tokens() <= self.max_recent_tokens or len(self.messages) <= self.keep_last:
            return
        cut = len(self.messages) - self.keep_last
        # The API expects the window to start with a user message, so move the cut forward.
        while cut < len(self.messages) and self.messages[cut]["role"] != "user":
            cut += 1
        if cut >= len(self.messages):
            return
        old, self.messages = self.messages[:cut], self.messages[cut:]
        self.summary = self.summarizer(self.summary, old)

    # ---- long-term facts ----
    def remember(self, fact: str):
        fact = fact.strip()
        if fact and fact not in self.facts:
            self.facts.append(fact)
            self.save()

    def forget(self, number: int) -> bool:
        if 1 <= number <= len(self.facts):
            del self.facts[number - 1]
            self.save()
            return True
        return False

    def recall(self, query: str, k: int = 3) -> list:
        """Return up to k facts that share words with the query, best match first."""
        query_words = tokens(query)
        scored = []
        for index, fact in enumerate(self.facts):
            fact_words = tokens(fact)
            overlap = len(query_words & fact_words)
            if overlap:
                scored.append((overlap / math.sqrt(len(fact_words) or 1), index, fact))
        scored.sort(key=lambda item: (-item[0], -item[1]))  # ties: newest fact first
        return [fact for _, _, fact in scored[:k]]

    # ---- assembling what the model sees ----
    def build_system(self, query: str) -> str:
        parts = ["You are a helpful assistant with persistent memory. Be concise."]
        if self.summary:
            parts.append("Summary of the earlier conversation:\n" + self.summary)
        relevant = self.recall(query)
        if relevant:
            parts.append("Things you remember that may be relevant:\n" + "\n".join(f"- {f}" for f in relevant))
        return "\n\n".join(parts)

    def stats(self) -> str:
        return (f"{len(self.messages)} recent messages (~{self.recent_tokens()} tokens), "
                f"summary {len(self.summary)} chars, {len(self.facts)} facts")


# --------------------------------------------------------------------------
# Offline demo
# --------------------------------------------------------------------------

def run_demo():
    print("DEMO: tiny token budget so you can watch compaction happen.\n")
    path = os.path.join(tempfile.mkdtemp(), "demo_memory.json")
    memory = MemoryManager(path, max_recent_tokens=90, keep_last=4)

    memory.remember("User is vegetarian and allergic to peanuts")
    memory.remember("User's dog is called Biscuit")

    turns = [
        "I'm planning a trip to Lisbon in March.",
        "I want to see the old trams and a few museums.",
        "I only have four days, so I'd rather not rush.",
        "Can you suggest a rough day-by-day plan for the four days?",
        "Day two sounds good, but swap the museum for a walking tour.",
        "Also, I'll be travelling with my sister.",
    ]
    for text in turns:
        memory.add("user", text)
        memory.add("assistant", f"(mock reply to: {text[:40]})")
        print(f"after turn -> {memory.stats()}")

    query = "Find me a vegetarian-friendly dinner spot"
    print(f"\nNew question: {query!r}")
    print("\n--- system prompt the model would receive ---")
    print(memory.build_system(query))
    print("--- recent messages sent alongside it ---")
    for m in memory.messages:
        print(f"{m['role']}: {m['content']}")

    reloaded = MemoryManager(path)
    print(f"\nReloaded from disk -> {reloaded.stats()}")


# --------------------------------------------------------------------------
# Interactive chat
# --------------------------------------------------------------------------

def run_chat(memory_file: str, model: str):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first, or try --demo.")
    import anthropic

    client = anthropic.Anthropic()
    memory = MemoryManager(memory_file, summarizer=make_llm_summarizer(client, model))
    print(f"Memory file: {memory_file}  ({memory.stats()})")
    print("Commands: /remember <fact>  /facts  /forget <number>  /memory  /quit\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/memory":
            print(memory.stats())
            if memory.summary:
                print("Summary:\n" + memory.summary)
            continue
        if user_input == "/facts":
            for i, fact in enumerate(memory.facts, 1):
                print(f"  {i}. {fact}")
            if not memory.facts:
                print("  (no facts yet)")
            continue
        if user_input.startswith("/remember "):
            memory.remember(user_input[len("/remember "):])
            print("Remembered.")
            continue
        if user_input.startswith("/forget "):
            arg = user_input[len("/forget "):].strip()
            print("Forgotten." if arg.isdigit() and memory.forget(int(arg)) else "No such fact number.")
            continue

        system = memory.build_system(user_input)  # facts relevant to THIS message
        memory.add("user", user_input)
        response = client.messages.create(
            model=model, max_tokens=600, system=system, messages=memory.messages
        )
        reply = "".join(b.text for b in response.content if b.type == "text")
        memory.add("assistant", reply)
        print(f"assistant> {reply}\n")


def main():
    parser = argparse.ArgumentParser(description="Local conversation memory manager.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Offline walkthrough, no API key needed")
    mode.add_argument("--chat", action="store_true", help="Interactive chat with persistent memory")
    parser.add_argument("--memory-file", default="memory.json")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_chat(args.memory_file, args.model)


if __name__ == "__main__":
    main()

