# Five Python Scripts for Building Your First AI Agent

Five self-contained scripts that take you from a bare agent loop to a supervised agent with memory, structured output and self-correcting plans. Each runs offline in a mock mode, so you can see how it works before adding an API key.

## Requirements

- Python 3.10+
- Install what each script needs:

| Script | Packages |
|---|---|
| ReAct loop | `anthropic` (real mode only) |
| Pydantic AI agent | `pydantic-ai` |
| Planner-executor agent | `anthropic` (real mode only) |
| Memory manager | `anthropic` (chat mode only) |
| Gatekeeper | `anthropic` (real mode only) |

```bash
pip install anthropic pydantic-ai
export ANTHROPIC_API_KEY=your-key-here     # only needed for real (non-mock) runs
```

The default model is `claude-sonnet-5-5`. Override it with the `AGENT_MODEL` environment variable or each script's `--model` flag. The Pydantic AI script expects a Pydantic AI model string, so it prefixes the name with `anthropic:` for you.

## Scripts

| Script | What it does |
|--------|---------------|
| `react_loop.py` | A reason-act-observe agent loop in plain Python, with a safe calculator, word counter and date tool. |
| `typed_agent_pydantic_ai.py` | Triages support tickets into a validated Pydantic model, with retries and a business-rule validator. |
| `planner_executor_agent.py` | Plans a CSV analysis up front, runs it, and re-plans when a step fails. |
| `memory_manager.py` | Rolling summary, recent window and retrievable facts, saved to a JSON file. |
| `human_in_the_loop_gatekeeper.py` | Risk-tiered approval prompts in the terminal, with an audit log. |

