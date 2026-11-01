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

