# agent-service

The Python/FastAPI sibling of [`ai-agent-demo`](../ai-agent-demo) — the same Claude tool-calling agentic loop (reason, call a tool, observe the result, repeat), rebuilt with a production-style Python stack instead of TypeScript.

## Why this exists

`ai-agent-demo` answered "what experience do you have building AI agents?" for TypeScript. This project exists to answer the natural follow-up honestly: several Boston-area senior full-stack postings want the same agent/backend work done in Python specifically (FastAPI, Pydantic, async, AI/LLM integration), and side-project familiarity with the language isn't the same claim as production backend experience in it. This is that — the same loop, same production-readiness ports, built with the type-safety and tooling a real Python backend job would expect.

## Type safety, the TS/Kotlin-honest way

Python's type hints aren't enforced at runtime the way Kotlin's are — `mypy` catches type errors the same way `tsc` does (statically, before running), but nothing stops bad data from getting in at runtime unless something else checks it. This project pairs two things to close that gap the way a real Python backend does:

- **`mypy --strict`** — static checking, run in CI. The `tsc` of this stack.
- **Pydantic** — runtime validation at the boundaries (API requests, tool inputs). Every tool's input is a Pydantic model; the same model both validates Claude's tool call input *and* generates the JSON Schema Claude sees (`model_json_schema()`), so the schema and the validation can't drift apart the way a hand-maintained schema and a hand-maintained type could.

## Project layout

```
src/agent_service/
  config.py       Settings via pydantic-settings — fails fast on a missing ANTHROPIC_API_KEY
  exceptions.py   TransientToolError
  tools.py        Pydantic-modeled tool inputs, tool business logic, run_tool() dispatch
  trace.py        JSONL tracing, ported from ai-agent-demo's shared/trace.ts
  agent.py        the agentic loop: retry/backoff (tenacity), loop detection, tool scoping
  schemas.py      API request/response models
  main.py         FastAPI app (/health, /chat)
  cli.py          REPL entry point, same shape as ai-agent-demo's CLI
tests/            pytest + pytest-asyncio, Anthropic API mocked via a fake client
```

## Tools implemented

A customer-support flavor rather than ai-agent-demo's calculator/weather/units, chosen to be a distinct demo rather than a port of the same three tools:

- `lookup_order_status(order_id)` — mock order lookup; simulates the backing service being flaky (~40% of calls raise a retryable `TransientToolError`)
- `search_knowledge_base(query)` — keyword search over a small in-memory article set
- `send_notification(user_id, message)` — a mock action tool with a side effect, independently flaky (~30%)

## Running it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your ANTHROPIC_API_KEY

# HTTP API
uvicorn agent_service.main:app --reload
# then: POST /chat {"message": "..."}  — docs at /docs

# or the REPL, same shape as ai-agent-demo's CLI
python -m agent_service.cli
```

## Checks

```
ruff check .      # lint
mypy              # strict type check
pytest -q         # unit + integration tests (fake Anthropic client, no network/API key needed)
```

All three run in CI on every push/PR (`.github/workflows/ci.yml`).

## Status

This is milestone 1 of the planned build-out: FastAPI skeleton + the full agent loop + Pydantic-validated tools + `mypy --strict` + tests, all passing. Not yet built:

- **Milestone 2** — Postgres (SQLAlchemy 2.0) for conversation/session state and tool-call logs, async end-to-end
- **Milestone 3** — Celery + Redis for background agent tasks (run-and-poll pattern)
- **Milestone 4** — Docker + docker-compose, wired into the CI pipeline
- **Milestone 5 (stretch)** — SSE streaming responses, a live deploy
