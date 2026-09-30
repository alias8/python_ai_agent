# agent-service

The Python/FastAPI sibling of [`ai-agent-demo`](../ai-agent-demo) — the same Claude tool-calling agentic loop (reason, call a tool, observe the result, repeat), rebuilt with a production-style Python stack instead of TypeScript.

Two implementations of that loop, same tools and system prompt, so the difference is visible side by side instead of theoretical:

- **`src/agent_service/direct_api/`** — built directly on the Anthropic Python SDK's tool use feature: the FastAPI service (`/chat`) and the production-readiness ports below.
- **`src/agent_service/langchain/`** — the same three tools via LangChain's current `create_agent()`, mirroring `ai-agent-demo`'s own `src/langchain/` comparison.

## Why this exists

`ai-agent-demo` answered "what experience do you have building AI agents?" for TypeScript. This project exists to answer the natural follow-up honestly: several Boston-area senior full-stack postings want the same agent/backend work done in Python specifically (FastAPI, Pydantic, async, AI/LLM integration), and side-project familiarity with the language isn't the same claim as production backend experience in it. This is that — the same loop, same production-readiness ports, built with the type-safety and tooling a real Python backend job would expect.

## Type safety, the TS/Kotlin-honest way

Python's type hints aren't enforced at runtime the way Kotlin's are — `mypy` catches type errors the same way `tsc` does (statically, before running), but nothing stops bad data from getting in at runtime unless something else checks it. This project pairs two things to close that gap the way a real Python backend does:

- **`mypy --strict`** — static checking, run in CI. The `tsc` of this stack.
- **Pydantic** — runtime validation at the boundaries (API requests, tool inputs). Every tool's input is a Pydantic model; the same model both validates Claude's tool call input *and* generates the JSON Schema Claude sees (`model_json_schema()`), so the schema and the validation can't drift apart the way a hand-maintained schema and a hand-maintained type could.

## Project layout

```
src/agent_service/
  shared/
    config.py       Settings via pydantic-settings — fails fast on a missing ANTHROPIC_API_KEY
    exceptions.py   TransientToolError
    tools_core.py   Pydantic-modeled tool inputs + tool business logic — framework-agnostic
    trace.py        JSONL tracing, ported from ai-agent-demo's shared/trace.ts, used by both implementations
  direct_api/
    tools.py         wraps tools_core as Claude ToolParam definitions + run_tool() dispatch
    agent.py         the agentic loop: retry/backoff (tenacity), loop detection, tool scoping
    cli.py           REPL entry point, same shape as ai-agent-demo's direct-api CLI
  langchain/
    tools.py           wraps tools_core as LangChain @tool objects, with the same JSONL tracing
    knowledge_base.py  semantic search (local embeddings + ChromaDB) — replaces search_knowledge_base,
                       LangChain-only; direct_api keeps the naive keyword version
    agent.py           the same job via create_agent()
    cli.py             REPL entry point, same shape as direct_api's
  schemas.py        API request/response models
  main.py           FastAPI app (/health, /chat) — runs on direct_api, LangChain is CLI-only
tests/              pytest + pytest-asyncio, Anthropic API mocked via a fake client
```

## Tools implemented

A customer-support flavor rather than ai-agent-demo's calculator/weather/units, chosen to be a distinct demo rather than a port of the same three tools. Defined once in `shared/tools_core.py` and wrapped in each implementation's own schema format (a Claude `ToolParam` built from `model_json_schema()` for `direct_api`, the same Pydantic model passed straight in as a LangChain tool's `args_schema` for `langchain`) so both implementations are doing identical work underneath:

- `lookup_order_status(order_id)` — mock order lookup; simulates the backing service being flaky (~40% of calls raise a retryable `TransientToolError`, toggle: `shared/tools_core.SIMULATE_TOOL_FAILURES`)
- `search_knowledge_base(query)` — keyword search over a small in-memory article set in `direct_api`; real semantic (embedding-based) search over the same articles in `langchain` — see below
- `send_notification(user_id, message)` — a mock action tool with a side effect, independently flaky (~30%)

## How langchain/ works

```python
_agent = create_agent(model=_model, tools=LANGCHAIN_TOOLS, system_prompt=SYSTEM_PROMPT)
result = await _agent.ainvoke({"messages": [{"role": "user", "content": user_message}]})
```

That's the entire loop. `create_agent()` (from the `langchain` package, matching `ai-agent-demo`'s TypeScript `createAgent()`) builds a LangGraph state machine that does everything `direct_api/agent.py` spells out by hand: the growing message list, the tool_use/tool_result wiring, the loop-until-no-more-tool-calls condition. None of `direct_api`'s other production-readiness ports (retry/backoff, tool scoping) are reimplemented here on purpose, to keep the comparison focused on what the framework gives you unmodified.

## RAG: langchain/'s search_knowledge_base is semantic, direct_api's isn't

`direct_api`'s `search_knowledge_base` (`shared/tools_core.py`) is a literal substring/keyword match: split the query into words, keep any article whose title or body contains one of them. `langchain/knowledge_base.py` replaces that, LangChain-only, with real retrieval:

1. Each KB article is embedded once with a local `sentence-transformers` model (`all-MiniLM-L6-v2`, 384-dim, no API key — runs in-process) and stored in an in-memory ChromaDB collection.
2. A query is embedded the same way and matched against the collection by cosine similarity, not word overlap.
3. A relevance cutoff (`_RELEVANCE_THRESHOLD = 0.35`, picked empirically — see the comment in `knowledge_base.py`) filters out the "least-bad" result when nothing is actually relevant, rather than always returning a top match regardless of quality.

Kept deliberately out of `shared/` — `direct_api` still uses the naive keyword version, so the two implementations now differ in retrieval strategy as well as loop mechanics, which turned out to produce a real, verifiable before/after rather than a contrived one. Querying both with **"how do I send something back"** (zero words in common with the article that answers it):

- **`direct_api`** (keyword) returns **all three** articles — not because they're all relevant, but because splitting the query into words includes the single-letter term `"i"`, which substring-matches every article in the set. A real, pre-existing bug in the naive version, not a contrived example.
- **`langchain`** (semantic) returns just `Return policy: Items can be returned within 30 days of delivery for a full refund.` — the right answer, despite no shared keywords.

And for a genuinely unrelated query like **"spaceship warranty"**, `langchain` correctly returns "No knowledge base articles matched" (best score ~0.25, below the 0.35 cutoff) rather than confidently citing whichever article happened to score highest.

## What happens on a tool failure (tested, not assumed)

`lookup_order_status`'s simulated ~40% failure rate turned out to be a good way to observe each implementation's default behavior rather than guess at it from docs:

- **direct_api** — a thrown `TransientToolError` is retried up to 3 times with exponential backoff via `tenacity`; only gives up and surfaces an error to Claude after exhausting retries.
- **langchain** — this is the interesting one, and it's the opposite of `ai-agent-demo`'s TypeScript finding. LangChain.js's `ToolNode` catches a thrown error by default and turns it into an error `ToolMessage`, no crash. LangGraph's *Python* `ToolNode` does not: its default `handle_tool_errors` handler (`langgraph.prebuilt.tool_node._default_handle_tool_errors`) only catches its own `ToolInvocationError` (malformed tool-call arguments) — any exception raised from inside the tool itself, `TransientToolError` included, propagates straight up through `agent.ainvoke()` and crashes the whole run. Confirmed by driving `create_agent()` with a scripted fake chat model (no network) and watching a raised `TransientToolError` blow past the graph instead of becoming a message. `create_agent()` builds its `ToolNode` without overriding this, so it inherits the crash-by-default behavior unmodified. Opting into "don't crash" means passing `handle_tool_errors=True` (or a handler) directly to a `ToolNode` — not exposed as a `create_agent()` parameter, so it isn't wired up here, matching this comparison's rule of only using what each framework gives you unmodified.

The TypeScript and Python LangChain ecosystems don't even agree with each other on what "the default" is for the exact same framework concept — that's the real finding here, and a sharper one than either single language would have surfaced alone.

## Running it

```
python3.12 -m venv .venv && source .venv/bin/activate   # must be 3.12, see pyproject.toml's rag extra
pip install -e ".[dev]"        # includes langchain + langchain-anthropic + chromadb/sentence-transformers
cp .env.example .env   # add your ANTHROPIC_API_KEY

# HTTP API (direct_api only)
uvicorn agent_service.main:app --reload
# then: POST /chat {"message": "..."}  — docs at /docs

# or a REPL, same shape as ai-agent-demo's CLIs
python -m agent_service.direct_api.cli
python -m agent_service.langchain.cli
```

## Checks

```
ruff check .      # lint
mypy              # strict type check — includes pydantic's mypy plugin, needed for LangChain's Pydantic-based model classes
pytest -q         # unit + integration tests
```

All three run in CI on every push/PR (`.github/workflows/ci.yml`), which caches both pip packages and the downloaded embedding model (`actions/cache` on `~/.cache/huggingface`) to keep runs fast after the first. No `ANTHROPIC_API_KEY` or paid API call is needed for any of it — the Anthropic client is mocked (`tests/conftest.py`'s `FakeAnthropicClient`) — but `tests/test_knowledge_base.py` does load a real `sentence-transformers` model and run it against a real ChromaDB collection (network needed once, to populate a cold Hugging Face cache; local after that), because retrieval *quality* — does it actually find a keyword-free paraphrase, does it actually reject an irrelevant query — is the entire point of that module and can't be demonstrated with a mock.

There's no automated test for `langchain/agent.py`'s full `create_agent()` run (mocking LangGraph's internals deep enough to be worth it wasn't — the tool-failure finding above was verified with a one-off scripted fake model instead, not committed as a test); `langchain/tools.py`'s tool wrapping and `langchain/knowledge_base.py`'s retrieval are both unit-tested like everything else.

## Status

This is milestone 1 of the planned build-out: FastAPI skeleton + the full agent loop + Pydantic-validated tools + `mypy --strict` + tests, all passing. Not yet built:

- **Milestone 2** — Postgres (SQLAlchemy 2.0) for conversation/session state and tool-call logs, async end-to-end
- **Milestone 3** — Celery + Redis for background agent tasks (run-and-poll pattern)
- **Milestone 4** — Docker + docker-compose, wired into the CI pipeline
- **Milestone 5 (stretch)** — SSE streaming responses, a live deploy
