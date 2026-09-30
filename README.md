# agent-service

The Python/FastAPI sibling of [`ai-agent-demo`](../ai-agent-demo) — the same Claude tool-calling agentic loop (reason, call a tool, observe the result, repeat), rebuilt with a production-style Python stack instead of TypeScript.

The loop is built on LangChain's current `create_agent()` (`src/agent_service/langchain/`), served over FastAPI (`/chat`), with a production-style retrieval tool: embedding-based search over a realistic knowledge base, measured by a labeled eval.

An earlier hand-rolled version built directly on the Anthropic SDK (`direct_api/`, with tenacity retries and loop detection) was removed once the side-by-side comparison had served its purpose; it's in the git history.

## Why this exists

`ai-agent-demo` answered "what experience do you have building AI agents?" for TypeScript. This project exists to answer the natural follow-up honestly: several Boston-area senior full-stack postings want the same agent/backend work done in Python specifically (FastAPI, Pydantic, async, AI/LLM integration), and side-project familiarity with the language isn't the same claim as production backend experience in it. This is that — the same loop, built with the type-safety and tooling a real Python backend job would expect.

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
    tools_core.py   Pydantic-modeled tool inputs + order/notification business logic
    trace.py        JSONL tracing, ported from ai-agent-demo's shared/trace.ts
  langchain/
    tools.py           wraps the tools as LangChain @tool objects, with JSONL tracing
    knowledge_base.py  semantic search (local embeddings + ChromaDB)
    kb_corpus.py       loads kb_articles/ (17 Markdown help-center articles)
    agent.py           the agent loop via create_agent(), with a per-request tool allowlist
    cli.py             REPL entry point
  schemas.py        API request/response models
  main.py           FastAPI app (/health, /chat)
evals/              retrieval eval: labeled queries + `python -m evals.kb_eval`
tests/              pytest + pytest-asyncio
```

## Tools implemented

A customer-support flavor rather than ai-agent-demo's calculator/weather/units. Each tool's input is a Pydantic model passed straight in as the LangChain tool's `args_schema`, so the same model validates Claude's input and generates the schema Claude sees:

- `lookup_order_status(order_id)` — mock order lookup; simulates the backing service being flaky (~40% of calls raise a `TransientToolError`, toggle: `shared/tools_core.SIMULATE_TOOL_FAILURES`)
- `search_knowledge_base(query)` — semantic (embedding-based) search over the knowledge base — see below
- `send_notification(user_id, message)` — a mock action tool with a side effect, independently flaky (~30%)

`/chat` accepts an optional `allowed_tools` allowlist. It's enforced by building the agent with only those tools, so Claude never sees the others' schemas; an unknown tool name is a 422.

## How langchain/ works

```python
_agent = create_agent(model=_model, tools=LANGCHAIN_TOOLS, system_prompt=SYSTEM_PROMPT)
result = await _agent.ainvoke({"messages": [{"role": "user", "content": user_message}]})
```

That's the entire loop. `create_agent()` (from the `langchain` package, matching `ai-agent-demo`'s TypeScript `createAgent()`) builds a LangGraph state machine that handles the growing message list, the tool_use/tool_result wiring, and the loop-until-no-more-tool-calls condition — much less code than a hand-rolled loop, at the cost of loop mechanics you can no longer point at and read.

## RAG: semantic search_knowledge_base

`langchain/knowledge_base.py` replaced the project's original literal keyword match with real retrieval:

1. The knowledge base is 17 realistic help-center articles (`langchain/kb_articles/`, one Markdown file each, ~400 words / 465-580 tokens). Each is embedded once with a local `sentence-transformers` model (`all-MiniLM-L6-v2`, 384-dim, no API key — runs in-process) and stored in an in-memory ChromaDB collection.
2. A query is embedded the same way and matched against the collection by cosine similarity, not word overlap.
3. The tool returns at most the top 3 articles, and only those above a relevance cutoff (`_RELEVANCE_THRESHOLD = 0.35`) — so when nothing is actually relevant it says so, rather than confidently citing the "least-bad" article.

The original motivating comparison, made when the knowledge base was three one-sentence articles: querying **"how do I send something back"** (zero words in common with the article that answers it), the keyword version returned **all three** articles — splitting the query into words includes the single-letter term `"i"`, which substring-matched every article — while semantic search returned just the return policy.

### Measuring retrieval: `python -m evals.kb_eval`

With a realistic corpus, "try a couple of queries and eyeball it" stops working, so retrieval is scored against a labeled set: `evals/kb_eval_queries.json` has 54 queries — 38 natural-language paraphrases, 7 exact-identifier queries (payment error codes like `E-4020`, plan names), and 9 unrelated ones that should return nothing, including near-misses like "how do I return a rental car". Each answerable query names the article(s) that answer it and quotes the exact evidence sentence; `tests/test_kb_eval_data.py` checks every label points at a real article and a real quote.

Baseline (one vector per whole article, 0.35 cutoff):

| | n | hit@1 | hit@3 | MRR |
| --- | --- | --- | --- | --- |
| all answerable | 45 | 73.3% | 93.3% | 0.827 |
| paraphrase | 38 | 78.9% | 97.4% | 0.871 |
| identifier | 7 | 42.9% | 71.4% | 0.592 |
| evidence inside the 256-token window | 20 | 80.0% | 100.0% | 0.883 |
| evidence past the window | 25 | 68.0% | 88.0% | 0.782 |

What it shows:

- **Every article is truncated.** `all-MiniLM-L6-v2` embeds only the first 256 tokens and silently drops the rest, so roughly half of each article is never embedded. Queries answered past the cutoff do measurably worse — they only succeed because the article's opening happens to be on-topic. This is what chunking fixes.
- **Exact identifiers are the weak spot.** Error codes like `E-6002` are near-meaningless to an embedding model (hit@1 43%); this is what hybrid keyword + vector search fixes.
- **The 0.35 cutoff no longer fits.** It was tuned on one-sentence articles; long-article vectors are diluted, so real matches score lower. At 0.35 only 64% of answerable queries get their answer back (vs 93% in the top 3 with no cutoff), while rejecting 89% of unrelated ones. A threshold is specific to a model *and* a corpus — it has to be re-tuned whenever either changes, which is why it isn't re-tuned until after chunking.

Retrieval roadmap, each step measured against this eval: chunking → pgvector (with milestone 2's Postgres) → hybrid search via Postgres full-text + vector ranking → tenant filtering. Order lookups get server-side authorization separately — per-customer order data is a database query scoped to the authenticated customer, not something to put in a vector index.

## What happens on a tool failure (tested, not assumed)

`lookup_order_status`'s simulated ~40% failure rate turned out to be a good way to observe LangChain's default behavior rather than guess at it from docs — and it's the opposite of `ai-agent-demo`'s TypeScript finding. LangChain.js's `ToolNode` catches a thrown error by default and turns it into an error `ToolMessage`, no crash. LangGraph's *Python* `ToolNode` does not: its default `handle_tool_errors` handler (`langgraph.prebuilt.tool_node._default_handle_tool_errors`) only catches its own `ToolInvocationError` (malformed tool-call arguments) — any exception raised from inside the tool itself, `TransientToolError` included, propagates straight up through `agent.ainvoke()` and crashes the whole run. Confirmed by driving `create_agent()` with a scripted fake chat model (no network) and watching a raised `TransientToolError` blow past the graph instead of becoming a message. `create_agent()` builds its `ToolNode` without overriding this, so it inherits the crash-by-default behavior. Opting into "don't crash" means passing `handle_tool_errors=True` (or a handler) directly to a `ToolNode` — not exposed as a `create_agent()` parameter.

The TypeScript and Python LangChain ecosystems don't even agree with each other on what "the default" is for the exact same framework concept.

**Consequence today:** nothing retries or catches tool errors, so with `SIMULATE_TOOL_FAILURES` on, a `/chat` request that calls `lookup_order_status` fails with a 500 roughly 40% of the time.

## Running it

```
python3.12 -m venv .venv && source .venv/bin/activate   # must be 3.12, see the pins in pyproject.toml
pip install -e ".[dev]"
cp .env.example .env   # add your ANTHROPIC_API_KEY

# HTTP API
uvicorn agent_service.main:app --reload
# then: POST /chat {"message": "..."}  — docs at /docs

# or a REPL, same shape as ai-agent-demo's CLIs
python -m agent_service.langchain.cli
```

## Checks

```
ruff check .      # lint
mypy              # strict type check — includes pydantic's mypy plugin, needed for LangChain's Pydantic-based model classes
pytest -q         # unit + integration tests
python -m evals.kb_eval  # retrieval quality report (not run in CI — it reports numbers, it doesn't pass/fail)
```

All three run in CI on every push/PR (`.github/workflows/ci.yml`), which caches both pip packages and the downloaded embedding model (`actions/cache` on `~/.cache/huggingface`) to keep runs fast after the first. No `ANTHROPIC_API_KEY` or paid API call is needed for any of it — no test calls Claude — but `tests/test_knowledge_base.py` does load a real `sentence-transformers` model and run it against a real ChromaDB collection (network needed once, to populate a cold Hugging Face cache; local after that), because retrieval *quality* — does it actually find a keyword-free paraphrase, does it actually reject an irrelevant query — is the entire point of that module and can't be demonstrated with a mock.

There's no automated test for a full `create_agent()` run (mocking LangGraph's internals deep enough to be worth it wasn't — the tool-failure finding above was verified with a one-off scripted fake model instead, not committed as a test); the tool allowlist, `/chat`, and `langchain/knowledge_base.py`'s retrieval are unit-tested like everything else.

## Status

This is milestone 1 of the planned build-out: FastAPI skeleton + the full agent loop + Pydantic-validated tools + `mypy --strict` + tests, all passing. Not yet built:

- **Milestone 2** — Postgres (SQLAlchemy 2.0) for conversation/session state and tool-call logs, async end-to-end
- **Milestone 3** — Celery + Redis for background agent tasks (run-and-poll pattern)
- **Milestone 4** — Docker + docker-compose, wired into the CI pipeline
- **Milestone 5 (stretch)** — SSE streaming responses, a live deploy
