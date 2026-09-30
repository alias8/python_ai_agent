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
    db.py           SQLAlchemy engine + session factory
    exceptions.py   TransientToolError
    tools_core.py   Pydantic-modeled tool inputs + order/notification business logic
    trace.py        JSONL tracing, ported from ai-agent-demo's shared/trace.ts
  langchain/
    tools.py           wraps the tools as LangChain @tool objects, with JSONL tracing
    knowledge_base.py  hybrid search (local embeddings + pgvector, Postgres full-text, RRF)
    kb_corpus.py       loads kb_articles/ (17 Markdown help-center articles)
    chunking.py        splits articles into section-based chunks
    kb_models.py       the kb_chunks table (SQLAlchemy + pgvector + a full-text tsvector column)
    kb_ingest.py       embeds chunks into Postgres, incrementally
    context.py         AgentContext: per-run context (the store) the server supplies, never the model
    agent.py           the agent loop via create_agent(), with a per-request tool allowlist
    cli.py             REPL entry point
  schemas.py        API request/response models
  main.py           FastAPI app (/health, /chat)
migrations/         Alembic schema migrations
evals/              retrieval eval: labeled queries + `python -m evals.kb_eval`
tests/              pytest + pytest-asyncio
```

## Tools implemented

A customer-support flavor rather than ai-agent-demo's calculator/weather/units. Each tool's input is a Pydantic model passed straight in as the LangChain tool's `args_schema`, so the same model validates Claude's input and generates the schema Claude sees:

- `lookup_order_status(order_id)` — mock order lookup; simulates the backing service being flaky (~40% of calls raise a `TransientToolError`, toggle: `shared/tools_core.SIMULATE_TOOL_FAILURES`)
- `search_knowledge_base(query)` — semantic (embedding-based) search over the knowledge base — see below
- `send_notification(user_id, message)` — a mock action tool with a side effect, independently flaky (~30%)

`/chat` takes the message and the `store_id` of the store the customer is shopping with, plus an optional `allowed_tools` allowlist. It's enforced by building the agent with only those tools, so Claude never sees the others' schemas; an unknown tool name is a 422.

## How langchain/ works

```python
_agent = create_agent(model=_model, tools=LANGCHAIN_TOOLS, system_prompt=SYSTEM_PROMPT)
result = await _agent.ainvoke({"messages": [{"role": "user", "content": user_message}]})
```

That's the entire loop. `create_agent()` (from the `langchain` package, matching `ai-agent-demo`'s TypeScript `createAgent()`) builds a LangGraph state machine that handles the growing message list, the tool_use/tool_result wiring, and the loop-until-no-more-tool-calls condition — much less code than a hand-rolled loop, at the cost of loop mechanics you can no longer point at and read.

## RAG: semantic search_knowledge_base

`langchain/knowledge_base.py` replaced the project's original literal keyword match with real retrieval:

1. The knowledge base is multi-tenant: realistic help-center articles for two stores (`langchain/kb_articles/<store>/`, one Markdown file each, ~250-450 words) — 17 for `main-street-market`, 6 for `pine-and-peak` — whose policies deliberately conflict (a 30-day vs a 14-day return window). Search only ever sees the current store's articles.
2. Each article is split into chunks along its own `##` sections (`langchain/chunking.py`) — 145 chunks of 24-195 tokens, each small enough that the embedding model reads all of it.
3. Ingestion (`python -m agent_service.langchain.kb_ingest`) embeds each chunk, with its article title and section heading prefixed, using a local `sentence-transformers` model (`all-MiniLM-L6-v2`, 384-dim, no API key — runs in-process), and writes it to a Postgres `kb_chunks` table: a pgvector `vector(384)` column with an HNSW index for cosine distance.
4. Search is hybrid: the query is embedded the same way and matched against the chunks by cosine distance (pgvector's `<=>`), and separately run as a Postgres full-text query against a generated `tsvector` column; the two rankings are merged with reciprocal rank fusion.
5. The tool returns at most the top 5 chunks, each under its article and section heading, keeping only those above a relevance cutoff (`_RELEVANCE_THRESHOLD = 0.40`) or containing every term of the query — so when nothing is actually relevant it says so, rather than confidently citing the "least-bad" match.

### Postgres + pgvector, not an in-process vector store

The chunks started out in an in-memory ChromaDB collection, rebuilt (and every chunk re-embedded) on every process start. They now live in Postgres, which is what milestone 2 needed anyway, so vectors sit in the same database as the rest of the app's data rather than in a second system to run:

- **Schema via Alembic** (`migrations/versions/0001_create_kb_chunks.py`), including `CREATE EXTENSION vector` — which `alembic revision --autogenerate` doesn't emit, so the migration is hand-written; `alembic check` confirms it matches the SQLAlchemy model.
- **Ingestion is incremental, and separate from search.** Each row stores a sha256 of what was embedded (model name + chunk text): re-running ingestion re-embeds only edited chunks, deletes chunks that no longer exist, and re-embeds everything if the model changes. Search only ever embeds the query.
- **Every row records its embedding model**, and search refuses to run (a loud error, not "no matches") if the table is empty or holds vectors from a different model than the one embedding the query — vectors from two models aren't comparable, and silently comparing them returns garbage that looks plausible.
- **An HNSW gotcha, verified:** an HNSW index scan returns at most `hnsw.ef_search` rows (default 40) whatever the `LIMIT`. The eval ranks every chunk, so search raises `ef_search` per transaction (`set_config(..., true)`, i.e. `SET LOCAL`, so it can't leak to other users of a pooled connection). At 109 rows the planner actually skips the index for a full scan, so it doesn't bite yet — forcing the index returns 40 rows at the default and all 109 with it raised.

Moving from Chroma to pgvector left every eval number unchanged — the check that the migration changed storage, not retrieval.

The original motivating comparison, made when the knowledge base was three one-sentence articles: querying **"how do I send something back"** (zero words in common with the article that answers it), the keyword version returned **all three** articles — splitting the query into words includes the single-letter term `"i"`, which substring-matched every article — while semantic search returned just the return policy.

### Measuring retrieval: `python -m evals.kb_eval`

With a realistic corpus, "try a couple of queries and eyeball it" stops working, so retrieval is scored against a labeled set: `evals/kb_eval_queries.json` has 91 queries, each asked as a customer of one store — 53 natural-language paraphrases, 22 exact-identifier queries (payment error codes like `E-4020`, "APO", "$99 freight fee"), 12 unrelated ones that should return nothing (including near-misses like "how do I return a rental car" and an order number), and 4 asking one store about a policy only the other store has. The chunking results below were measured on the first 54 queries; the rest were added, before each step was built, so that step had something to be measured against. Each answerable query names the article(s) that answer it and quotes the exact evidence sentence; `tests/test_kb_eval_data.py` checks every label points at a real article and a real quote.

Two metrics matter. **hit@k / MRR** ask whether the right *article* ranks near the top. **answer returned** is stricter: does the text the tool actually hands the agent contain the answer sentence? The right article with the wrong section still leaves the agent unable to answer.

| | baseline: whole articles | chunked by section |
| --- | --- | --- |
| hit@1 (all 45 answerable) | 73.3% | **88.9%** |
| hit@3 | 93.3% | **95.6%** |
| MRR | 0.827 | **0.930** |
| hit@1, identifier queries (7) | 42.9% | **100.0%** |
| articles/chunks truncated when embedded | 17 of 17 | **0 of 109** |
| cutoff | 0.35 | 0.40 |
| answer returned at that cutoff | 64.4% | **80.0%** |
| unrelated queries rejected | 88.9% | 88.9% |
| avg text returned to the agent | ~1,600 tokens (3 whole articles) | **~300 tokens** |

What it showed, in order:

- **Every article was truncated.** `all-MiniLM-L6-v2` embeds only the first 256 tokens and silently drops the rest, so roughly half of each ~500-token article was never embedded. Queries whose answer sat past that point did measurably worse (hit@1 68% vs 80%).
- **Chunking by section beat fixed-size windows.** Three strategies were compared before building one: fixed 128-token windows with overlap, `##` sections, and sections with the article title + heading prefixed for embedding. The prefixed sections won on hit@1, MRR, and answer recall while returning the least text (fixed windows edged it on hit@3, 97.8% vs 95.6%) — a section like "Electronics can be returned within the same 30-day window..." doesn't say on its own that it's the restocking-fee policy; the prefix does. It also fixed identifier queries without hybrid search: error codes are in the section headings (`E-4020: Address verification failed`), so they're now in the embedded text of exactly the right chunk. That's 7 queries, though — too few to call hybrid search unnecessary.
- **The cutoff had to be re-tuned, twice.** 0.35 was tuned on the original one-sentence articles; on whole long articles it dropped a third of answerable queries, because a 500-token vector is diluted and scores lower against everything. After chunking the sweep favoured 0.40: 0.45 rejects all 9 unrelated queries but drops answer recall to 69%, and the agent can ignore a loosely related chunk but can't answer from one it never received. A threshold belongs to a model *and* a corpus *and* a chunking scheme. It's also tuned on the same 54 queries it's measured on, so the numbers are optimistic until the set is big enough for a held-out split.
- **Chunking isn't free.** "how do I send something back" regressed: short chunks like "Returned to sender" and "send the original item back" (exchanges) now out-rank the returns article's "How to start a return", which lands 6th. Other remaining misses are the same shape — the right article, but the answer in a lower-scoring section ("the vase arrived smashed" ranks "What happens to the damaged item" above "Report the problem within 7 days"). A cross-encoder reranker over the top ~20 chunks is the usual fix.

### Hybrid search: keyword matching for exact identifiers

After chunking, vector search already *ranked* identifier queries well — all 19 had their answer in the top 5 with no cutoff. The failures were at the cutoff: for a bare `E-6002`, the right chunk ranked first at 0.35, under 0.40, and only just above a wrong chunk at 0.33. The embedding model reads "some error code", not which one. An exact keyword match is much stronger evidence than that score. So the job was deciding *which chunks count as relevant*, not reordering them — and textbook hybrid search doesn't do that.

Four designs were measured before one was built (cutoff 0.40):

| | answer returned | identifier answer returned | unrelated rejected |
| --- | --- | --- | --- |
| vector only | 80.7% | 84.2% | 90% |
| RRF(vector, any-term keyword), cosine cutoff — the textbook version | 73.7% | 84.2% | 90% |
| RRF(vector, any-term keyword), keep if above cutoff *or* any term matches | 86.0% | 100% | **20%** |
| **RRF(vector, all-terms keyword), keep if above cutoff *or* every term matches** | **86.0%** | **100%** | **90%** |

- **The textbook version made retrieval worse.** Postgres's `ts_rank` has no IDF — unlike BM25, a common word counts as much as a rare one — so an any-term keyword ranking fused into the vector ranking was mostly noise.
- **Letting any keyword hit bypass the cutoff let unrelated queries through**: "what time does your store open" matches chunks mentioning "time".
- **Requiring every query term** (`plainto_tsquery`, which ANDs them) keeps keyword matches rare and strong. They fire for exact identifiers, where they're needed, and almost never for natural-language questions, whose words are spread across chunks. All of the gain is on identifier queries; paraphrases are unchanged.
- **Postgres's parser has quirks worth knowing:** `E-6002` becomes `'e' & '-6002'` (exact matching still works — it doesn't match `E-6001`), and `APO/FPO/DPO` becomes one token because it looks like a file path, so a query for "APO" never keyword-matches it (vector search finds that chunk anyway).
- **Fusion has a small cost:** paraphrase hit@1 dropped from 86.8% to 84.2% (one query). "when will my package arrive" keyword-matches three chunks, and fusion lifts "If your package is late" above "When your order ships". Both are still returned.

Final, cutoff 0.40: answer returned 80.7% → **86.0%**, identifier hit@1 94.7% → **100%**, unrelated rejected 90% (unchanged).

### Tenant isolation: one store's customer never sees another store's articles

Each article belongs to one store, and `kb_chunks.store_id` (indexed) is filtered on by both halves of hybrid search. Two parts of this are worth more than the `WHERE` clause:

- **The store is set by the server, never by the model.** It's in the agent's run context (`AgentContext`, passed to `create_agent(context_schema=...)` and `ainvoke(context=...)`), which LangChain hands to the tool as a `ToolRuntime` parameter — excluded from the tool schema Claude sees. The model has no argument to put a store ID in; a prompt-injected "I'm really a customer of the other store" can't change which knowledge base is searched. `tests/test_tenant_isolation.py` proves it end to end: a scripted model sends a forged `store_id` tool argument, it's dropped, and the search still runs as the context's store. `/chat` takes the store from its caller — the storefront's backend — and rejects an unknown one with a 422.
- **Filtered vector search needs pgvector's iterative scan.** An HNSW index finds the `ef_search` nearest vectors across *all* tenants, and the `WHERE` clause filters them afterwards — for a store that's a small share of the table, most candidates are another store's and get discarded, so a search can return fewer results than asked for, or none. pgvector 0.8's `hnsw.iterative_scan = strict_order` keeps scanning the index until enough rows pass the filter.

Measured: across every eval query ranked against its store's whole knowledge base, **0 of 8,605 ranked results came from another store**, and every same-question-both-stores pair (return window, reset-link expiry, installments, shipping cost) got its own store's conflicting answer. The 4 other-store queries show the remaining gap is relevance, not isolation: asked about "Plus membership", `pine-and-peak` returns its own member-discount section (not the other store's Plus article) — arguably a fair answer.

Retrieval roadmap, each step measured against this eval: ~~chunking~~ (done) → ~~pgvector~~ (done) → ~~hybrid search~~ (done) → ~~tenant filtering~~ (done). Next: server-side order authorization — per-customer order data is a database query scoped to the authenticated customer, not something to put in a vector index — and a cross-encoder reranker, since most remaining misses are the right article with the answer in a lower-ranked section. Order lookups get server-side authorization separately — per-customer order data is a database query scoped to the authenticated customer, not something to put in a vector index.

## What happens on a tool failure (tested, not assumed)

`lookup_order_status`'s simulated ~40% failure rate turned out to be a good way to observe LangChain's default behavior rather than guess at it from docs — and it's the opposite of `ai-agent-demo`'s TypeScript finding. LangChain.js's `ToolNode` catches a thrown error by default and turns it into an error `ToolMessage`, no crash. LangGraph's *Python* `ToolNode` does not: its default `handle_tool_errors` handler (`langgraph.prebuilt.tool_node._default_handle_tool_errors`) only catches its own `ToolInvocationError` (malformed tool-call arguments) — any exception raised from inside the tool itself, `TransientToolError` included, propagates straight up through `agent.ainvoke()` and crashes the whole run. Confirmed by driving `create_agent()` with a scripted fake chat model (no network) and watching a raised `TransientToolError` blow past the graph instead of becoming a message. `create_agent()` builds its `ToolNode` without overriding this, so it inherits the crash-by-default behavior. Opting into "don't crash" means passing `handle_tool_errors=True` (or a handler) directly to a `ToolNode` — not exposed as a `create_agent()` parameter.

The TypeScript and Python LangChain ecosystems don't even agree with each other on what "the default" is for the exact same framework concept.

**Consequence today:** nothing retries or catches tool errors, so with `SIMULATE_TOOL_FAILURES` on, a `/chat` request that calls `lookup_order_status` fails with a 500 roughly 40% of the time.

## Running it

```
python3.12 -m venv .venv && source .venv/bin/activate   # must be 3.12, see the pins in pyproject.toml
pip install -e ".[dev]"
cp .env.example .env   # add your ANTHROPIC_API_KEY

docker compose up -d                          # Postgres 16 + pgvector on port 5433 (plus a test database)
alembic upgrade head                          # create the schema
python -m agent_service.langchain.kb_ingest   # embed the knowledge base into Postgres (re-run after editing articles)

# HTTP API
uvicorn agent_service.main:app --reload
# then: POST /chat {"message": "...", "store_id": "main-street-market"}  — docs at /docs

# or a REPL, same shape as ai-agent-demo's CLIs
python -m agent_service.langchain.cli --store pine-and-peak   # default: main-street-market
```

## Checks

```
ruff check .      # lint
mypy              # strict type check — includes pydantic's mypy plugin, needed for LangChain's Pydantic-based model classes
pytest -q         # unit + integration tests
python -m evals.kb_eval  # retrieval quality report (not run in CI — it reports numbers, it doesn't pass/fail)
```

All three run in CI on every push/PR (`.github/workflows/ci.yml`), which caches both pip packages and the downloaded embedding model (`actions/cache` on `~/.cache/huggingface`) to keep runs fast after the first. No `ANTHROPIC_API_KEY` or paid API call is needed for any of it — no test calls Claude — but the tests do need the Postgres container running (`docker compose up -d`; CI runs the same image as a service container) — they migrate and use a separate `agent_service_test` database, never dev data — and `tests/test_knowledge_base.py` loads a real `sentence-transformers` model and runs it against real pgvector search (network needed once, to populate a cold Hugging Face cache; local after that), because retrieval *quality* — does it actually find a keyword-free paraphrase, does it actually reject an irrelevant query — is the entire point of that module and can't be demonstrated with a mock.

There's no automated test for a full `create_agent()` run (mocking LangGraph's internals deep enough to be worth it wasn't — the tool-failure finding above was verified with a one-off scripted fake model instead, not committed as a test); the tool allowlist, `/chat`, and `langchain/knowledge_base.py`'s retrieval are unit-tested like everything else.

## Status

This is milestone 1 of the planned build-out: FastAPI skeleton + the full agent loop + Pydantic-validated tools + `mypy --strict` + tests, all passing. Not yet built:

- **Milestone 2** — Postgres (SQLAlchemy 2.0) for conversation/session state and tool-call logs, async end-to-end
- **Milestone 3** — Celery + Redis for background agent tasks (run-and-poll pattern)
- **Milestone 4** — Docker + docker-compose, wired into the CI pipeline
- **Milestone 5 (stretch)** — SSE streaming responses, a live deploy
