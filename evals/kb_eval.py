"""Retrieval eval for langchain/knowledge_base.py's semantic search.

Scores the real embedding model + Postgres (pgvector) index against the labeled
queries in kb_eval_queries.json, so a change to retrieval (chunking, a new
embedding model, hybrid search, a different threshold) is judged by numbers
rather than by eyeballing a couple of queries.

Run from the repo root:

    python -m evals.kb_eval

Reports three things:

1. Ranking quality, ignoring the threshold. Two levels:
   - Article: articles ranked by their best-matching chunk — hit@1, hit@3,
     and MRR (mean reciprocal rank: 1.0 if the right article is always
     ranked first, 0.5 if it's always second, and so on).
   - Answer: does the text the tool would hand the agent (its top-k chunks)
     actually contain the answer? The right article with the wrong section
     of it still leaves the agent unable to answer, so this is the stricter
     and more important number.
2. A threshold sweep: for each cosine cutoff (a chunk containing every
   query term is kept regardless, as the tool does), the share of answerable
   queries whose answer text still gets returned vs the share of unrelated
   queries that correctly get nothing back. Raising the cutoff trades the first for the
   second.
3. Tenant isolation: every query runs as one store's customer, and any
   chunk from another store in its results is a leak — the count must be 0.
   Some queries ask one store about a policy only the other store has
   (category "other_store"); the correct result for those is nothing.
4. Every query the current configuration gets wrong, with its top chunks.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_service.langchain.knowledge_base import (
    _RELEVANCE_THRESHOLD,
    _TOP_K,
    ScoredChunk,
    count_tokens,
    get_chunks,
    search_chunks,
)

_QUERIES_PATH = Path(__file__).parent / "kb_eval_queries.json"

# The embedding model's input window, including its 2 special tokens.
_MODEL_WINDOW = 256

Category = Literal["paraphrase", "identifier", "unrelated", "other_store"]


@dataclass(frozen=True)
class EvalQuery:
    query: str
    # The store whose customer is asking: search is scoped to it. The same
    # question can appear once per store, with a different correct answer.
    store: str
    category: Category
    # Every article that counts as a correct answer, as full article ids
    # ("<store>/<article>"); empty for an unrelated or other-store query,
    # where the only correct result is nothing at all.
    expected: tuple[str, ...]
    # An exact quote of the answer, from the first expected article. Used to
    # check the label is real, and to check the returned chunks contain it.
    evidence: str | None


def load_queries() -> list[EvalQuery]:
    raw = json.loads(_QUERIES_PATH.read_text(encoding="utf-8"))
    return [
        EvalQuery(
            query=item["query"],
            store=item["store"],
            category=item["category"],
            # Labels name articles within the query's store; the store prefix
            # is added here so a label can't point at another store's article.
            expected=tuple(f"{item['store']}/{article}" for article in item["expected"]),
            evidence=item.get("evidence"),
        )
        for item in raw
    ]


def _article_rank(ranking: list[ScoredChunk], expected: tuple[str, ...]) -> int | None:
    # Articles in order of their best chunk: the first time each article
    # shows up in the chunk ranking is its position.
    articles_seen: list[str] = []
    for scored in ranking:
        if scored.chunk.article_id not in articles_seen:
            articles_seen.append(scored.chunk.article_id)
            if scored.chunk.article_id in expected:
                return len(articles_seen)
    return None


def _answer_returned(chunks: list[ScoredChunk], q: EvalQuery) -> bool:
    return q.evidence is not None and any(
        s.chunk.article_id in q.expected and q.evidence in s.chunk.text for s in chunks
    )


def _returned(ranking: list[ScoredChunk], cutoff: float) -> list[ScoredChunk]:
    # What the tool hands the agent: the top-k chunks, minus any that are
    # neither above the cutoff nor a keyword match (knowledge_base.is_relevant,
    # with the cutoff as a parameter for the sweep).
    return [s for s in ranking[:_TOP_K] if s.similarity >= cutoff or s.keyword_match]


def _pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:5.1f}%" if denominator else "    -"


def main() -> None:
    chunks = get_chunks()
    queries = load_queries()
    chunks_per_store = Counter(c.store_id for c in chunks)
    # Each query ranks every chunk of its own store.
    rankings = {
        q: search_chunks(q.query, k=chunks_per_store[q.store], store_id=q.store) for q in queries
    }

    # --- Corpus -----------------------------------------------------------
    chunk_tokens = [count_tokens(c.embedding_text) for c in chunks]
    truncated = sum(1 for tokens in chunk_tokens if tokens > _MODEL_WINDOW - 2)
    for store, count in sorted(chunks_per_store.items()):
        articles = len({c.article_id for c in chunks if c.store_id == store})
        print(f"Store {store}: {articles} articles -> {count} chunks")
    print(
        f"  chunk length {min(chunk_tokens)}-{max(chunk_tokens)} tokens embedded; "
        f"{truncated}/{len(chunks)} truncated by the {_MODEL_WINDOW}-token window"
    )
    # A quote split across two chunks can never be "returned" whole, which
    # would wrongly count as a retrieval miss — flag it rather than hide it.
    split = [
        q.query
        for q in queries
        if q.evidence is not None
        and not any(c.article_id == q.expected[0] and q.evidence in c.text for c in chunks)
    ]
    print(f"  evidence quotes split across chunks: {len(split)}" + (f" {split}" if split else ""))
    print()

    # --- 1. Ranking quality -------------------------------------------------
    answerable = [q for q in queries if q.expected]
    groups: dict[str, list[EvalQuery]] = {"all answerable": answerable}
    for category in ("paraphrase", "identifier"):
        groups[category] = [q for q in answerable if q.category == category]
    for store in sorted(chunks_per_store):
        groups[store] = [q for q in answerable if q.store == store]

    print(f"Ranking quality (no threshold; 'answer@{_TOP_K}' = answer text in the top {_TOP_K} chunks)")
    print(f"  {'group':<20}{'n':>4}{'hit@1':>9}{'hit@3':>9}{'MRR':>8}{f'answer@{_TOP_K}':>11}")
    for name, group in groups.items():
        ranks = [_article_rank(rankings[q], q.expected) for q in group]
        hit1 = sum(1 for r in ranks if r is not None and r <= 1)
        hit3 = sum(1 for r in ranks if r is not None and r <= 3)
        mrr = sum(1 / r for r in ranks if r is not None) / len(ranks) if ranks else 0.0
        answered = sum(1 for q in group if _answer_returned(rankings[q][:_TOP_K], q))
        print(
            f"  {name:<20}{len(group):>4}{_pct(hit1, len(group)):>9}{_pct(hit3, len(group)):>9}"
            f"{mrr:>8.3f}{_pct(answered, len(group)):>11}"
        )
    returned_tokens = [
        sum(count_tokens(s.chunk.text) for s in _returned(rankings[q], _RELEVANCE_THRESHOLD))
        for q in answerable
    ]
    print(
        f"  avg text returned per answerable query at the current cutoff: "
        f"{sum(returned_tokens) / len(returned_tokens):.0f} tokens"
    )

    # --- 2. Threshold sweep ---------------------------------------------------
    unrelated = [q for q in queries if q.category == "unrelated"]
    other_store = [q for q in queries if q.category == "other_store"]
    print(f"\nThreshold sweep (top {_TOP_K} chunks, as the tool returns them)")
    print(f"  {'cutoff':<10}{'answer returned':>17}{'unrelated rejected':>20}{'other-store rejected':>22}")
    for step in range(20, 65, 5):
        cutoff = step / 100
        answered = sum(1 for q in answerable if _answer_returned(_returned(rankings[q], cutoff), q))
        rejected = sum(1 for q in unrelated if not _returned(rankings[q], cutoff))
        other_rejected = sum(1 for q in other_store if not _returned(rankings[q], cutoff))
        marker = "  <- current" if abs(cutoff - _RELEVANCE_THRESHOLD) < 1e-9 else ""
        print(
            f"  {cutoff:<10.2f}{_pct(answered, len(answerable)):>17}"
            f"{_pct(rejected, len(unrelated)):>20}{_pct(other_rejected, len(other_store)):>22}{marker}"
        )

    # --- 3. Tenant isolation ------------------------------------------------
    leaks = [(q, s.chunk.id) for q in queries for s in rankings[q] if s.chunk.store_id != q.store]
    print(
        f"\nTenant isolation: {len(leaks)} chunks from another store across all "
        f"{sum(len(r) for r in rankings.values())} ranked results"
    )
    for q, chunk_id in leaks[:10]:
        print(f'  LEAK: "{q.query}" as {q.store} got {chunk_id}')

    # --- 4. Failures at the current configuration -------------------------
    print(f"\nFailures at the current cutoff ({_RELEVANCE_THRESHOLD})")
    for q in queries:
        returned = _returned(rankings[q], _RELEVANCE_THRESHOLD)
        correct = _answer_returned(returned, q) if q.expected else not returned
        if correct:
            continue
        wanted = ", ".join(q.expected) or "nothing"
        print(f'  "{q.query}" as {q.store} ({q.category}) wanted: {wanted}')
        for s in rankings[q][:3]:
            has_answer = " <- has answer" if _answer_returned([s], q) else ""
            keyword = " [keyword match]" if s.keyword_match else ""
            print(f"      {s.similarity:.3f}  {s.chunk.id} ({s.chunk.context}){keyword}{has_answer}")


if __name__ == "__main__":
    main()
