"""Retrieval eval for langchain/knowledge_base.py's semantic search.

Scores the real embedding model + ChromaDB collection against the labeled
queries in kb_eval_queries.json, so a change to retrieval (chunking, a new
embedding model, hybrid search, a different threshold) is judged by numbers
rather than by eyeballing a couple of queries.

Run from the repo root:

    python -m evals.kb_eval

Reports three things:

1. Ranking quality, ignoring the threshold: hit@1, hit@3, and MRR (mean
   reciprocal rank — 1.0 if the right article is always ranked first, 0.5 if
   it's always second, and so on). Broken down by query category, and by
   whether the query's evidence sits inside or past the embedding model's
   input window — text past the window is silently truncated, never
   embedded, so retrieval for those queries can only succeed by luck.
2. A threshold sweep: for each cutoff, the share of answerable queries whose
   answer still gets returned (recall) vs the share of unrelated queries that
   correctly get nothing back (rejection). Raising the cutoff trades the
   first for the second.
3. Every query the current configuration gets wrong, with its top 3 scores.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_service.langchain.kb_corpus import KBArticle, load_articles
from agent_service.langchain.knowledge_base import (
    _RELEVANCE_THRESHOLD,
    _TOP_K,
    ScoredArticle,
    _get_model,
    rank_articles,
)

_QUERIES_PATH = Path(__file__).parent / "kb_eval_queries.json"

Category = Literal["paraphrase", "identifier", "unrelated"]


@dataclass(frozen=True)
class EvalQuery:
    query: str
    category: Category
    # Every article that counts as a correct answer; empty for an unrelated
    # query, where the only correct result is nothing at all.
    expected: tuple[str, ...]
    # An exact quote of the answer, from the first expected article. Used to
    # check the label is real, and to locate the answer relative to the
    # model's input window.
    evidence: str | None


def load_queries() -> list[EvalQuery]:
    raw = json.loads(_QUERIES_PATH.read_text(encoding="utf-8"))
    return [
        EvalQuery(
            query=item["query"],
            category=item["category"],
            expected=tuple(item["expected"]),
            evidence=item.get("evidence"),
        )
        for item in raw
    ]


def evidence_token_offset(article: KBArticle, evidence: str) -> int:
    """How many tokens into the embedded text the evidence starts, counting
    the [CLS] token the model prepends."""
    start = article.body.index(evidence)
    tokenizer = _get_model().tokenizer
    return 1 + len(tokenizer(article.body[:start], add_special_tokens=False, verbose=False)["input_ids"])


def _window_size() -> int:
    # Includes the [CLS] and [SEP] special tokens, so usable text is 2 fewer.
    return int(_get_model().max_seq_length)


def _rank_of_first_expected(ranking: list[ScoredArticle], expected: tuple[str, ...]) -> int | None:
    for position, scored in enumerate(ranking, start=1):
        if scored.article.id in expected:
            return position
    return None


def _pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:5.1f}%" if denominator else "    -"


def main() -> None:
    articles = {article.id: article for article in load_articles()}
    queries = load_queries()
    window = _window_size()
    rankings = {q.query: rank_articles(q.query, k=len(articles)) for q in queries}

    # --- Corpus -----------------------------------------------------------
    tokenizer = _get_model().tokenizer
    article_tokens = {
        article.id: len(tokenizer(article.body, add_special_tokens=False, verbose=False)["input_ids"])
        for article in articles.values()
    }
    truncated = sum(1 for tokens in article_tokens.values() if tokens > window - 2)
    print(f"Corpus: {len(articles)} articles, model window {window} tokens")
    print(
        f"  article length {min(article_tokens.values())}-{max(article_tokens.values())} tokens; "
        f"{truncated}/{len(articles)} truncated when embedded\n"
    )

    # --- 1. Ranking quality -------------------------------------------------
    answerable = [q for q in queries if q.expected]
    groups: dict[str, list[EvalQuery]] = {"all answerable": answerable}
    for category in ("paraphrase", "identifier"):
        groups[category] = [q for q in answerable if q.category == category]
    in_window: list[EvalQuery] = []
    past_window: list[EvalQuery] = []
    for q in answerable:
        assert q.evidence is not None
        offset = evidence_token_offset(articles[q.expected[0]], q.evidence)
        (in_window if offset < window - 1 else past_window).append(q)
    groups["evidence inside window"] = in_window
    groups["evidence past window"] = past_window

    print("Ranking quality (no threshold)")
    print(f"  {'group':<24}{'n':>4}{'hit@1':>9}{'hit@3':>9}{'MRR':>8}")
    for name, group in groups.items():
        ranks = [_rank_of_first_expected(rankings[q.query], q.expected) for q in group]
        hit1 = sum(1 for r in ranks if r is not None and r <= 1)
        hit3 = sum(1 for r in ranks if r is not None and r <= 3)
        mrr = sum(1 / r for r in ranks if r is not None) / len(ranks) if ranks else 0.0
        print(f"  {name:<24}{len(group):>4}{_pct(hit1, len(group)):>9}{_pct(hit3, len(group)):>9}{mrr:>8.3f}")

    # --- 2. Threshold sweep ---------------------------------------------------
    unrelated = [q for q in queries if not q.expected]
    print(f"\nThreshold sweep (top {_TOP_K}, as the tool returns them)")
    print(f"  {'cutoff':<10}{'answer recall':>15}{'unrelated rejected':>20}")
    for step in range(20, 65, 5):
        cutoff = step / 100
        recalled = 0
        for q in answerable:
            returned = [s for s in rankings[q.query][:_TOP_K] if s.similarity >= cutoff]
            recalled += any(s.article.id in q.expected for s in returned)
        rejected = sum(1 for q in unrelated if rankings[q.query][0].similarity < cutoff)
        marker = "  <- current" if abs(cutoff - _RELEVANCE_THRESHOLD) < 1e-9 else ""
        print(
            f"  {cutoff:<10.2f}{_pct(recalled, len(answerable)):>15}"
            f"{_pct(rejected, len(unrelated)):>20}{marker}"
        )

    # --- 3. Failures at the current configuration -------------------------
    print(f"\nFailures at the current cutoff ({_RELEVANCE_THRESHOLD})")
    for q in queries:
        top = rankings[q.query][:_TOP_K]
        returned = [s for s in top if s.similarity >= _RELEVANCE_THRESHOLD]
        correct = (
            any(s.article.id in q.expected for s in returned) if q.expected else not returned
        )
        if correct:
            continue
        where = ""
        if q.expected and q.evidence is not None:
            where = " [evidence past window]" if q in past_window else " [evidence inside window]"
        wanted = ", ".join(q.expected) or "nothing"
        print(f'  "{q.query}" ({q.category}) wanted: {wanted}{where}')
        for s in top:
            print(f"      {s.similarity:.3f}  {s.article.id}")


if __name__ == "__main__":
    main()
