"""Exercises the real embedding model + a real Postgres (pgvector), not mocks —
the whole point of this module is retrieval quality (does it actually find
a paraphrase with no shared keywords, does it actually reject an unrelated
query), which a mocked encoder couldn't demonstrate. First run downloads
~90MB from Hugging Face Hub (cached under ~/.cache/huggingface after that);
everything after model load is local, no ANTHROPIC_API_KEY needed.

These are smoke tests for the tool's behavior; retrieval quality across the
whole labeled query set is measured by evals/kb_eval.py, not asserted here."""

import pytest

from agent_service.langchain import knowledge_base
from agent_service.langchain.knowledge_base import (
    _RELEVANCE_THRESHOLD,
    _TOP_K,
    count_tokens,
    get_chunks,
    search_chunks,
    search_knowledge_base_semantic,
)
from agent_service.shared.tools_core import SearchKnowledgeBaseInput

pytestmark = pytest.mark.usefixtures("kb_database")

MSM = "main-street-market"


def _search(query: str) -> str:
    return search_knowledge_base_semantic(SearchKnowledgeBaseInput(query=query), MSM)


def test_finds_paraphrase_with_no_shared_keywords() -> None:
    # Never says "return" or "refund"; the only word it shares with the
    # returns article's top-matching chunk is "can".
    result = _search("can I get my money back")
    assert "# Returns and refunds" in result
    assert "30 days" in result


def test_finds_shipping_paraphrase() -> None:
    result = _search("when will my package arrive")
    assert "# Shipping options and delivery times" in result


def test_returns_at_most_top_k_chunks() -> None:
    result = _search("when will my package arrive")
    assert result.count("\n---\n") <= _TOP_K - 1


def test_result_is_chunks_under_their_section_heading_not_whole_articles() -> None:
    result = _search("it says my account is locked")
    assert "# Password reset and account lockout — Account lockout" in result
    article_chunks = [c for c in get_chunks() if c.article_id == f"{MSM}/password-reset-and-account-lockout"]
    returned = [c for c in article_chunks if f"# {c.context}\n\n{c.text}" in result]
    # Some of the article's sections, not the whole article.
    assert 0 < len(returned) < len(article_chunks)


def test_unrelated_query_returns_no_match() -> None:
    # Not "spaceship warranty": with a warranties article in the knowledge
    # base, that query matches it — it's no longer unrelated.
    result = _search("recipe for banana bread")
    assert result.startswith("No knowledge base articles matched")


def test_no_chunk_is_truncated_by_the_model_window() -> None:
    # The reason chunking exists: all-MiniLM-L6-v2 silently drops anything
    # past 256 tokens (2 of them special tokens). Guards against an article
    # edit producing a paragraph too long to chunk under the limit.
    for chunk in get_chunks():
        assert count_tokens(chunk.embedding_text) <= 254, chunk.id


def test_search_refuses_vectors_from_a_different_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # As if the model had been swapped without re-running ingestion: the
    # stored vectors are no longer comparable to the query's.
    monkeypatch.setattr(knowledge_base, "EMBEDDING_MODEL_NAME", "some-other-model")
    knowledge_base._check_index_ready.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="embedded by a model other than some-other-model"):
            _search("refund")
    finally:
        knowledge_base._check_index_ready.cache_clear()


def test_exact_identifier_is_kept_by_keyword_match_despite_a_low_embedding_score() -> None:
    # A bare error code embeds poorly: the right chunk ranks first but scores
    # under the cosine cutoff. Hybrid search keeps it because the chunk
    # contains the query's exact terms.
    top = search_chunks("E-6002", k=1, store_id=MSM)[0]
    assert top.chunk.id.startswith(f"{MSM}/gift-cards-and-promo-codes#")
    assert top.similarity < _RELEVANCE_THRESHOLD
    assert top.keyword_match
    result = _search("E-6002")
    assert "E-6002: Minimum not met." in result


def test_identifier_not_in_the_knowledge_base_matches_nothing() -> None:
    # Order numbers belong to lookup_order_status, not the knowledge base;
    # an identifier-shaped query mustn't pull in chunks on shared words alone.
    result = _search("where is order ORD-1042")
    assert result.startswith("No knowledge base articles matched")


def test_all_stopword_query_skips_keyword_search_without_error() -> None:
    assert all(not s.keyword_match for s in search_chunks("how do I", k=_TOP_K, store_id=MSM))
