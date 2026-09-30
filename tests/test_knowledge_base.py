"""Exercises the real embedding model + ChromaDB collection, not a mock —
the whole point of this module is retrieval quality (does it actually find
a paraphrase with no shared keywords, does it actually reject an unrelated
query), which a mocked encoder couldn't demonstrate. First run downloads
~90MB from Hugging Face Hub (cached under ~/.cache/huggingface after that);
everything after model load is local, no ANTHROPIC_API_KEY needed.

These are smoke tests for the tool's behavior; retrieval quality across the
whole labeled query set is measured by evals/kb_eval.py, not asserted here."""

from agent_service.langchain.knowledge_base import (
    _TOP_K,
    count_tokens,
    get_chunks,
    search_knowledge_base_semantic,
)
from agent_service.shared.tools_core import SearchKnowledgeBaseInput


def test_finds_paraphrase_with_no_shared_keywords() -> None:
    # Never says "return" or "refund"; the only word it shares with the
    # returns article's top-matching chunk is "can".
    result = search_knowledge_base_semantic(SearchKnowledgeBaseInput(query="can I get my money back"))
    assert "# Returns and refunds" in result
    assert "30 days" in result


def test_finds_shipping_paraphrase() -> None:
    result = search_knowledge_base_semantic(
        SearchKnowledgeBaseInput(query="when will my package arrive")
    )
    assert "# Shipping options and delivery times" in result


def test_returns_at_most_top_k_chunks() -> None:
    result = search_knowledge_base_semantic(SearchKnowledgeBaseInput(query="when will my package arrive"))
    assert result.count("\n---\n") <= _TOP_K - 1


def test_result_is_chunks_under_their_section_heading_not_whole_articles() -> None:
    result = search_knowledge_base_semantic(SearchKnowledgeBaseInput(query="it says my account is locked"))
    assert "# Password reset and account lockout — Account lockout" in result
    article_chunks = [c for c in get_chunks() if c.article_id == "password-reset-and-account-lockout"]
    returned = [c for c in article_chunks if f"# {c.context}\n\n{c.text}" in result]
    # Some of the article's sections, not the whole article.
    assert 0 < len(returned) < len(article_chunks)


def test_unrelated_query_returns_no_match() -> None:
    # Not "spaceship warranty": with a warranties article in the knowledge
    # base, that query matches it — it's no longer unrelated.
    result = search_knowledge_base_semantic(SearchKnowledgeBaseInput(query="recipe for banana bread"))
    assert result.startswith("No knowledge base articles matched")


def test_no_chunk_is_truncated_by_the_model_window() -> None:
    # The reason chunking exists: all-MiniLM-L6-v2 silently drops anything
    # past 256 tokens (2 of them special tokens). Guards against an article
    # edit producing a paragraph too long to chunk under the limit.
    for chunk in get_chunks():
        assert count_tokens(chunk.embedding_text) <= 254, chunk.id
