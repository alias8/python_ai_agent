"""Exercises the real embedding model + ChromaDB collection, not a mock —
the whole point of this module is retrieval quality (does it actually find
a paraphrase with no shared keywords, does it actually reject an unrelated
query), which a mocked encoder couldn't demonstrate. First run downloads
~90MB from Hugging Face Hub (cached under ~/.cache/huggingface after that);
everything after model load is local, no ANTHROPIC_API_KEY needed.

These are smoke tests for the tool's behavior; retrieval quality across the
whole labeled query set is measured by evals/kb_eval.py, not asserted here."""

from agent_service.langchain.knowledge_base import search_knowledge_base_semantic
from agent_service.shared.tools_core import SearchKnowledgeBaseInput


def test_finds_paraphrase_with_no_shared_keywords() -> None:
    # Zero literal word overlap with the "Returns and refunds" article's
    # opening — only a keyword-blind search can find this.
    result = search_knowledge_base_semantic(SearchKnowledgeBaseInput(query="how do I send something back"))
    assert "# Returns and refunds" in result
    assert "30 days" in result


def test_finds_shipping_paraphrase() -> None:
    result = search_knowledge_base_semantic(
        SearchKnowledgeBaseInput(query="when will my package arrive")
    )
    assert "# Shipping options and delivery times" in result


def test_returns_at_most_top_k_articles() -> None:
    result = search_knowledge_base_semantic(SearchKnowledgeBaseInput(query="when will my package arrive"))
    assert result.count("\n---\n") <= 2


def test_unrelated_query_returns_no_match() -> None:
    # Not "spaceship warranty" any more: with a warranties article in the
    # knowledge base, that query scores 0.46 against it — no longer unrelated.
    result = search_knowledge_base_semantic(SearchKnowledgeBaseInput(query="recipe for banana bread"))
    assert result.startswith("No knowledge base articles matched")
