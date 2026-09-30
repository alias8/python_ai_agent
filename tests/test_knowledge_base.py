"""Exercises the real embedding model + ChromaDB collection, not a mock —
the whole point of this module is retrieval quality (does it actually find
a paraphrase with no shared keywords, does it actually reject an unrelated
query), which a mocked encoder couldn't demonstrate. First run downloads
~90MB from Hugging Face Hub (cached under ~/.cache/huggingface after that);
everything after model load is local, no ANTHROPIC_API_KEY needed."""

from agent_service.langchain.knowledge_base import search_knowledge_base_semantic
from agent_service.shared.tools_core import SearchKnowledgeBaseInput


def test_finds_paraphrase_with_no_shared_keywords() -> None:
    # Zero literal word overlap with the "Return policy" article's text —
    # only a keyword-blind search can find this.
    result = search_knowledge_base_semantic(SearchKnowledgeBaseInput(query="how do I send something back"))
    assert "Return policy" in result
    assert "30 days" in result


def test_finds_shipping_paraphrase() -> None:
    result = search_knowledge_base_semantic(
        SearchKnowledgeBaseInput(query="when will my package arrive")
    )
    assert "Shipping times" in result


def test_unrelated_query_returns_no_match() -> None:
    result = search_knowledge_base_semantic(SearchKnowledgeBaseInput(query="spaceship warranty"))
    assert result.startswith("No knowledge base articles matched")
