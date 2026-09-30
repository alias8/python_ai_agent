"""Chunking logic only — a whitespace word count stands in for the model's
tokenizer, so no embedding model is loaded."""

from agent_service.langchain.chunking import chunk_article
from agent_service.langchain.kb_corpus import KBArticle


def _words(text: str) -> int:
    return len(text.split())


ARTICLE = KBArticle(
    id="widgets",
    store_id="store",
    title="Widgets",
    body=(
        "Intro paragraph about widgets.\n\n"
        "## Sizes\n\n"
        "Widgets come in three sizes.\n\n"
        "## Care\n\n"
        "one two three four five\n\n"
        "six seven eight nine ten\n\n"
        "eleven twelve"
    ),
)


def test_intro_and_each_section_become_chunks_with_their_heading() -> None:
    chunks = chunk_article(ARTICLE, max_tokens=100, count_tokens=_words)
    assert [(c.heading, c.text) for c in chunks] == [
        ("", "Intro paragraph about widgets."),
        ("Sizes", "Widgets come in three sizes."),
        ("Care", "one two three four five\n\nsix seven eight nine ten\n\neleven twelve"),
    ]
    assert [c.id for c in chunks] == ["widgets#0", "widgets#1", "widgets#2"]


def test_long_section_splits_between_paragraphs_under_the_limit() -> None:
    care = [c for c in chunk_article(ARTICLE, max_tokens=10, count_tokens=_words) if c.heading == "Care"]
    assert [c.text for c in care] == [
        "one two three four five\n\nsix seven eight nine ten",
        "eleven twelve",
    ]
    assert all(_words(c.text) <= 10 for c in care)


def test_embedding_text_carries_title_and_heading_but_returned_text_does_not() -> None:
    sizes = chunk_article(ARTICLE, max_tokens=100, count_tokens=_words)[1]
    assert sizes.embedding_text == "Widgets — Sizes\n\nWidgets come in three sizes."
    assert sizes.text == "Widgets come in three sizes."
    intro = chunk_article(ARTICLE, max_tokens=100, count_tokens=_words)[0]
    assert intro.context == "Widgets"
