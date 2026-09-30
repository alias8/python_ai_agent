"""Splits knowledge base articles into chunks small enough to embed whole.

all-MiniLM-L6-v2 only reads the first 256 tokens of its input and silently
drops the rest, and every article in the corpus is 465-580 tokens — so
embedding whole articles meant roughly half of each one was never seen.
Chunks are embedded instead, and search matches a query against chunks.

Chunking follows the articles' own structure rather than cutting fixed-size
token windows: each "## " section is a chunk (the intro before the first
heading is one too), and a section too long for one chunk is split between
paragraphs. Sections are self-contained units of meaning, so a chunk rarely
starts or ends mid-thought — evals/kb_eval.py measured fixed 128-token
windows (with overlap) as worse on both ranking and answer recall.

Each chunk is embedded with its article title and section heading prefixed
("Returns and refunds — Opened electronics"), because a section on its own
often doesn't say what it's about: "Electronics can be returned within the
same 30-day window..." never says "restocking fee policy". The prefix is
only for the embedding; the text handed back to the agent is the chunk
itself, under its heading.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from .kb_corpus import KBArticle

_SECTION_HEADING = re.compile(r"^## (.+)$", flags=re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    id: str  # "<article id>#<n>", e.g. "returns-and-refunds#3"
    article_id: str
    article_title: str
    heading: str  # the section heading, or "" for the article's intro
    text: str

    @property
    def context(self) -> str:
        return f"{self.article_title} — {self.heading}" if self.heading else self.article_title

    @property
    def embedding_text(self) -> str:
        return f"{self.context}\n\n{self.text}"


def _sections(body: str) -> list[tuple[str, str]]:
    # re.split with a capture group alternates [intro, heading, text, heading, text, ...]
    parts = _SECTION_HEADING.split(body)
    sections = [("", parts[0].strip())] if parts[0].strip() else []
    for i in range(1, len(parts), 2):
        sections.append((parts[i].strip(), parts[i + 1].strip()))
    return sections


def _pack_paragraphs(text: str, max_tokens: int, count_tokens: Callable[[str], int]) -> list[str]:
    # Greedily packs whole paragraphs into pieces of at most max_tokens. A
    # single paragraph longer than max_tokens becomes its own piece and would
    # be truncated when embedded — none in the current corpus come close, and
    # tests/test_knowledge_base.py checks that stays true.
    pieces: list[str] = []
    current = ""
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if current and count_tokens(candidate) > max_tokens:
            pieces.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_article(
    article: KBArticle, max_tokens: int, count_tokens: Callable[[str], int]
) -> list[Chunk]:
    """count_tokens is passed in, not looked up, so chunking can be tested
    without loading the embedding model's tokenizer."""
    chunks: list[Chunk] = []
    for heading, text in _sections(article.body):
        for piece in _pack_paragraphs(text, max_tokens, count_tokens):
            chunks.append(
                Chunk(
                    id=f"{article.id}#{len(chunks)}",
                    article_id=article.id,
                    article_title=article.title,
                    heading=heading,
                    text=piece,
                )
            )
    return chunks
