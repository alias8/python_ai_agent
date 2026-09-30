"""Loads the support knowledge base from kb_articles/<store id>/, one
Markdown file per article.

The knowledge base is multi-tenant: each store has its own articles, and
its policies can contradict another store's (one store's return window is
30 days, another's is 14). An article belongs to exactly one store, and
search only ever looks at the current store's articles.

Articles are realistic help-center length (roughly 250-450 words each), not
one-liners — long enough that most of them run past all-MiniLM-L6-v2's
256-token input window, which the embedding model silently truncates. That's
deliberate: it's the problem chunking has to solve, and evals/kb_eval.py
measures how many queries it costs.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_ARTICLES_DIR = Path(__file__).parent / "kb_articles"


@dataclass(frozen=True)
class KBArticle:
    # "<store id>/<file stem>", e.g. "main-street-market/returns-and-refunds"
    # — stable, unlike the title, and unique across stores (two stores can
    # both have a "returns" article).
    id: str
    store_id: str
    title: str
    body: str


def _parse_article(path: Path) -> KBArticle:
    first_line, _, rest = path.read_text(encoding="utf-8").partition("\n")
    if not first_line.startswith("# "):
        raise ValueError(f'{path.name}: first line must be a "# Title" heading')
    store_id = path.parent.name
    return KBArticle(
        id=f"{store_id}/{path.stem}",
        store_id=store_id,
        title=first_line.removeprefix("# ").strip(),
        body=rest.strip(),
    )


@lru_cache
def load_articles() -> tuple[KBArticle, ...]:
    # Sorted so chunks are ingested in the same order every run.
    return tuple(_parse_article(path) for path in sorted(_ARTICLES_DIR.glob("*/*.md")))


@lru_cache
def store_ids() -> frozenset[str]:
    return frozenset(article.store_id for article in load_articles())
