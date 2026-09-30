"""Semantic search over the support knowledge base — LangChain-only.

Embedding-based retrieval, replacing the project's original literal
substring/keyword match: article chunks (see chunking.py) and the query are
both embedded with a local sentence-transformers model, and matches are
found by vector similarity in Postgres (pgvector), not literal word
overlap — so "how do I send something back" finds the returns article with
no words in common with it. Retrieval quality is measured by
evals/kb_eval.py.

Chunks are embedded ahead of time by kb_ingest.py and stored in the
kb_chunks table; a search only embeds the query. The embedding model runs
locally: sentence-transformers downloads it (~90MB, once, cached under
~/.cache/huggingface) the first time it's used — no API key, no network
call once it's cached.
"""

from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sqlalchemy import func, select

from ..shared.db import new_session
from ..shared.tools_core import SearchKnowledgeBaseInput
from .chunking import Chunk, chunk_article
from .kb_corpus import load_articles
from .kb_models import KBChunkRow

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Upper bound on a chunk's own text. The model's window is 256 tokens
# including 2 special tokens, and the title/heading prefix adds up to ~20
# more — 200 leaves headroom so no chunk is ever truncated when embedded.
_MAX_CHUNK_TOKENS = 200

# Below this cosine similarity, the "best match" is really just the
# least-bad chunk in the collection, not something actually relevant to
# the query — real retrieval needs a relevance cutoff, not just "return the
# top result no matter what".
#
# Tuned from `python -m evals.kb_eval`'s threshold sweep after chunking. At
# 0.40, 80% of answerable queries get their answer text back while 8 of 9
# unrelated queries get nothing; 0.45 rejects all 9 but drops answer recall
# to 69%. The lower value wins because the agent is a second filter: it can
# ignore a loosely related chunk, but it can't answer from a chunk it never
# received. Caveat: tuned on the same 54 queries it's evaluated on, so the
# numbers are optimistic — a held-out split is the fix once the set grows.
# Re-tune after any change to the model, the corpus, or the chunking.
_RELEVANCE_THRESHOLD = 0.40

# How many chunks the tool hands back to the agent at most. Chunks are
# ~100-200 tokens, so 5 of them (~440 tokens on average) is still about a
# quarter of the text the old 3-whole-articles result sent, while catching
# answers that sit in a different section from the best-matching one.
_TOP_K = 5


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    similarity: float  # cosine similarity: 1 - pgvector's cosine distance (<=>)


@lru_cache
def _get_model() -> SentenceTransformer:
    # Loading the model is the expensive part (a few seconds) — cached so
    # it happens once per process, not once per search.
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def count_tokens(text: str) -> int:
    """Tokens as the embedding model counts them, excluding its 2 special tokens."""
    return len(_get_model().tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])


@lru_cache
def get_chunks() -> tuple[Chunk, ...]:
    return tuple(
        chunk
        for article in load_articles()
        for chunk in chunk_article(article, _MAX_CHUNK_TOKENS, count_tokens)
    )


# Turns chunk text into vectors that will be used for searching/comparing to the user question
def embed_texts(texts: list[str]) -> list[list[float]]:
    # it reads the ~90MB of weights from ~/.cache/huggingface into memory.
    # The result is a PyTorch neural net that turns text into a list of 384 floats from -1 to +1
    model = _get_model()
    # one row per chunk. Chunk turned into vector of length 384.
    bodies_encoded: list[list[float]] = model.encode(texts).tolist()
    return bodies_encoded


@lru_cache
def _check_index_ready() -> None:
    # Fails loudly instead of letting search quietly return "no matches":
    # an empty table (ingestion never run) or vectors from a different
    # embedding model (model changed, not re-ingested) would both look like
    # "nothing relevant" to the agent. Cached on success only — lru_cache
    # doesn't cache an exception, so a fixed database is picked up next call.
    with new_session() as session:
        total = session.scalar(select(func.count()).select_from(KBChunkRow)) or 0
        other_model = session.scalar(
            select(func.count()).where(KBChunkRow.embedding_model != EMBEDDING_MODEL_NAME)
        ) or 0
    fix = "run `python -m agent_service.langchain.kb_ingest`"
    if total == 0:
        raise RuntimeError(f"knowledge base is empty — {fix}")
    if other_model:
        raise RuntimeError(
            f"{other_model} knowledge base chunks were embedded by a model other than "
            f"{EMBEDDING_MODEL_NAME}, so they can't be compared to query vectors — {fix}"
        )


# HNSW's search-time candidate list size (pgvector's default is 40). An HNSW
# index scan returns at most this many rows, however high the LIMIT — so the
# eval, which ranks every chunk, would silently get only 40 back. Raised to k
# when a caller asks for more; pgvector's max is 1000. (At today's 109 rows
# the planner skips the index for a cheaper full scan + sort, so the cap
# doesn't bite yet — verified by forcing the index: 40 rows at the default,
# 109 with ef_search raised. It would once the table grows.)
_MIN_EF_SEARCH = 40
_MAX_EF_SEARCH = 1000


def search_chunks(query: str, k: int) -> list[ScoredChunk]:
    """Top-k chunks by similarity, best first, with no relevance cutoff
    applied — the raw ranking, which evals/kb_eval.py scores directly."""
    _check_index_ready()
    query_vector = _get_model().encode(query).tolist()
    distance = KBChunkRow.embedding.cosine_distance(query_vector)
    ef_search = min(max(k, _MIN_EF_SEARCH), _MAX_EF_SEARCH)

    with new_session() as session:
        # set_config(..., true) is SET LOCAL: it lasts only for this
        # transaction, so it never leaks to other users of the pooled connection.
        session.execute(select(func.set_config("hnsw.ef_search", str(ef_search), True)))
        rows = session.execute(select(KBChunkRow, distance.label("distance")).order_by(distance).limit(k))
        return [
            ScoredChunk(
                chunk=Chunk(
                    id=row.id,
                    article_id=row.article_id,
                    article_title=row.article_title,
                    heading=row.heading,
                    text=row.text,
                ),
                similarity=1 - dist,
            )
            for row, dist in rows
        ]


def search_knowledge_base_semantic(input: SearchKnowledgeBaseInput) -> str:
    matches = [
        scored
        for scored in search_chunks(input.query, k=_TOP_K)
        if scored.similarity >= _RELEVANCE_THRESHOLD
    ]
    if not matches:
        return f'No knowledge base articles matched "{input.query}"'
    return "\n\n---\n\n".join(f"# {m.chunk.context}\n\n{m.chunk.text}" for m in matches)
