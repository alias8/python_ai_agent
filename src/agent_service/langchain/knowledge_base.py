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
from typing import Any

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
# A chunk that contains every term of the query (see _keyword_query) is kept
# even below this cutoff — that's hybrid search's job here, since exact
# identifiers like "E-6002" embed poorly and score just under it.
#
# Tuned from `python -m evals.kb_eval`'s threshold sweep. At 0.40 (with the
# keyword rule), 86% of answerable queries get their answer text back and 9
# of 10 unrelated queries get nothing; 0.45 rejects all 10 but drops answer
# recall to 79%. The lower value wins because the agent is a second filter:
# it can ignore a loosely related chunk, but it can't answer from a chunk it
# never received. Caveat: tuned on the same queries it's evaluated on, so the
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
    keyword_match: bool  # the chunk contains every term in the query


def is_relevant(scored: ScoredChunk) -> bool:
    return scored.similarity >= _RELEVANCE_THRESHOLD or scored.keyword_match


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


# Each side of hybrid search contributes its top candidates to the fusion.
_MIN_CANDIDATES = 50

# Reciprocal rank fusion's constant: a chunk's fused score is the sum over
# rankings of 1 / (_RRF_K + its rank there). 60 is the value from the paper
# that introduced RRF and the usual default; it damps the gap between rank 1
# and rank 2 so neither ranking can dominate on a single top result.
_RRF_K = 60


def _keyword_query(query: str) -> Any:
    # plainto_tsquery ANDs every term: a chunk matches only if it contains
    # all of them (after stemming and stopword removal). That's deliberate.
    # Matching ANY term (OR) let common words through — "what time does your
    # store open" matched chunks mentioning "time", and unrelated-query
    # rejection fell from 90% to 20% in evals/kb_eval.py. Requiring every
    # term keeps keyword matches rare and strong: they fire for exact
    # identifiers ("E-6002", "$99 freight fee"), where they're needed.
    return func.plainto_tsquery("english", query)


def _to_scored(row: KBChunkRow, distance: float, keyword_match: bool) -> ScoredChunk:
    chunk = Chunk(
        id=row.id,
        article_id=row.article_id,
        article_title=row.article_title,
        heading=row.heading,
        text=row.text,
    )
    return ScoredChunk(chunk=chunk, similarity=1 - distance, keyword_match=keyword_match)


def search_chunks(query: str, k: int) -> list[ScoredChunk]:
    """Top-k chunks, best first: vector search and keyword search fused with
    reciprocal rank fusion. No relevance cutoff is applied — that's
    is_relevant()'s job — so evals/kb_eval.py can score the raw ranking."""
    _check_index_ready()
    query_vector = _get_model().encode(query).tolist()
    distance = KBChunkRow.embedding.cosine_distance(query_vector)
    tsquery = _keyword_query(query)
    candidates = max(k, _MIN_CANDIDATES)
    ef_search = min(max(candidates, _MIN_EF_SEARCH), _MAX_EF_SEARCH)

    with new_session() as session:
        # set_config(..., true) is SET LOCAL: it lasts only for this
        # transaction, so it never leaks to other users of the pooled connection.
        session.execute(select(func.set_config("hnsw.ef_search", str(ef_search), True)))
        vector_rows = session.execute(
            select(KBChunkRow, distance).order_by(distance).limit(candidates)
        ).all()
        keyword_ids = session.scalars(
            select(KBChunkRow.id)
            # numnode = 0 means the query was all stopwords ("how do I"):
            # nothing to match, and Postgres would warn on every row.
            .where(func.numnode(tsquery) > 0, KBChunkRow.search_vector.op("@@")(tsquery))
            .order_by(func.ts_rank_cd(KBChunkRow.search_vector, tsquery).desc())
            .limit(candidates)
        ).all()

        rows = {row.id: (row, dist) for row, dist in vector_rows}
        # Keyword matches the vector side didn't rank highly enough to fetch.
        missing = [chunk_id for chunk_id in keyword_ids if chunk_id not in rows]
        if missing:
            for row, dist in session.execute(
                select(KBChunkRow, distance).where(KBChunkRow.id.in_(missing))
            ):
                rows[row.id] = (row, dist)

    fused: dict[str, float] = {}
    for ranking in ([row.id for row, _ in vector_rows], keyword_ids):
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (_RRF_K + rank)
    keyword_set = set(keyword_ids)
    ranked = sorted(fused, key=lambda chunk_id: fused[chunk_id], reverse=True)[:k]
    return [_to_scored(*rows[chunk_id], keyword_match=chunk_id in keyword_set) for chunk_id in ranked]


def search_knowledge_base_semantic(input: SearchKnowledgeBaseInput) -> str:
    matches = [scored for scored in search_chunks(input.query, k=_TOP_K) if is_relevant(scored)]
    if not matches:
        return f'No knowledge base articles matched "{input.query}"'
    return "\n\n---\n\n".join(f"# {m.chunk.context}\n\n{m.chunk.text}" for m in matches)
