"""Semantic search over the support knowledge base — LangChain-only.

Embedding-based retrieval, replacing the project's original literal
substring/keyword match: article chunks (see chunking.py) and the query are
both embedded with a local sentence-transformers model, and matches are
found by vector similarity in a ChromaDB collection, not literal word
overlap — so "how do I send something back" finds the returns article with
no words in common with it. Retrieval quality is measured by
evals/kb_eval.py.

Runs fully locally: sentence-transformers downloads its model (~90MB, once,
cached under ~/.cache/huggingface) the first time it's used; both the
embedding model and ChromaDB run in-process after that — no new API key, no
network call once the model is cached.
"""

from dataclasses import dataclass
from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from ..shared.tools_core import SearchKnowledgeBaseInput
from .chunking import Chunk, chunk_article
from .kb_corpus import load_articles

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

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
    similarity: float  # cosine similarity, 1 - Chroma's cosine distance


@lru_cache
def _get_model() -> SentenceTransformer:
    # Loading the model is the expensive part (a few seconds) — cached so
    # it happens once per process, not once per search.
    return SentenceTransformer(_EMBEDDING_MODEL_NAME)


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


@lru_cache
def _chunks_by_id() -> dict[str, Chunk]:
    return {chunk.id: chunk for chunk in get_chunks()}


# Turns chunk text into vectors that will be used for searching/comparing to the user question
@lru_cache
def _get_collection() -> Collection:
    # it reads the ~90MB of weights from ~/.cache/huggingface into memory.
    # The result is a PyTorch neural net that turns text into a list of 384 floats from -1 to +1
    model = _get_model()
    client = chromadb.EphemeralClient() # creates an in-memory Chroma database
    # collection is roughly a table
    collection = client.get_or_create_collection("kb_chunks", metadata={"hnsw:space": "cosine"})
    chunks = get_chunks()
    ids = [chunk.id for chunk in chunks]
    bodies = [chunk.embedding_text for chunk in chunks]
    # one row per chunk. Chunk turned into vector of length 384.
    # Unused — kept only to inspect in the debugger when stepping through.
    bodies_encoded = model.encode(bodies).tolist()  # noqa: F841
    collection.add(ids=ids, documents=bodies, embeddings=model.encode(bodies).tolist())
    return collection


def search_chunks(query: str, k: int) -> list[ScoredChunk]:
    """Top-k chunks by similarity, best first, with no relevance cutoff
    applied — the raw ranking, which evals/kb_eval.py scores directly."""
    collection = _get_collection() # built once at first call
    query_embedding = _get_model().encode([query]).tolist()
    result = collection.query(query_embeddings=query_embedding, n_results=k)

    ids = result["ids"][0]
    distances = result["distances"][0] if result["distances"] is not None else []
    by_id = _chunks_by_id()
    return [
        ScoredChunk(chunk=by_id[chunk_id], similarity=1 - distance)
        for chunk_id, distance in zip(ids, distances, strict=True)
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
