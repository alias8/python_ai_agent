"""Semantic search over the support knowledge base — LangChain-only.

Upgrades shared/tools_core.py's search_knowledge_base (a literal substring/
keyword match) to real embedding-based retrieval: articles and the query are
both embedded with a local sentence-transformers model, and matches are
found by vector similarity in a ChromaDB collection, not literal word
overlap. Kept out of shared/ on purpose — direct_api keeps the original
keyword version, so the two implementations now differ in retrieval
strategy, not just loop mechanics. That's its own comparison worth having:
see the README for a query ("how do I send something back") that only the semantic
version answers correctly, having no words in common with the article that
answers it.

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
from .kb_corpus import KBArticle, load_articles

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Below this cosine similarity, the "best match" is really just the
# least-bad article in the collection, not something actually relevant to
# the query — real retrieval needs a relevance cutoff, not just "return the
# top result no matter what".
#
# 0.35 was tuned on the original three one-sentence articles, where real
# matches scored 0.46-0.71 and an unrelated query topped out at 0.25. It's
# too strict for the current corpus: a whole ~500-token article squeezed into
# one vector is diluted, so real matches score lower across the board, and
# `python -m evals.kb_eval` shows 0.35 now drops about a third of answerable
# queries. Deliberately not re-tuned yet — chunking (up next) changes the
# score distribution again, so the cutoff gets re-tuned after that, against
# the eval, rather than twice.
_RELEVANCE_THRESHOLD = 0.35

# How many articles the tool hands back to the agent at most. With a real-size
# knowledge base, "everything above the threshold" can be a lot of text for
# the LLM to wade through; a top-k cap keeps the tool result bounded.
_TOP_K = 3


@dataclass(frozen=True)
class ScoredArticle:
    article: KBArticle
    similarity: float  # cosine similarity, 1 - Chroma's cosine distance


@lru_cache
def _get_model() -> SentenceTransformer:
    # Loading the model is the expensive part (a few seconds) — cached so
    # it happens once per process, not once per search.
    return SentenceTransformer(_EMBEDDING_MODEL_NAME)

# Turns article text into vectors that will be used for searching/comparing to the user question
@lru_cache
def _get_collection() -> Collection:
    # it reads the ~90MB of weights from ~/.cache/huggingface into memory.
    # The result is a PyTorch neural net that turns text into a list of 384 floats from 0 to 1 
    model = _get_model()
    client = chromadb.EphemeralClient() # creates an in-memory Chroma database
    # collection is roughly a table
    collection = client.get_or_create_collection("kb_articles", metadata={"hnsw:space": "cosine"})
    articles = load_articles()
    ids = [article.id for article in articles]
    bodies = [article.body for article in articles]
    # one row per article. Article turned into vector of length 384.
    # Unused — kept only to inspect in the debugger when stepping through.
    bodies_encoded = model.encode(bodies).tolist()  # noqa: F841
    collection.add(ids=ids, documents=bodies, embeddings=model.encode(bodies).tolist())
    return collection


def rank_articles(query: str, k: int) -> list[ScoredArticle]:
    """Top-k articles by similarity, best first, with no relevance cutoff
    applied — the raw ranking, which evals/kb_eval.py scores directly."""
    collection = _get_collection() # built once at first call
    query_embedding = _get_model().encode([query]).tolist()
    result = collection.query(query_embeddings=query_embedding, n_results=k)

    ids = result["ids"][0]
    distances = result["distances"][0] if result["distances"] is not None else []
    by_id = {article.id: article for article in load_articles()}
    return [
        ScoredArticle(article=by_id[article_id], similarity=1 - distance)
        for article_id, distance in zip(ids, distances, strict=True)
    ]


def search_knowledge_base_semantic(input: SearchKnowledgeBaseInput) -> str:
    matches = [
        scored
        for scored in rank_articles(input.query, k=_TOP_K)
        if scored.similarity >= _RELEVANCE_THRESHOLD
    ]
    if not matches:
        return f'No knowledge base articles matched "{input.query}"'
    return "\n\n---\n\n".join(f"# {m.article.title}\n\n{m.article.body}" for m in matches)
