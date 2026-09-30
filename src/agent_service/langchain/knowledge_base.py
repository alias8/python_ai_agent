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

from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from ..shared.tools_core import SearchKnowledgeBaseInput

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Below this cosine similarity, the "best match" is really just the
# least-bad article in the collection, not something actually relevant to
# the query — real retrieval needs a relevance cutoff, not just "return the
# top result no matter what". Picked empirically, not guessed: genuine
# paraphrase matches against these three articles ("how do I send something back"
# -> Return policy, "when will my package arrive" -> Shipping times, "I
# forgot my password" -> Account recovery) scored 0.46-0.71, while a
# genuinely unrelated query ("spaceship warranty") topped out at 0.25 —
# 0.35 sits cleanly between the two.
_RELEVANCE_THRESHOLD = 0.35

_KB_ARTICLES: list[tuple[str, str]] = [
    ("Return policy", "Items can be returned within 30 days of delivery for a full refund."),
    ("Shipping times", "Standard shipping takes 3-5 business days; expedited takes 1-2."),
    ("Account recovery", "Use the 'forgot password' link on the login page to reset your account."),
]


@lru_cache
def _get_model() -> SentenceTransformer:
    # Loading the model is the expensive part (a few seconds) — cached so
    # it happens once per process, not once per search.
    return SentenceTransformer(_EMBEDDING_MODEL_NAME)

# Turns article text into vectors that will be used for searching/comparing to the user question
@lru_cache
def _get_collection() -> Collection:
    # it reads the ~90MB of weights from ~/.cache/huggingface into memory.
    # The result is a PyTorch neural net that turns text into a list of 384 numbers
    model = _get_model()
    client = chromadb.EphemeralClient() # creates an in-memory Chroma database
    # collection is roughly a table
    collection = client.get_or_create_collection("kb_articles", metadata={"hnsw:space": "cosine"})
    titles = [title for title, _ in _KB_ARTICLES]
    bodies = [body for _, body in _KB_ARTICLES]
    # one row per article. Article turned into vector of length 384.
    # Unused — kept only to inspect in the debugger when stepping through.
    bodies_encoded = model.encode(bodies).tolist()  # noqa: F841
    collection.add(ids=titles, documents=bodies, embeddings=model.encode(bodies).tolist())
    return collection


def search_knowledge_base_semantic(input: SearchKnowledgeBaseInput) -> str:
    collection = _get_collection() # built once at first call
    query_embedding = _get_model().encode([input.query]).tolist()
    result = collection.query(query_embeddings=query_embedding, n_results=len(_KB_ARTICLES))

    ids = result["ids"][0]
    documents = result["documents"][0] if result["documents"] is not None else []
    distances = result["distances"][0] if result["distances"] is not None else []

    matches = [
        (title, body)
        for title, body, distance in zip(ids, documents, distances, strict=True)
        if (1 - distance) >= _RELEVANCE_THRESHOLD
    ]
    if not matches:
        return f'No knowledge base articles matched "{input.query}"'
    return "; ".join(f"{title}: {body}" for title, body in matches)
