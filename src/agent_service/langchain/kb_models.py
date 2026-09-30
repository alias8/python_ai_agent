"""The knowledge base's Postgres table: one row per chunk, with its embedding."""

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Index, String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from ..shared.db import Base

EMBEDDING_DIMENSIONS = 384  # all-MiniLM-L6-v2's output size

# What keyword search matches against: the same text the embedding sees.
SEARCH_VECTOR_SQL = "to_tsvector('english', article_title || ' ' || heading || ' ' || text)"


class KBChunkRow(Base):
    __tablename__ = "kb_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # "<article id>#<n>"
    # The tenant. Every search filters on it — see knowledge_base.search_chunks.
    store_id: Mapped[str] = mapped_column(String, index=True)
    article_id: Mapped[str] = mapped_column(String, index=True)
    article_title: Mapped[str] = mapped_column(Text)
    heading: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    # Which model produced `embedding`. Vectors from different models live in
    # different spaces and can't be compared, so search refuses to run
    # against rows embedded by a model other than the one embedding the query.
    embedding_model: Mapped[str] = mapped_column(String)
    # sha256 of what was embedded (the model name + the chunk's embedding
    # text). Ingestion skips re-embedding any chunk whose hash is unchanged.
    content_hash: Mapped[str] = mapped_column(String(64))
    # Postgres full-text search terms for the keyword half of hybrid search.
    # A generated column: Postgres computes it from the columns above on
    # every insert/update, so ingestion never has to keep it in sync.
    search_vector: Mapped[Any] = mapped_column(TSVECTOR, Computed(SEARCH_VECTOR_SQL, persisted=True))

    __table_args__ = (
        # HNSW: approximate nearest-neighbour index for cosine distance (<=>).
        # Without it, every search is a full scan comparing the query to
        # every row — fine for 109 chunks, not for 10 million.
        Index(
            "kb_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # GIN: an inverted index (term -> rows), what makes `search_vector @@
        # query` a lookup instead of a full scan.
        Index("kb_chunks_search_vector_gin", "search_vector", postgresql_using="gin"),
    )
