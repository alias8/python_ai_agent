"""The knowledge base's Postgres table: one row per chunk, with its embedding."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..shared.db import Base

EMBEDDING_DIMENSIONS = 384  # all-MiniLM-L6-v2's output size


class KBChunkRow(Base):
    __tablename__ = "kb_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # "<article id>#<n>"
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
    )
