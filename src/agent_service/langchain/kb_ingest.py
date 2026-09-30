"""Embeds the knowledge base's chunks and writes them to Postgres.

    python -m agent_service.langchain.kb_ingest

Run it after `alembic upgrade head`, and again whenever an article in
kb_articles/ changes. It's incremental: each row stores a hash of what was
embedded (the model name plus the chunk's embedding text), so an unchanged
chunk is left alone, an edited one is re-embedded, and a chunk that no
longer exists — an article deleted or restructured — is deleted. Changing
the embedding model changes every hash, so every chunk is re-embedded.

Search never embeds articles itself: embedding is the slow, batchable part,
and doing it at ingestion time keeps it off the request path.
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..shared.db import new_session
from .chunking import Chunk
from .kb_models import KBChunkRow
from .knowledge_base import EMBEDDING_MODEL_NAME, embed_texts, get_chunks

Embedder = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class IngestReport:
    embedded: int
    unchanged: int
    deleted: int


def content_hash(chunk: Chunk, model_name: str) -> str:
    return hashlib.sha256(f"{model_name}\n{chunk.embedding_text}".encode()).hexdigest()


def ingest(
    session: Session, chunks: Sequence[Chunk], embed: Embedder, model_name: str
) -> IngestReport:
    """Syncs kb_chunks to `chunks`. Doesn't commit — the caller owns the
    transaction, so a failure part-way (e.g. while embedding) leaves the
    table as it was rather than half-updated."""
    existing = dict(session.execute(select(KBChunkRow.id, KBChunkRow.content_hash)).all())
    hashes = {chunk.id: content_hash(chunk, model_name) for chunk in chunks}

    stale_ids = existing.keys() - hashes.keys()
    if stale_ids:
        session.execute(delete(KBChunkRow).where(KBChunkRow.id.in_(stale_ids)))

    to_embed = [chunk for chunk in chunks if existing.get(chunk.id) != hashes[chunk.id]]
    if to_embed:
        vectors = embed([chunk.embedding_text for chunk in to_embed])
        rows = [
            {
                "id": chunk.id,
                "store_id": chunk.store_id,
                "article_id": chunk.article_id,
                "article_title": chunk.article_title,
                "heading": chunk.heading,
                "text": chunk.text,
                "embedding": vector,
                "embedding_model": model_name,
                "content_hash": hashes[chunk.id],
            }
            for chunk, vector in zip(to_embed, vectors, strict=True)
        ]
        statement = insert(KBChunkRow).values(rows)
        session.execute(
            statement.on_conflict_do_update(
                index_elements=[KBChunkRow.id],
                set_={column: statement.excluded[column] for column in rows[0] if column != "id"},
            )
        )

    return IngestReport(
        embedded=len(to_embed), unchanged=len(chunks) - len(to_embed), deleted=len(stale_ids)
    )


def main() -> None:
    with new_session() as session:
        report = ingest(session, get_chunks(), embed_texts, EMBEDDING_MODEL_NAME)
        session.commit()
    print(
        f"kb_chunks: {report.embedded} embedded, {report.unchanged} unchanged, "
        f"{report.deleted} deleted"
    )


if __name__ == "__main__":
    main()
