"""Ingestion's incremental sync, against the real kb_chunks table but with a
fake embedder (no model load) and every change rolled back afterwards."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from agent_service.langchain.chunking import Chunk
from agent_service.langchain.kb_ingest import content_hash, ingest
from agent_service.langchain.kb_models import KBChunkRow

MODEL = "fake-model"


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, article_id="a", article_title="A", heading="", text=text)


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1] * 384 for _ in texts]


def _stored(session: Session) -> dict[str, str]:
    return dict(session.execute(select(KBChunkRow.id, KBChunkRow.text)).all())


def test_first_ingest_embeds_everything_and_replaces_other_rows(rollback_session: Session) -> None:
    embed = FakeEmbedder()
    report = ingest(rollback_session, [_chunk("a#0", "one"), _chunk("a#1", "two")], embed, MODEL)
    # The test database already holds the real knowledge base; syncing to a
    # different chunk list deletes every row not in it.
    assert report.embedded == 2 and report.unchanged == 0 and report.deleted > 0
    assert _stored(rollback_session) == {"a#0": "one", "a#1": "two"}


def test_reingest_only_embeds_what_changed(rollback_session: Session) -> None:
    ingest(rollback_session, [_chunk("a#0", "one"), _chunk("a#1", "two")], FakeEmbedder(), MODEL)

    embed = FakeEmbedder()
    report = ingest(rollback_session, [_chunk("a#0", "one"), _chunk("a#1", "two, edited")], embed, MODEL)

    assert (report.embedded, report.unchanged, report.deleted) == (1, 1, 0)
    assert embed.calls == [["A\n\ntwo, edited"]]
    assert _stored(rollback_session)["a#1"] == "two, edited"


def test_removed_chunks_are_deleted(rollback_session: Session) -> None:
    ingest(rollback_session, [_chunk("a#0", "one"), _chunk("a#1", "two")], FakeEmbedder(), MODEL)
    report = ingest(rollback_session, [_chunk("a#0", "one")], FakeEmbedder(), MODEL)
    assert (report.embedded, report.unchanged, report.deleted) == (0, 1, 1)
    assert _stored(rollback_session) == {"a#0": "one"}


def test_changing_the_model_reembeds_everything() -> None:
    chunk = _chunk("a#0", "one")
    assert content_hash(chunk, "model-a") != content_hash(chunk, "model-b")
