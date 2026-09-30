"""Add kb_chunks.store_id: the knowledge base becomes multi-tenant.

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # kb_chunks is derived data — rebuilt from kb_articles/ by kb_ingest —
    # and existing rows have no store to backfill (their ids also change to
    # "<store>/<article>#<n>"). So clear it; re-run ingestion after
    # upgrading. A table holding data that exists nowhere else would need a
    # real backfill instead.
    op.execute("DELETE FROM kb_chunks")
    op.add_column("kb_chunks", sa.Column("store_id", sa.String(), nullable=False))
    op.create_index("ix_kb_chunks_store_id", "kb_chunks", ["store_id"])


def downgrade() -> None:
    op.drop_index("ix_kb_chunks_store_id", table_name="kb_chunks")
    op.drop_column("kb_chunks", "store_id")
