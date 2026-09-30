"""Add kb_chunks.search_vector: full-text search terms for hybrid search.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import TSVECTOR

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A stored generated column: Postgres fills it in for existing rows as
    # part of this ALTER, so no re-ingestion is needed.
    op.add_column(
        "kb_chunks",
        sa.Column(
            "search_vector",
            TSVECTOR(),
            sa.Computed(
                "to_tsvector('english', article_title || ' ' || heading || ' ' || text)",
                persisted=True,
            ),
            nullable=True,
        ),
    )
    op.create_index(
        "kb_chunks_search_vector_gin", "kb_chunks", ["search_vector"], postgresql_using="gin"
    )


def downgrade() -> None:
    op.drop_index("kb_chunks_search_vector_gin", table_name="kb_chunks")
    op.drop_column("kb_chunks", "search_vector")
