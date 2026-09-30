import os
from collections.abc import Iterator
from pathlib import Path

import pytest

# Settings() requires an API key at import time (it fails fast on a missing
# one, by design); the suite never calls Claude, so any value will do.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
# Tests mint their own tokens with this secret; see agent_service.shared.auth.
os.environ["JWT_SECRET"] = "test-jwt-secret-not-for-production-use"
# Always the test database — assigned, not setdefault, so a DATABASE_URL in
# the shell pointing at dev data can never be picked up by the suite.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://agent:agent@localhost:5433/agent_service_test"
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

from alembic import command  # noqa: E402  (imported after the env is set)
from alembic.config import Config  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from agent_service.shared.db import new_session  # noqa: E402


@pytest.fixture(scope="session")
def migrated_database() -> None:
    """Migrates the test database to the latest schema, once per test run.
    Needs `docker compose up -d`."""
    config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    command.upgrade(config, "head")


@pytest.fixture(scope="session")
def orders_database(migrated_database: None) -> None:
    """Seeds the sample orders (upsert, so re-runs are safe)."""
    from agent_service.shared.orders import seed_orders

    with new_session() as session:
        seed_orders(session)
        session.commit()


@pytest.fixture(scope="session")
def kb_database(migrated_database: None) -> None:
    """Ingests the real knowledge base into the test database."""
    from agent_service.langchain.kb_ingest import ingest
    from agent_service.langchain.knowledge_base import EMBEDDING_MODEL_NAME, embed_texts, get_chunks

    with new_session() as session:
        ingest(session, get_chunks(), embed_texts, EMBEDDING_MODEL_NAME)
        session.commit()


@pytest.fixture
def rollback_session(kb_database: None) -> Iterator[Session]:
    """A session whose changes are all rolled back at the end of the test."""
    with new_session() as session:
        yield session
        session.rollback()
