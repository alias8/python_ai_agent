"""Alembic migration environment, wired to the app's Settings and models."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from agent_service.langchain import kb_models  # noqa: F401  (registers kb_chunks on Base.metadata)
from agent_service.shared import orders  # noqa: F401  (registers orders on Base.metadata)
from agent_service.shared.config import get_settings
from agent_service.shared.db import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    # Tests pass their database's URL in via the Config object; otherwise
    # it's the app's own DATABASE_URL.
    return config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(url=_database_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_database_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
