"""SQLAlchemy engine and session factory for the app's Postgres database."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


@lru_cache
def get_engine() -> Engine:
    # One engine (and so one connection pool) per process, created on first
    # use rather than at import, so importing the app never needs a live
    # database. pool_pre_ping discards connections the server has dropped,
    # e.g. after a Postgres restart, instead of failing the next query.
    return create_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(get_engine())


def new_session() -> Session:
    return _session_factory()()
