"""
SQLAlchemy engine and session setup. Every other module that needs to
talk to the database imports `SessionLocal` (or the `get_db` dependency,
used later by FastAPI) from here - the engine itself is created once and
reused, not recreated per request.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.config import settings

# The engine manages a pool of actual database connections. pool_pre_ping
# checks a connection is still alive before handing it out, avoiding
# "connection closed" errors after periods of idle time - relevant once
# the API is a long-running server, not just short scripts like this one.
engine = create_engine(settings.database_url, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    """Every ORM model class inherits from this."""
    pass


def get_db():
    """
    Yields a database session and guarantees it's closed afterward, even
    if an error occurs. FastAPI (Phase 7) will use this as a dependency
    so each request gets its own session, not a shared global one.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
