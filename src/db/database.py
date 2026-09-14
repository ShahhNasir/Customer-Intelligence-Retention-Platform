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
#
# pool_size/max_overflow raised above SQLAlchemy's defaults (5/10) for
# load testing under real concurrency: each uvicorn WORKER PROCESS gets
# its own engine/pool (they don't share one), so with N worker processes
# the real ceiling is N * (pool_size + max_overflow) connections to
# Postgres. Sized to stay under the container's max_connections=100 even
# with several workers running (e.g. 4 workers * 20 = 80).
engine = create_engine(
    settings.database_url, pool_pre_ping=True, pool_size=10, max_overflow=10
)

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
