"""
Creates all tables defined in src/db/models.py, if they don't already
exist. Safe to re-run - SQLAlchemy's create_all() skips tables that are
already there rather than erroring or duplicating them.
"""

from src.db.database import Base, engine
from src.db import models  # noqa: F401 - import registers the model classes with Base

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Tables created:", list(Base.metadata.tables.keys()))
