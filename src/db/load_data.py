"""
Loads data/raw/customers.csv and interactions.csv into the Postgres
tables created by init_db.py. Run once (or after wiping the database) to
seed it - this is the bridge between our training-data CSVs and the
live-serving database.
"""

import time

import pandas as pd

from src.db.database import engine

BATCH_SIZE = 5000


def load_table(csv_path: str, table_name: str, parse_dates: list[str]):
    df = pd.read_csv(csv_path, parse_dates=parse_dates)
    start = time.time()

    df.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=BATCH_SIZE,
    )

    elapsed = time.time() - start
    print(f"Loaded {len(df):,} rows into '{table_name}' in {elapsed:.1f}s")


if __name__ == "__main__":
    load_table("data/raw/customers.csv", "customers", parse_dates=["signup_date"])
    load_table("data/raw/interactions.csv", "interactions", parse_dates=["timestamp"])
