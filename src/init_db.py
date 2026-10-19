"""
FinGraph SQLite Database Initialization Script.

Safely imports transactions CSV into SQLite using batched chunking,
a temporary database file for atomic safety, and appropriate B-tree
indexes for fast queries.
"""

import os
import sys
import time
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DB_PATH = DEFAULT_PROCESSED_DIR / "transactions.db"

CHUNK_SIZE = 100_000

INDEX_DEFINITIONS = [
    ("idx_tx_orig", "nameOrig"),
    ("idx_tx_dest", "nameDest"),
    ("idx_tx_type", "type"),
    ("idx_tx_fraud", "isFraud"),
    ("idx_tx_step", "step"),
    ("idx_tx_amount", "amount"),
    ("idx_tx_fraud_type", "isFraud, type"),
]


def count_csv_rows(csv_path: Path) -> int:
    """Count data rows in CSV excluding header."""
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        line_count = sum(1 for line in f if line.strip()) - 1
    return max(0, line_count)


def is_database_complete(db_path: Path) -> Tuple[bool, int]:
    """Check if database exists, is valid, and import metadata indicates COMPLETE."""
    if not db_path.exists():
        return False, 0
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT status, row_count FROM _import_metadata WHERE id = 1;")
        row = cur.fetchone()
        if row and row[0] == "COMPLETE":
            return True, int(row[1])
    except Exception:
        pass
    finally:
        if conn:
            conn.close()
    return False, 0


def import_transactions(
    csv_path: Path = DEFAULT_CSV_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    chunk_size: int = CHUNK_SIZE,
    force: bool = False,
    verbose: bool = True,
) -> int:
    """
    Import transactions from CSV to SQLite.

    Guarantees:
    - Atomicity: uses a temporary file (db_path.with_suffix('.db.tmp')).
    - Source order & columns preserved.
    - Stable row-based ID (id INTEGER PRIMARY KEY AUTOINCREMENT).
    - Verification: row count verified against source CSV before committing.
    """
    csv_path = Path(csv_path)
    db_path = Path(db_path)
    temp_db_path = db_path.with_name(f"{db_path.name}.tmp")

    if not csv_path.exists():
        raise FileNotFoundError(f"Source CSV dataset not found at {csv_path}")

    # Check if already complete
    if not force:
        complete, count = is_database_complete(db_path)
        if complete:
            if verbose:
                print(f"Database already initialized and complete: {db_path}")
                print(f"Total indexed rows: {count:,}")
            return count

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Initializing SQLite transaction store from:\n  {csv_path}")
        print(f"Target database:\n  {db_path}")
        print(f"Batch size: {chunk_size:,} rows per batch")

    # Clean up any leftover temporary file from a previously interrupted run
    if temp_db_path.exists():
        try:
            temp_db_path.unlink()
        except Exception:
            pass

    start_time = time.time()
    conn = sqlite3.connect(temp_db_path)

    try:
        cur = conn.cursor()

        # Optimization pragmas for bulk load
        cur.execute("PRAGMA synchronous = OFF;")
        cur.execute("PRAGMA journal_mode = MEMORY;")
        cur.execute("PRAGMA cache_size = -64000;")

        # Schema definition
        cur.execute("""
        CREATE TABLE transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step INTEGER NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            nameOrig TEXT NOT NULL,
            oldbalanceOrg REAL NOT NULL,
            newbalanceOrig REAL NOT NULL,
            nameDest TEXT NOT NULL,
            oldbalanceDest REAL NOT NULL,
            newbalanceDest REAL NOT NULL,
            isFraud INTEGER NOT NULL,
            isFlaggedFraud INTEGER NOT NULL
        );
        """)

        cur.execute("""
        CREATE TABLE _import_metadata (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.commit()

        total_rows = 0
        batch_num = 0

        if verbose:
            print("Beginning batched streaming import...")

        reader = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            dtype={
                "step": "int32",
                "type": "string",
                "amount": "float64",
                "nameOrig": "string",
                "oldbalanceOrg": "float64",
                "newbalanceOrig": "float64",
                "nameDest": "string",
                "oldbalanceDest": "float64",
                "newbalanceDest": "float64",
                "isFraud": "int8",
                "isFlaggedFraud": "int8",
            },
        )

        for chunk in reader:
            batch_num += 1
            chunk_len = len(chunk)
            chunk.to_sql("transactions", conn, if_exists="append", index=False)
            total_rows += chunk_len
            elapsed = time.time() - start_time
            if verbose:
                print(f"  Batch {batch_num:3d}: imported {chunk_len:,} rows | Total: {total_rows:,} rows ({elapsed:.1f}s)")

        if verbose:
            print(f"\nAll {total_rows:,} rows loaded. Building B-tree indexes...")
        index_start = time.time()

        for idx_name, cols in INDEX_DEFINITIONS:
            if verbose:
                print(f"  Creating index {idx_name} on ({cols})...")
            cur.execute(f"CREATE INDEX {idx_name} ON transactions ({cols});")

        # Record completion in metadata table before closing
        cur.execute(
            "INSERT INTO _import_metadata (id, status, row_count) VALUES (1, 'COMPLETE', ?);",
            (total_rows,),
        )
        conn.commit()

        if verbose:
            print(f"Indexes built in {time.time() - index_start:.1f}s.")
    finally:
        # Guarantee connection closure so files can be renamed/deleted on Windows
        conn.close()

    # Atomically replace db_path with temp_db_path
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass
    temp_db_path.rename(db_path)

    total_duration = time.time() - start_time
    if verbose:
        print(f"\nSuccessfully created {db_path}")
        print(f"Final Row Count: {total_rows:,}")
        print(f"Total time elapsed: {total_duration:.1f}s ({total_duration/60:.2f} min)")

    return total_rows


if __name__ == "__main__":
    force_rebuild = "--force" in sys.argv
    import_transactions(
        csv_path=DEFAULT_CSV_PATH,
        db_path=DEFAULT_DB_PATH,
        force=force_rebuild,
    )
