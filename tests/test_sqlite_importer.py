"""
Unit tests for FinGraph SQLite Transaction Importer.

Tests verify:
- Batch chunking behavior on small temporary datasets (fast execution).
- Preservation of all source columns, row order, and record counts.
- Stable row-based auto-incrementing ID.
- Correct SQLite column data types.
- Creation of required B-tree indexes (sender, receiver, type, fraud, step, amount).
- Interrupted import safety (temporary database strategy prevents incomplete DB promotion).
- Idempotent execution and force-rebuild behavior.
"""

import sqlite3
import pytest
import pandas as pd
from pathlib import Path
from src.init_db import (
    import_transactions,
    is_database_complete,
    INDEX_DEFINITIONS,
)

SAMPLE_CSV_DATA = """step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
1,PAYMENT,9839.64,C1231006815,170136.0,160296.36,M1979787155,0.0,0.0,0,0
1,PAYMENT,1864.28,C1666544295,21249.0,19384.72,M2044282225,0.0,0.0,0,0
1,TRANSFER,181.0,C1305486145,181.0,0.0,C553264065,0.0,0.0,1,0
1,CASH_OUT,181.0,C840083671,181.0,0.0,C38997010,21182.0,0.0,1,0
2,PAYMENT,11668.14,C2048537720,41554.0,29885.86,M1230701703,0.0,0.0,0,0
2,CASH_IN,15998.6,C1385419999,0.0,15998.6,M1979787155,0.0,0.0,0,0
3,DEBIT,4500.0,C556677889,50000.0,45500.0,C112233445,10000.0,14500.0,0,0
3,TRANSFER,500000.0,C999888777,500000.0,0.0,C111222333,0.0,0.0,1,0
"""


@pytest.fixture
def temp_dataset(tmp_path: Path):
    csv_file = tmp_path / "test_transactions.csv"
    csv_file.write_text(SAMPLE_CSV_DATA.strip(), encoding="utf-8")
    db_file = tmp_path / "test_transactions.db"
    return csv_file, db_file


def test_import_preserves_records_columns_and_order(temp_dataset):
    csv_file, db_file = temp_dataset
    # Use chunk_size=3 to force multi-batch streaming
    total_imported = import_transactions(
        csv_path=csv_file,
        db_path=db_file,
        chunk_size=3,
        force=True,
        verbose=False,
    )

    assert total_imported == 8
    assert db_file.exists()

    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Check total row count
        cur.execute("SELECT COUNT(*) FROM transactions;")
        assert cur.fetchone()[0] == 8

        # Check order and fields against raw input
        cur.execute("SELECT * FROM transactions ORDER BY id ASC;")
        rows = cur.fetchall()

        # First row checks
        assert rows[0]["id"] == 1
        assert rows[0]["step"] == 1
        assert rows[0]["type"] == "PAYMENT"
        assert rows[0]["amount"] == 9839.64
        assert rows[0]["nameOrig"] == "C1231006815"
        assert rows[0]["isFraud"] == 0

        # Third row checks (Fraud Transfer)
        assert rows[2]["id"] == 3
        assert rows[2]["type"] == "TRANSFER"
        assert rows[2]["amount"] == 181.0
        assert rows[2]["nameOrig"] == "C1305486145"
        assert rows[2]["isFraud"] == 1

        # Last row checks
        assert rows[7]["id"] == 8
        assert rows[7]["step"] == 3
        assert rows[7]["type"] == "TRANSFER"
        assert rows[7]["amount"] == 500000.0
        assert rows[7]["isFraud"] == 1


def test_stable_row_based_id(temp_dataset):
    csv_file, db_file = temp_dataset
    import_transactions(csv_path=csv_file, db_path=db_file, chunk_size=2, verbose=False)

    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM transactions ORDER BY id ASC;")
        ids = [row[0] for row in cur.fetchall()]
        assert ids == [1, 2, 3, 4, 5, 6, 7, 8]


def test_csv_data_types(temp_dataset):
    csv_file, db_file = temp_dataset
    import_transactions(csv_path=csv_file, db_path=db_file, chunk_size=4, verbose=False)

    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(transactions);")
        columns = {row[1]: row[2] for row in cur.fetchall()}

        expected_types = {
            "id": "INTEGER",
            "step": "INTEGER",
            "type": "TEXT",
            "amount": "REAL",
            "nameOrig": "TEXT",
            "oldbalanceOrg": "REAL",
            "newbalanceOrig": "REAL",
            "nameDest": "TEXT",
            "oldbalanceDest": "REAL",
            "newbalanceDest": "REAL",
            "isFraud": "INTEGER",
            "isFlaggedFraud": "INTEGER",
        }

        for col_name, col_type in expected_types.items():
            assert col_name in columns, f"Missing column: {col_name}"
            assert columns[col_name].upper() == col_type, (
                f"Column {col_name} type {columns[col_name]} != {col_type}"
            )


def test_required_indexes_created(temp_dataset):
    csv_file, db_file = temp_dataset
    import_transactions(csv_path=csv_file, db_path=db_file, chunk_size=5, verbose=False)

    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='index';")
        created_indexes = {row[0] for row in cur.fetchall()}

        # Verify all required indexes exist
        required_index_names = {
            "idx_tx_orig",      # sender
            "idx_tx_dest",      # receiver
            "idx_tx_type",      # type
            "idx_tx_fraud",     # fraud label
            "idx_tx_step",      # step
            "idx_tx_amount",    # amount
        }

        for idx in required_index_names:
            assert idx in created_indexes, f"Missing required index: {idx}"


def test_interrupted_import_safety(temp_dataset, monkeypatch):
    csv_file, db_file = temp_dataset
    temp_db_file = db_file.with_name(f"{db_file.name}.tmp")

    # Simulate an interruption during the second chunk
    original_to_sql = pd.DataFrame.to_sql
    call_count = 0

    def fail_on_second_chunk(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise KeyboardInterrupt("Simulated process interruption mid-import")
        return original_to_sql(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_sql", fail_on_second_chunk)

    with pytest.raises(KeyboardInterrupt):
        import_transactions(
            csv_path=csv_file,
            db_path=db_file,
            chunk_size=2,
            force=True,
            verbose=False,
        )

    # The final target database must NOT have been created/promoted
    assert not db_file.exists()
    # is_database_complete must return False
    is_complete, _ = is_database_complete(db_file)
    assert is_complete is False

    # Now restore original behavior and verify that a rerun recovers cleanly
    monkeypatch.undo()

    total_recovered = import_transactions(
        csv_path=csv_file,
        db_path=db_file,
        chunk_size=3,
        force=True,
        verbose=False,
    )
    assert total_recovered == 8
    assert db_file.exists()
    assert not temp_db_file.exists()

    is_complete_after, row_count = is_database_complete(db_file)
    assert is_complete_after is True
    assert row_count == 8


def test_idempotent_skip_and_force(temp_dataset):
    csv_file, db_file = temp_dataset

    # Initial import
    count1 = import_transactions(csv_path=csv_file, db_path=db_file, verbose=False)
    assert count1 == 8

    # Second import without force: should return existing count immediately
    count2 = import_transactions(csv_path=csv_file, db_path=db_file, force=False, verbose=False)
    assert count2 == 8

    # Force rebuild: re-runs import safely
    count3 = import_transactions(csv_path=csv_file, db_path=db_file, force=True, verbose=False)
    assert count3 == 8
