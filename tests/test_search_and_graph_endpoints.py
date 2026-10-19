"""
Backend Search & Graph API tests using an isolated temporary SQLite database.

Verifies:
- `GET /transactions` with all search filters, sorting, pagination, and limits.
- `GET /transactions/{id}` record retrieval and 404 handling.
- `GET /graph/account/{account_id}` 1-hop and 2-hop topological network expansion.
- `GET /graph/network` connected cluster generation.
- Strict distinction between ground-truth fraud and legitimate records.
- Node and edge limits.
- Preservation of `POST /predict`.
"""

import sqlite3
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app import app
from backend.data_access import data_access, DB_PATH
from src.init_db import import_transactions

TEST_TRANSACTIONS_CSV = """step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
1,PAYMENT,1000.0,C_ALICE,5000.0,4000.0,M_STORE,0.0,0.0,0,0
1,TRANSFER,50000.0,C_BOB,50000.0,0.0,C_CHARLIE,0.0,0.0,1,0
1,CASH_OUT,50000.0,C_CHARLIE,50000.0,0.0,C_MULE,20000.0,70000.0,1,0
2,PAYMENT,250.0,C_DAVE,1000.0,750.0,M_STORE,0.0,0.0,0,0
2,TRANSFER,15000.0,C_ALICE,4000.0,0.0,C_BOB,0.0,15000.0,0,0
3,TRANSFER,80000.0,C_MULE,70000.0,0.0,C_OFFSHORE,0.0,80000.0,1,0
3,CASH_IN,5000.0,C_EVE,0.0,5000.0,M_STORE,0.0,0.0,0,0
"""


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def temp_database(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("db_test")
    csv_file = temp_dir / "test_tx.csv"
    csv_file.write_text(TEST_TRANSACTIONS_CSV.strip(), encoding="utf-8")
    db_file = temp_dir / "test_tx.db"

    # Import into temp db
    import_transactions(
        csv_path=csv_file,
        db_path=db_file,
        chunk_size=3,
        force=True,
        verbose=False,
    )

    # Point data_access to the temporary database
    original_path = data_access.db_path
    data_access.set_db_path(db_file)

    yield db_file

    # Reset data_access back to production database
    data_access.set_db_path(original_path)


def test_transactions_pagination(client):
    # Page 1 (3 items)
    r1 = client.get("/transactions?page=1&page_size=3&sort_by=id&sort_dir=asc")
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["total"] == 7
    assert d1["page"] == 1
    assert d1["page_size"] == 3
    assert d1["total_pages"] == 3
    assert len(d1["items"]) == 3
    assert [it["id"] for it in d1["items"]] == [1, 2, 3]

    # Page 2 (3 items)
    r2 = client.get("/transactions?page=2&page_size=3&sort_by=id&sort_dir=asc")
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["page"] == 2
    assert [it["id"] for it in d2["items"]] == [4, 5, 6]

    # Page 3 (1 item)
    r3 = client.get("/transactions?page=3&page_size=3&sort_by=id&sort_dir=asc")
    assert r3.status_code == 200
    d3 = r3.json()
    assert len(d3["items"]) == 1
    assert d3["items"][0]["id"] == 7


def test_transactions_direct_id_filter(client):
    r = client.get("/transactions?transaction_id=2")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == 2
    assert data["items"][0]["nameOrig"] == "C_BOB"


def test_transactions_sender_and_receiver_filter(client):
    # Sender filter
    r_sender = client.get("/transactions?sender=C_ALICE")
    assert r_sender.status_code == 200
    d_sender = r_sender.json()
    assert d_sender["total"] == 2
    for item in d_sender["items"]:
        assert item["nameOrig"] == "C_ALICE"

    # Receiver filter
    r_recv = client.get("/transactions?receiver=M_STORE")
    assert r_recv.status_code == 200
    d_recv = r_recv.json()
    assert d_recv["total"] == 3
    for item in d_recv["items"]:
        assert item["nameDest"] == "M_STORE"

    # Account filter (either sender or receiver)
    r_acc = client.get("/transactions?account=C_BOB")
    assert r_acc.status_code == 200
    d_acc = r_acc.json()
    # C_BOB is sender in row 2 and receiver in row 5
    assert d_acc["total"] == 2


def test_transactions_type_and_fraud_filter(client):
    # Filter for ground-truth fraud only
    r_fraud = client.get("/transactions?is_fraud=1")
    assert r_fraud.status_code == 200
    d_fraud = r_fraud.json()
    assert d_fraud["total"] == 3
    for it in d_fraud["items"]:
        assert it["isFraud"] == 1

    # Filter for legitimate only
    r_legit = client.get("/transactions?is_fraud=0")
    assert r_legit.status_code == 200
    d_legit = r_legit.json()
    assert d_legit["total"] == 4
    for it in d_legit["items"]:
        assert it["isFraud"] == 0

    # Filter by type + fraud
    r_tf = client.get("/transactions?type=TRANSFER&is_fraud=1")
    assert r_tf.status_code == 200
    d_tf = r_tf.json()
    assert d_tf["total"] == 2
    for it in d_tf["items"]:
        assert it["type"] == "TRANSFER"
        assert it["isFraud"] == 1


def test_transactions_amount_and_step_filter(client):
    # Amount range: $10,000 to $60,000
    r_amt = client.get("/transactions?min_amount=10000&max_amount=60000")
    assert r_amt.status_code == 200
    d_amt = r_amt.json()
    # Rows 2 ($50k), 3 ($50k), 5 ($15k)
    assert d_amt["total"] == 3
    for it in d_amt["items"]:
        assert 10000 <= it["amount"] <= 60000

    # Step filter: step = 3
    r_step = client.get("/transactions?step=3")
    assert r_step.status_code == 200
    d_step = r_step.json()
    assert d_step["total"] == 2
    for it in d_step["items"]:
        assert it["step"] == 3


def test_transactions_sorting(client):
    # Sort by amount DESC
    r_desc = client.get("/transactions?sort_by=amount&sort_dir=desc")
    assert r_desc.status_code == 200
    items = r_desc.json()["items"]
    amounts = [it["amount"] for it in items]
    assert amounts == sorted(amounts, reverse=True)
    assert amounts[0] == 80000.0

    # Sort by amount ASC
    r_asc = client.get("/transactions?sort_by=amount&sort_dir=asc")
    assert r_asc.status_code == 200
    items_asc = r_asc.json()["items"]
    amounts_asc = [it["amount"] for it in items_asc]
    assert amounts_asc == sorted(amounts_asc)
    assert amounts_asc[0] == 250.0


def test_transactions_input_validation(client):
    # Invalid sort direction
    r_bad_dir = client.get("/transactions?sort_dir=sideways")
    assert r_bad_dir.status_code == 422

    # Negative page
    r_bad_page = client.get("/transactions?page=0")
    assert r_bad_page.status_code == 422

    # Excessive page_size (> 100)
    r_bad_size = client.get("/transactions?page_size=150")
    assert r_bad_size.status_code == 422


def test_single_transaction_by_id(client):
    # Valid ID lookup
    r = client.get("/transactions/2")
    assert r.status_code == 200
    tx = r.json()
    assert tx["id"] == 2
    assert tx["nameOrig"] == "C_BOB"
    assert tx["nameDest"] == "C_CHARLIE"
    assert tx["amount"] == 50000.0
    assert tx["isFraud"] == 1

    # Missing ID lookup
    r_missing = client.get("/transactions/999")
    assert r_missing.status_code == 404


def test_account_graph_1hop_exploration(client):
    # C_CHARLIE receives from C_BOB ($50k fraud) and sends to C_MULE ($50k fraud)
    r = client.get("/graph/account/C_CHARLIE?hops=1")
    assert r.status_code == 200
    data = r.json()

    assert data["root"] == "C_CHARLIE"
    assert data["isSample"] is False

    # Nodes should include C_CHARLIE, C_BOB, C_MULE
    node_ids = {n["id"] for n in data["nodes"]}
    assert "C_CHARLIE" in node_ids
    assert "C_BOB" in node_ids
    assert "C_MULE" in node_ids

    # Root node check
    charlie = next(n for n in data["nodes"] if n["id"] == "C_CHARLIE")
    assert charlie["isRoot"] is True
    assert charlie["inDegree"] == 1
    assert charlie["outDegree"] == 1
    assert charlie["totalInAmount"] == 50000.0
    assert charlie["totalOutAmount"] == 50000.0
    assert charlie["isFraud"] is True

    # Edges check: 2 directed edges
    assert len(data["edges"]) == 2
    edge_pairs = {(e["source"], e["target"]) for e in data["edges"]}
    assert ("C_BOB", "C_CHARLIE") in edge_pairs
    assert ("C_CHARLIE", "C_MULE") in edge_pairs

    # Ground truth fraud counts
    assert data["stats"]["fraudNodes"] >= 2
    assert data["stats"]["fraudEdges"] == 2


def test_account_graph_2hop_expansion(client):
    # 2-hop from C_BOB should reach:
    # 1-hop: C_CHARLIE (to), C_ALICE (from)
    # 2-hop: C_MULE (from C_CHARLIE), M_STORE (from C_ALICE)
    r1 = client.get("/graph/account/C_BOB?hops=1")
    r2 = client.get("/graph/account/C_BOB?hops=2")

    d1 = r1.json()
    d2 = r2.json()

    assert len(d2["nodes"]) > len(d1["nodes"])
    assert len(d2["edges"]) > len(d1["edges"])
    node_ids_2hop = {n["id"] for n in d2["nodes"]}
    assert "C_MULE" in node_ids_2hop


def test_connected_network_generation(client):
    r = client.get("/graph/network?limit=20&include_fraud=true")
    assert r.status_code == 200
    data = r.json()

    assert data["stats"]["totalNodes"] > 0
    assert data["stats"]["totalEdges"] > 0
    # Must contain ground truth fraud cases
    assert data["stats"]["fraudNodes"] > 0
    assert data["stats"]["fraudEdges"] > 0

    # Every edge must have required directional attributes
    for e in data["edges"]:
        assert e["source"]
        assert e["target"]
        assert e["amount"] > 0
        assert "isFraud" in e


def test_predict_endpoint_preserved(client):
    # Verify POST /predict remains fully operational with existing ML pipeline
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 25000.0,
        "oldbalanceOrg": 25000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "nameOrig": "C_TEST_ORIG",
        "nameDest": "C_TEST_DEST",
        "model_preference": "random_forest",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    res = r.json()
    assert "fraud_probability" in res
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert res["model_used"] == "Random Forest"
    assert "risk_level" in res
    assert len(res["top_reasons"]) > 0
