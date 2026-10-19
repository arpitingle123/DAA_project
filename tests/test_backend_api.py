from fastapi.testclient import TestClient

from backend.app import app


client = TestClient(app)


def test_health_endpoint_reports_artifacts():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "metrics_available" in data


def test_database_status_endpoint():
    response = client.get("/database/status")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["row_count"] == 6362620
    assert data["fraud_count"] == 8213
    assert data["db_size_mb"] > 0


def test_transactions_pagination_and_sorting():
    response = client.get("/transactions?page=1&page_size=10&sort_by=id&sort_dir=asc")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 6362620
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert len(data["items"]) == 10
    assert data["items"][0]["id"] == 1
    assert data["items"][1]["id"] == 2


def test_transactions_filter_by_type_and_fraud():
    response = client.get("/transactions?type=TRANSFER&is_fraud=1&page_size=5")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 4097
    assert len(data["items"]) == 5
    for item in data["items"]:
        assert item["type"] == "TRANSFER"
        assert item["isFraud"] == 1


def test_transactions_filter_by_sender():
    response = client.get("/transactions?sender=C1231006815")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    for item in data["items"]:
        assert item["nameOrig"] == "C1231006815"


def test_transactions_empty_result():
    response = client.get("/transactions?sender=NONEXISTENT_ACCOUNT_99999")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["items"] == []


def test_transactions_invalid_sort_dir():
    response = client.get("/transactions?sort_dir=invalid_direction")
    assert response.status_code == 422  # validation error


def test_single_transaction_lookup():
    response = client.get("/transactions/1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1
    assert data["nameOrig"] == "C1231006815"
    assert data["type"] == "PAYMENT"
    assert "amount" in data
    assert "oldbalanceOrg" in data
    assert "newbalanceOrig" in data


def test_single_transaction_not_found():
    response = client.get("/transactions/999999999")
    assert response.status_code == 404


def test_graph_account_exploration():
    # Account C1305486145 is the first fraud transaction in the dataset
    response = client.get("/graph/account/C1305486145?hops=1&limit=20")
    assert response.status_code == 200
    data = response.json()
    assert data["root"] == "C1305486145"
    assert len(data["nodes"]) >= 2
    assert len(data["edges"]) >= 1
    assert data["stats"]["totalNodes"] == len(data["nodes"])
    assert data["stats"]["totalEdges"] == len(data["edges"])
    # Verify edge schema
    edge = data["edges"][0]
    assert "id" in edge
    assert "source" in edge
    assert "target" in edge
    assert "amount" in edge
    assert "type" in edge
    assert "isFraud" in edge


def test_graph_initial_network():
    response = client.get("/graph/network?limit=30&include_fraud=true")
    assert response.status_code == 200
    data = response.json()
    assert len(data["nodes"]) > 0
    assert len(data["edges"]) > 0
    assert data["stats"]["fraudNodes"] > 0
    assert data["stats"]["fraudEdges"] > 0


def test_predict_endpoint_regression():
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 2806.0,
        "oldbalanceOrg": 2806.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "nameOrig": "C1420196421",
        "nameDest": "C972765878",
        "model_preference": "random_forest",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["fraud_probability"] <= 1
    assert data["risk_level"] in {"Low", "Medium", "High"}
    assert data["model_used"] == "Random Forest"
    assert len(data["top_reasons"]) > 0
