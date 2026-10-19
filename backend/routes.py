from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from .data_access import data_access
from .schemas import (
    GraphResponse,
    HealthResponse,
    PredictionResponse,
    SummaryResponse,
    TransactionInput,
    TransactionItem,
    TransactionListResponse,
)
from .services import artifact_store, predict_transaction


router = APIRouter()


@router.get("/", response_model=SummaryResponse)
def get_summary():
    metrics = artifact_store.load_metrics() if artifact_store.metrics_available() else {}
    return {
        "project": "FinGraph",
        "backend_status": "FastAPI backend running",
        "best_model": metrics.get("best_model"),
        "sample_rows": metrics.get("sample_rows"),
        "fraud_cases_in_sample": metrics.get("fraud_cases_in_sample"),
    }


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "model_loaded": artifact_store.model_available(),
        "metrics_available": artifact_store.metrics_available(),
        "predictions_available": artifact_store.predictions_available(),
        "feature_importance_available": artifact_store.feature_importance_available(),
    }


@router.get("/database/status")
def get_db_status():
    return data_access.get_stats()


@router.get("/metrics")
def get_model_metrics():
    try:
        return artifact_store.load_metrics()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get("/metrics/elliptic")
def get_elliptic_metrics():
    try:
        return artifact_store.load_elliptic_metrics()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get("/predictions/top")
def get_top_predictions(limit: int = Query(20, ge=1, le=200)):
    try:
        return {
            "limit": limit,
            "predictions": artifact_store.load_top_predictions(limit),
        }
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get("/features/importance")
def get_feature_importance(limit: int = Query(20, ge=1, le=50)):
    try:
        return {
            "limit": limit,
            "features": artifact_store.load_feature_importance(limit),
        }
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    try:
        return predict_transaction(transaction)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get("/transactions", response_model=TransactionListResponse)
def search_transactions(
    transaction_id: Optional[int] = Query(None, ge=1, description="Direct search by SQLite row ID"),
    sender: Optional[str] = Query(None, description="Sender account ID (nameOrig)"),
    receiver: Optional[str] = Query(None, description="Receiver account ID (nameDest)"),
    account: Optional[str] = Query(None, description="Match either sender or receiver account ID"),
    type: Optional[str] = Query(None, description="Transaction type (TRANSFER, CASH_OUT, etc.)"),
    min_amount: Optional[float] = Query(None, ge=0, description="Minimum transaction amount"),
    max_amount: Optional[float] = Query(None, ge=0, description="Maximum transaction amount"),
    is_fraud: Optional[int] = Query(None, ge=0, le=1, description="Ground truth fraud status (0 or 1)"),
    step: Optional[int] = Query(None, ge=0, description="Transaction time step"),
    sort_by: str = Query("id", description="Field to sort by"),
    sort_dir: str = Query("asc", pattern="^(asc|desc|ASC|DESC)$", description="Sort direction"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page"),
):
    try:
        return data_access.search_transactions(
            transaction_id=transaction_id,
            sender=sender,
            receiver=receiver,
            account=account,
            type=type,
            min_amount=min_amount,
            max_amount=max_amount,
            is_fraud=is_fraud,
            step=step,
            sort_by=sort_by,
            sort_dir=sort_dir,
            page=page,
            page_size=page_size,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Transaction database not initialized. Please run init_db.py")
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid search parameters")


@router.get("/transactions/{transaction_id}", response_model=TransactionItem)
def get_transaction(transaction_id: int):
    try:
        tx = data_access.get_transaction_by_id(transaction_id)
        if not tx:
            raise HTTPException(status_code=404, detail=f"Transaction with ID {transaction_id} not found")
        return tx
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to retrieve transaction")


@router.get("/graph/account/{account_id}", response_model=GraphResponse)
def get_account_graph(
    account_id: str,
    hops: int = Query(1, ge=1, le=2, description="Exploration depth (1 or 2 hops)"),
    limit: int = Query(50, ge=1, le=100, description="Max transaction limit"),
):
    try:
        return data_access.get_account_graph(account_id=account_id, hops=hops, limit=limit)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Transaction database not initialized. Please run init_db.py")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate account graph")


@router.get("/graph/network", response_model=GraphResponse)
def get_initial_network(
    limit: int = Query(40, ge=1, le=80, description="Max transaction limit"),
    include_fraud: bool = Query(True, description="Whether to include fraud clusters"),
):
    try:
        return data_access.get_connected_network(limit=limit, include_fraud=include_fraud)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Transaction database not initialized. Please run init_db.py")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate network graph")

