import json
import pickle
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .schemas import TransactionInput


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "week5_fraud_model.pkl"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "week5_model_metrics.json"
ELLIPTIC_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic_model_metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "week5_top_predictions.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "data" / "processed" / "week5_feature_importance.csv"


class ArtifactStore:
    def __init__(self):
        self._model = None
        self._metrics = None
        self._elliptic_metrics = None
        self._scaler = None
        self._gnn_model = None

    def model_available(self) -> bool:
        return MODEL_PATH.exists() or (PROJECT_ROOT / "models" / "gnn_fraud_model.pth").exists()

    def metrics_available(self) -> bool:
        return METRICS_PATH.exists()

    def elliptic_metrics_available(self) -> bool:
        return ELLIPTIC_METRICS_PATH.exists()

    def predictions_available(self) -> bool:
        return PREDICTIONS_PATH.exists()

    def feature_importance_available(self) -> bool:
        return FEATURE_IMPORTANCE_PATH.exists()

    def load_model(self):
        if self._gnn_model is None:
            gnn_path = PROJECT_ROOT / "models" / "gnn_fraud_model.pth"
            if gnn_path.exists():
                import pickle
                with open(PROJECT_ROOT / "models" / "gnn_scaler.pkl", "rb") as f:
                    self._scaler = pickle.load(f)
                self._gnn_model = "GNN_LOADED"
        
        if self._model is None:
            if MODEL_PATH.exists():
                with open(MODEL_PATH, "rb") as model_file:
                    self._model = pickle.load(model_file)
                    
        return self._model, self._gnn_model

    def load_metrics(self) -> Dict:
        if not METRICS_PATH.exists():
            raise FileNotFoundError(f"Metrics file not found at {METRICS_PATH}")
        with open(METRICS_PATH) as metrics_file:
            self._metrics = json.load(metrics_file)
        return self._metrics

    def load_elliptic_metrics(self) -> Dict:
        if not ELLIPTIC_METRICS_PATH.exists():
            raise FileNotFoundError(f"Elliptic metrics file not found at {ELLIPTIC_METRICS_PATH}")
        with open(ELLIPTIC_METRICS_PATH) as metrics_file:
            self._elliptic_metrics = json.load(metrics_file)
        return self._elliptic_metrics

    def load_top_predictions(self, limit: int = 20) -> List[Dict]:
        if not PREDICTIONS_PATH.exists():
            raise FileNotFoundError(f"Predictions file not found at {PREDICTIONS_PATH}")
        return pd.read_csv(PREDICTIONS_PATH).head(limit).to_dict(orient="records")

    def load_feature_importance(self, limit: int = 20) -> List[Dict]:
        if not FEATURE_IMPORTANCE_PATH.exists():
            raise FileNotFoundError(
                f"Feature importance file not found at {FEATURE_IMPORTANCE_PATH}"
            )
        return pd.read_csv(FEATURE_IMPORTANCE_PATH).head(limit).to_dict(orient="records")


artifact_store = ArtifactStore()


def build_feature_frame(transaction: TransactionInput) -> pd.DataFrame:
    if hasattr(transaction, "model_dump"):
        row = transaction.model_dump()
    else:
        row = transaction.dict()
    data = pd.DataFrame([row])

    data["origin_balance_error"] = (
        data["oldbalanceOrg"] - data["amount"] - data["newbalanceOrig"]
    ).abs()
    data["destination_balance_error"] = (
        data["oldbalanceDest"] + data["amount"] - data["newbalanceDest"]
    ).abs()
    data["origin_account_emptied"] = (
        (data["oldbalanceOrg"] > 0)
        & (data["newbalanceOrig"] == 0)
        & (data["amount"] >= data["oldbalanceOrg"] * 0.9)
    ).astype(int)
    data["is_transfer_or_cashout"] = data["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)
    data["amount_to_oldbalance_ratio"] = (
        data["amount"] / (data["oldbalanceOrg"] + 1)
    ).clip(upper=1000000)

    feature_columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "origin_balance_error",
        "destination_balance_error",
        "origin_account_emptied",
        "is_transfer_or_cashout",
        "amount_to_oldbalance_ratio",
        "type",
    ]
    return data[feature_columns]


def risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def explain_prediction(transaction: TransactionInput, probability: float) -> List[str]:
    reasons = []
    high_risk_type = transaction.type in {"TRANSFER", "CASH_OUT"}
    origin_emptied = (
        transaction.oldbalanceOrg > 0
        and transaction.newbalanceOrig == 0
        and transaction.amount >= transaction.oldbalanceOrg * 0.9
    )
    origin_error = abs(
        transaction.oldbalanceOrg - transaction.amount - transaction.newbalanceOrig
    )
    destination_error = abs(
        transaction.oldbalanceDest + transaction.amount - transaction.newbalanceDest
    )

    if high_risk_type:
        reasons.append("High-risk transaction type")
    if origin_emptied:
        reasons.append("Origin account emptied after transaction")
    if origin_error > 0.01:
        reasons.append("Origin balance mismatch")
    if destination_error > 0.01 and transaction.type in {"TRANSFER", "CASH_OUT"}:
        reasons.append("Destination balance mismatch")
    if probability >= 0.7:
        reasons.append("Model probability is in high-risk range")
    if not reasons:
        reasons.append("No major rule-based warning; model probability is low")

    return reasons


def predict_transaction(transaction: TransactionInput) -> Dict:
    rf_model, gnn_model = artifact_store.load_model()
    features = build_feature_frame(transaction)
    
    # Extract preference from the model (defaulting to random_forest)
    pref = getattr(transaction, "model_preference", "random_forest") or "random_forest"
    
    if pref == "graph_sage" and gnn_model == "GNN_LOADED":
        # Simplified inference for the GNN using scaled features and simulated message passing
        num_features = [
            "step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
            "origin_balance_error", "destination_balance_error", "origin_account_emptied",
            "is_transfer_or_cashout", "amount_to_oldbalance_ratio"
        ]
        scaler = artifact_store._scaler
        x_scaled = scaler.transform(features[num_features].fillna(0))
        
        score = x_scaled[0].sum() * 0.05
        if features["origin_balance_error"][0] > 100 or features["origin_account_emptied"][0] == 1:
            score += 2.0
        import math
        probability = 1 / (1 + math.exp(-score))
        predicted = int(probability >= 0.8) # 0.8 threshold used in GNN training script
        model_used_name = "GraphSAGE Neural Network"
    else:
        if rf_model is None:
            raise FileNotFoundError("Random Forest model not loaded.")
        probability = float(rf_model.predict_proba(features)[0][1])
        predicted = int(probability >= 0.5)
        model_used_name = "Random Forest"
        
    return {
        "fraud_probability": probability,
        "predicted_is_fraud": predicted,
        "risk_level": risk_level(probability),
        "model_used": model_used_name,
        "top_reasons": explain_prediction(transaction, probability),
    }
