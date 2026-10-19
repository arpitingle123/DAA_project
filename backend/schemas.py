from typing import List, Optional

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    step: int = Field(..., ge=0, description="Transaction time step")
    type: str = Field(..., description="Transaction type, e.g. TRANSFER or CASH_OUT")
    amount: float = Field(..., ge=0, description="Transaction amount")
    oldbalanceOrg: float = Field(..., ge=0, description="Origin balance before transaction")
    newbalanceOrig: float = Field(..., ge=0, description="Origin balance after transaction")
    oldbalanceDest: float = Field(..., ge=0, description="Destination balance before transaction")
    newbalanceDest: float = Field(..., ge=0, description="Destination balance after transaction")
    nameOrig: Optional[str] = Field(None, description="Optional origin account ID")
    nameDest: Optional[str] = Field(None, description="Optional destination account ID")
    model_preference: Optional[str] = Field("random_forest", description="Preferred model to use for inference (random_forest or graph_sage)")


class PredictionResponse(BaseModel):
    fraud_probability: float
    predicted_is_fraud: int
    risk_level: str
    model_used: str
    top_reasons: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    metrics_available: bool
    predictions_available: bool
    feature_importance_available: bool


class SummaryResponse(BaseModel):
    project: str
    backend_status: str
    best_model: Optional[str]
    sample_rows: Optional[int]
    fraud_cases_in_sample: Optional[int]


class TransactionItem(BaseModel):
    id: int
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    isFraud: int
    isFlaggedFraud: int


class TransactionListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    items: List[TransactionItem]


class GraphNode(BaseModel):
    id: str
    label: str
    isFraud: bool
    isRoot: bool
    inDegree: int
    outDegree: int
    totalInAmount: float
    totalOutAmount: float


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    amount: float
    type: str
    step: int
    isFraud: bool


class GraphStats(BaseModel):
    totalNodes: int
    totalEdges: int
    fraudNodes: int
    fraudEdges: int
    totalVolume: float


class GraphResponse(BaseModel):
    root: Optional[str] = None
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    stats: GraphStats
    isSample: bool = False

