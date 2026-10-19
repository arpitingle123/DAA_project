# FinGraph Final Report Notes

These notes can be used while preparing the final Week 9 report and presentation.

## Project Title

FinGraph: Detecting Financial Fraud Using Graph-Based Analysis and Machine Learning

## Objective

The objective of FinGraph is to detect suspicious financial transactions by combining graph-based account relationship analysis, rule-based fraud scoring, and machine learning classification. The final system presents model outputs through a professional fraud investigation dashboard.

## Dataset

The project uses the Kaggle online payment fraud detection dataset. The dataset contains simulated financial transactions with sender accounts, receiver accounts, transaction type, amount, balances before and after the transaction, and fraud labels.

## Methodology

1. Load and inspect transaction data.
2. Clean and validate the dataset.
3. Build a directed transaction graph where accounts are nodes and transactions are edges.
4. Apply graph analysis including PageRank, community detection, and circular pattern checks.
5. Create rule-based fraud signals such as balance mismatch, account emptied, and high-risk transaction types.
6. Train baseline machine learning models using engineered transaction features.
7. Evaluate Logistic Regression and Random Forest models.
8. Serve saved model artifacts using a FastAPI backend.
9. Build a React dashboard for fraud monitoring, transaction scoring, graph intelligence, and analyst case review.

## Final System Components

- `src/`: data cleaning, graph construction, graph analysis, rule-based fraud logic, and ML training.
- `backend/`: FastAPI service for health checks, metrics, top predictions, feature importance, and single-transaction prediction.
- `frontend/`: React + Vite risk operations dashboard.
- `docs/`: progress chart, deployment guide, and report notes.

## Key Results

- Random Forest was selected as the best baseline model.
- The Week 5 training sample used 200,000 transactions.
- The model achieved high precision and recall on the held-out test sample.
- The dashboard converts raw model outputs into analyst-friendly workflows.

## Dashboard Features

- Operations overview with fraud and model KPIs.
- Risk queue showing high-risk transactions.
- Transaction detail panel with risk gauge and triggered signals.
- Guided transaction scoring form with realistic presets.
- Graph intelligence visualization for account relationships.
- Model performance page with confusion matrix and feature importance.
- Rules and signals simulator.
- Frontend-only case review workflow for analyst decisions.

## Testing And Verification

- Backend API tests verify health and prediction endpoints.
- Frontend production build is verified using Vite.
- Manual verification includes checking backend `/health`, `/metrics`, `/predict`, and frontend dashboard API status.

## Limitations

- The current ML model uses transaction-level engineered features, not a full GNN model.
- Case review decisions are stored in frontend state only and are not persisted in a database.
- Dataset and model artifacts are local and not deployed publicly.
- Authentication and role-based analyst access are not included.

## Future Scope

- Add graph-derived features directly into the machine learning model.
- Experiment with Graph Neural Network models.
- Add persistent case management with a database.
- Add authentication and analyst roles.
- Deploy backend and frontend to cloud hosting.
- Add batch prediction and real-time transaction stream support.

## Conclusion

FinGraph demonstrates a complete fraud detection workflow from data processing to model serving and analyst-facing dashboard design. The final Week 9 delivery includes a full-stack prototype, documented setup process, verified backend endpoints, and a professional dashboard suitable for project demonstration.

