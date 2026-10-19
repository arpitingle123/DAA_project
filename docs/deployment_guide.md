# FinGraph Deployment Guide

This guide summarizes the final Week 9 delivery steps for running and verifying FinGraph.

## 1. Project Requirements

- Python 3.9+
- Node.js 18+
- npm
- Local Kaggle transaction dataset at `data/raw/transactions.csv`
- Generated Week 5 artifacts:
  - `data/processed/week5_model_metrics.json`
  - `data/processed/week5_top_predictions.csv`
  - `data/processed/week5_feature_importance.csv`
  - `models/week5_fraud_model.pkl`

The dataset, generated outputs, and model files are intentionally not pushed to GitHub.

## 2. Backend Setup

```bash
cd FinGraph
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start the backend:

```bash
venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Verify:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

If the backend runs on a different URL, create `frontend/.env`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## 4. Production Build Check

```bash
cd frontend
npm run build
```

The build output is generated in `frontend/dist/` and is ignored by Git.

## 5. Backend Test Check

```bash
cd FinGraph
source venv/bin/activate
python -m pytest tests/test_backend_api.py -q
```

Expected result:

```text
2 passed
```

## 6. Suggested Hosting Approach

For a simple final submission:

- Backend: run locally with Uvicorn and show API docs.
- Frontend: run locally with Vite or serve the production build.
- GitHub: submit source code, README, screenshots, documentation, and progress chart.

For cloud deployment:

- Backend can be hosted on Render, Railway, or similar Python hosting.
- Frontend can be hosted on Vercel or Netlify.
- Set `VITE_API_BASE_URL` to the hosted backend URL.
- Keep dataset/model artifacts private unless the hosting environment supports secure storage.

## 7. Final Verification Checklist

- Backend `/health` returns `status: ok`.
- Backend `/metrics` returns Week 5 model metrics.
- Backend `/predict` returns fraud probability for a sample transaction.
- Frontend shows `Live API` when backend is available.
- Risk Queue loads top predictions.
- Transaction Scoring can submit a prediction request.
- Model Performance page shows Random Forest and Logistic Regression results.
- README and progress chart show Week 9 final delivery status.

