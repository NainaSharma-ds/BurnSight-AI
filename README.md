# BurnSight-AI: AI Predictive Risk Monitoring

**Project ID: SIH26170**

An interactive dashboard for burn-in intelligence, anomaly detection, 168-hour prediction
and QA decision support. This repository currently holds the **frontend** (Streamlit + Plotly).

> **DEMO MODE:** the frontend runs on synthetic sample data until the FastAPI backend is
> connected. Nothing shown in demo mode is real model output.

## What the dashboard shows

| Section | What it does |
|---|---|
| KPI cards | Overall risk, predicted 168h value, anomaly score, QA decision |
| Prediction trend | Actual vs predicted values with an acceptance threshold |
| Anomaly analysis | Mahalanobis distance, LOF score, normal vs anomalous units |
| Risk engine | Overall risk gauge, anomaly risk, drift risk, safety slope |
| QA decision engine | PASS / MONITOR / EXTEND / REJECT with the rules behind the decision |
| Explainability | Permutation importance, LIME-style contributions, plain-English reason |
| Trends and history | Risk, anomaly and prediction trends, recent-analysis table |
| Export | CSV of results and an HTML analysis report |

## Architecture

```
Burn-in Data -> Preprocessing -> Module A + Module B -> Risk Engine
             -> QA Decision Engine -> Explainability -> Dashboard
```

Module A: Mahalanobis distance + LOF. Module B: XGBoost regressor.
Explainability: permutation importance, LIME, QA rule engine.

## Run it locally (Windows)

```
git clone https://github.com/NainaSharma-ds/BurnSight-AI.git
cd BurnSight-AI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run frontend/app.py
```

Then open http://localhost:8501

## Project structure

```
BurnSight-AI/
├── .streamlit/config.toml     dark theme
├── frontend/
│   ├── app.py                 the dashboard page
│   └── utils/
│       ├── demo_data.py       synthetic sample data (get_demo_data)
│       ├── api.py             FastAPI client (get_api_data)
│       ├── charts.py          Plotly charts
│       └── report.py          HTML report builder
├── requirements.txt
└── README.md
```

## Connecting the FastAPI backend

1. Start the backend so it serves `GET /analyze?component=...&lot=...&time_window=...`
2. In `frontend/app.py` set `USE_API = True`.
3. If the backend uses different JSON key names, only `to_dashboard_format()` in
   `frontend/utils/api.py` needs editing. The dashboard code does not change.

If the backend is unreachable, the dashboard shows an error message instead of crashing.

## Tech stack

Python, Streamlit, Plotly, Pandas, NumPy, Requests. Backend: FastAPI.
