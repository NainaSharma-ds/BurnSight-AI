# BurnSight-AI
### AI-Driven Predictive & Explainable Burn-In Screening System

**SIH 2026 | Problem Statement: SIH26170 | ISRO**

BurnSight-AI is an AI-powered burn-in screening system designed to detect abnormal component behaviour early and predict future degradation before completion of the full 168-hour burn-in cycle.

Instead of relying only on fixed parametric limits, the system analyses **lot-relative behaviour, early drift and predicted future values** to support faster and more explainable QA decisions.

---
## 🎥 PROTOTYPE DEMO

▶️ **[Watch the BurnSight-AI Prototype Demo on YouTube](https://youtu.be/2aIapldTpCE?si=AxbXhU9hhrK-gk8J)**

## 🖥️ Prototype Screenshots
## 📊 Main Dashboard:
<img width="750" height="404" alt="main_dashboard" src="https://github.com/user-attachments/assets/232d5782-6b1e-4d19-9beb-6e1baa666ad6" />
## 🔍 Anomaly Analysis
<img width="750" height="404" alt="anamoly analysis" src="https://github.com/user-attachments/assets/2d37313c-48e4-48c8-8670-bfc5bdc446ca" />
## ⚠️ Risk Engine & QA Decision
<img width="750" height="404" alt="riskengine" src="https://github.com/user-attachments/assets/c79cf600-6634-4372-9539-7b6e343515af" />
## 🧠 Explainable AI
<img width="750" height="404" alt="xai" src="https://github.com/user-attachments/assets/f39afda5-0939-4762-85f0-095fda1db8a2" />

##  What Does BurnSight-AI Do?

The system works in two stages:

### 1. Dynamic Anomaly Detection
Analyses component behaviour at **24h and 96h** using:

- Mahalanobis Distance
- Local Outlier Factor (LOF)
- Lot-relative deviation
- Drift and percentage drift

This enables early warning as well as re-screening at a later burn-in stage.

### 2. 168h Degradation Prediction
An **XGBoost Regressor** uses early measurements:

`0h + 24h + early drift`

to predict the expected **168h value**.

The predicted degradation is then compared with a reference safety boundary.

---

##  QA Decision Engine

The system combines anomaly detection and future-risk prediction to generate an explainable decision:

| Decision | Meaning |
|---|---|
| ✅ PASS | Behaviour remains within expected range |
| 🟡 MONITOR | Behaviour is close to the risk boundary |
| 🟠 EXTENDED SCREENING | Further screening is recommended |
| 🔴 REJECT | Abnormal behaviour is detected |

Each decision is accompanied by a human-readable explanation.

---

## 📊 Model Performance

### Anomaly Detection

| Stage | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 24h Screening | 91% | 92% | 78% | 84% |
| 96h Re-screening | 98% | 94% | 100% | 97% |

### 168h Prediction — XGBoost

| Metric | Score |
|---|---:|
| MAE | 3.08 |
| RMSE | 5.16 |
| R² | 0.70 |

The strongest predictive features were **early drift and percentage drift**, showing that early degradation behaviour is important for forecasting future component behaviour.

---

##  Explainable AI

BurnSight-AI provides explanations for its predictions using:

- **Permutation Importance** — identifies the most influential features.
- **LIME** — explains individual component predictions.
- **QA Rule Engine** — converts model outputs into understandable QA reasons.

Example:

> High early drift + high predicted degradation → Extended Screening

---

##  What-If Simulation

The system also supports **What-If analysis**.

QA users can modify early measurements such as:

- Value at 0h
- Value at 24h

and observe:

**Predicted 168h value → Predicted drift → Risk ratio → QA decision**

This helps explore how changes in early component behaviour can affect the final screening decision.

---

## System Architecture

```text
                    BURN-IN DATA
              0h | 24h | 96h | 168h
                         ↓
              DATA PREPROCESSING
        Feature Engineering • Lot Analysis
                         ↓
          ┌──────────────┴──────────────┐
          ↓                             ↓
   MODULE A: ANOMALY              MODULE B: PREDICTION
      DETECTION                       168h Forecast
          ↓                             ↓
   24h Early Screening             0h + 24h
          ↓                        + Early Drift
   96h Re-screening                     ↓
          ↓                          XGBoost
 Mahalanobis + LOF                     ↓
          │                    Predicted 168h
          └──────────────┬──────────────┘
                         ↓
                    RISK ENGINE
          Anomaly + Predicted Drift
              + Safety Reference
                         ↓
                  QA DECISION ENGINE
              PASS | MONITOR |
          EXTENDED SCREENING | REJECT
                         ↓
                  EXPLAINABLE AI
           LIME + Permutation Importance
                 + QA Rule Engine
                         ↓
                  WHAT-IF ANALYSIS
                         ↓
                   QA DASHBOARD
```

## Run Locally

From the repository root, install the frontend dependencies and start the
Streamlit dashboard:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run frontend/app.py
```

The dashboard starts in demo mode by default. To use the real prediction
models, install the backend dependencies in the same Python environment,
start the API in a second terminal, and set `USE_API = True` in
`frontend/app.py`:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload
```

The dashboard is available at `http://localhost:8501`; the API health check
is available at `http://127.0.0.1:8000/health`.
