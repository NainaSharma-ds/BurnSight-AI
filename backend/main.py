import math
import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import PredictionRequest
from backend.model_service import predict_component


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_DATA = os.path.join(
    BASE_DIR, "data", "raw", "burn_in_raw.csv"
)


# ============================================================
# CONFIG
# ============================================================

# Max allowed |drift| of the predicted 168h value vs the 0h value (%).
# Set this from your real spec limit.
PRED_DRIFT_LIMIT_PCT = 10.0

ALLOWED_TIME_WINDOWS = {"24h", "96h", "168h"}

# Streamlit default origin. Add others if needed.
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="BurnSight AI API",
    description="Burn-In Component Risk Prediction API",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================

def clean_id(value):
    """Keep component/lot IDs as stripped strings."""
    if value is None:
        return ""
    return str(value).strip()


def safe_float(value):
    """Convert to float; return None for NaN / inf / invalid."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(f) or math.isinf(f):
        return None

    return f


def round_or_none(value, digits=3):
    return None if value is None else round(value, digits)


_cache = {"mtime": None, "df": None}


def load_raw_data():
    """Load the CSV, cached until the file changes on disk."""
    if not os.path.exists(RAW_DATA):
        raise HTTPException(
            status_code=500,
            detail=f"Dataset not found: {RAW_DATA}"
        )

    mtime = os.path.getmtime(RAW_DATA)

    if _cache["df"] is not None and _cache["mtime"] == mtime:
        return _cache["df"]

    try:
        df = pd.read_csv(RAW_DATA)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read dataset: {e}"
        )

    required_columns = [
        "Component_ID",
        "Lot_ID",
        "Value_0h",
        "Value_24h",
        "Value_96h",
        "Value_168h",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Dataset is missing required columns: "
                + ", ".join(missing)
            )
        )

    df["Component_ID"] = df["Component_ID"].astype(str).str.strip()
    df["Lot_ID"] = df["Lot_ID"].astype(str).str.strip()

    _cache["mtime"] = mtime
    _cache["df"] = df

    return df


def linear_slope_pct_per_24h(points, base):
    """
    Least-squares slope of value vs hours, expressed as
    % of the 0h value per 24 hours.
    points: list of (hour, value) with valid values only.
    """
    if base in (None, 0) or len(points) < 2:
        return 0.0

    n = len(points)
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n

    denom = sum((p[0] - mean_x) ** 2 for p in points)

    if denom == 0:
        return 0.0

    slope = sum(
        (p[0] - mean_x) * (p[1] - mean_y) for p in points
    ) / denom

    return (slope * 24 / base) * 100


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "BurnSight AI Backend",
        "dataset": os.path.basename(RAW_DATA),
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    try:
        df = load_raw_data()

        return {
            "status": "healthy",
            "dataset": os.path.basename(RAW_DATA),
            "rows": int(len(df)),
            "components": int(df["Component_ID"].nunique()),
            "lots": int(df["Lot_ID"].nunique()),
        }

    except HTTPException as e:
        return {"status": "unhealthy", "error": str(e.detail)}

    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# ============================================================
# COMPONENTS
# ============================================================

@app.get("/components")
def get_components():
    df = load_raw_data()

    components = (
        df["Component_ID"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return {"count": len(components), "components": components}


# ============================================================
# LOTS
# ============================================================

@app.get("/lots")
def get_lots(component: Optional[str] = None):
    df = load_raw_data()

    if component:
        component = clean_id(component)
        df = df[df["Component_ID"] == component]

    lots = (
        df["Lot_ID"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return {"count": len(lots), "lots": lots}


# ============================================================
# COMPONENT INFORMATION
# ============================================================

@app.get("/component-info")
def component_info(component: str):
    component = clean_id(component)

    df = load_raw_data()
    match = df[df["Component_ID"] == component]

    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Component {component} not found."
        )

    rows = []

    for _, row in match.iterrows():
        rows.append({
            "Component_ID": clean_id(row["Component_ID"]),
            "Lot_ID": clean_id(row["Lot_ID"]),
            "Component_Type": str(row.get("Component_Type", "")),
            "Ground_Truth": str(row.get("Ground_Truth", "")),
        })

    return {
        "component": component,
        "count": len(rows),
        "data": rows,
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        result = predict_component(
            value_0h=float(request.Value_0h),
            value_24h=float(request.Value_24h),
            lot_mean_0h=float(request.Lot_Mean_0h),
        )

        return {
            "status": "success",
            "component": {
                "Component_ID": clean_id(request.Component_ID),
                "Lot_ID": clean_id(request.Lot_ID),
                "Component_Type": request.Component_Type,
            },
            "result": result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ANALYZE
# ============================================================

@app.get("/analyze")
def analyze(
    component: str,
    lot: str,
    time_window: str = "168h"
):
    component = clean_id(component)
    lot = clean_id(lot)

    if time_window not in ALLOWED_TIME_WINDOWS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid time_window '{time_window}'. "
                f"Use one of: {sorted(ALLOWED_TIME_WINDOWS)}"
            )
        )

    try:
        # ----------------------------------------------------
        # LOAD + FIND COMPONENT
        # ----------------------------------------------------

        df = load_raw_data()

        match = df[
            (df["Component_ID"] == component)
            & (df["Lot_ID"] == lot)
        ]

        if match.empty:
            available_components = (
                df["Component_ID"].drop_duplicates().tolist()
            )

            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"{component} in {lot} not found.",
                    "requested_component": component,
                    "requested_lot": lot,
                    "available_components": available_components[:50],
                    "total_components": len(available_components),
                }
            )

        row = match.iloc[0]

        # ----------------------------------------------------
        # VALUES (96h / 168h may be missing -> None)
        # ----------------------------------------------------

        value_0h = safe_float(row["Value_0h"])
        value_24h = safe_float(row["Value_24h"])
        value_96h = safe_float(row["Value_96h"])
        value_168h = safe_float(row["Value_168h"])

        if value_0h is None or value_24h is None:
            raise HTTPException(
                status_code=422,
                detail="Component has missing 0h/24h values."
            )

        component_type = str(row.get("Component_Type", ""))
        ground_truth = str(row.get("Ground_Truth", ""))

        # ----------------------------------------------------
        # LOT DATA
        # ----------------------------------------------------

        lot_data = df[df["Lot_ID"] == lot]

        lot_mean = safe_float(
            pd.to_numeric(lot_data["Value_0h"], errors="coerce").mean()
        )

        if lot_mean is None:
            raise HTTPException(
                status_code=422,
                detail=f"Lot {lot} has no valid 0h values."
            )

        # ----------------------------------------------------
        # ML MODEL
        # ----------------------------------------------------

        ml = predict_component(
            value_0h=value_0h,
            value_24h=value_24h,
            lot_mean_0h=lot_mean,
        )

        anomaly = ml["anomaly_detection"]
        prediction = ml["prediction_168h"]

        predicted_168h = float(prediction["predicted_value"])

        # ----------------------------------------------------
        # DRIFT
        # ----------------------------------------------------

        drift = value_24h - value_0h

        drift_pct = (
            (drift / value_0h) * 100 if value_0h != 0 else 0
        )

        slope_points = [
            (h, v)
            for h, v in (
                (0, value_0h),
                (24, value_24h),
                (96, value_96h),
                (168, value_168h),
            )
            if v is not None
        ]

        slope_pct = linear_slope_pct_per_24h(slope_points, value_0h)

        # ----------------------------------------------------
        # ANOMALY SCORES
        # ----------------------------------------------------

        mahalanobis = float(anomaly["mahalanobis_distance"])
        lof = float(anomaly["lof_score"])
        threshold = float(anomaly["threshold"])
        anomaly_detected = bool(anomaly["anomaly_detected"])

        if anomaly_detected:
            if mahalanobis > threshold:
                anomaly_status = "ANOMALY DETECTED"
            else:
                anomaly_status = "ELEVATED"
        else:
            anomaly_status = "NORMAL"

        # ----------------------------------------------------
        # PREDICTED 168h DRIFT CHECK
        # (compared against a drift limit, NOT the Mahalanobis
        #  threshold, which is in different units)
        # ----------------------------------------------------

        pred_drift_pct = (
            (predicted_168h - value_0h) / value_0h * 100
            if value_0h != 0 else 0
        )

        pred_out_of_spec = abs(pred_drift_pct) > PRED_DRIFT_LIMIT_PCT

        # ----------------------------------------------------
        # RISK SCORE (0-100)
        # ----------------------------------------------------

        anomaly_component = min(
            max((mahalanobis / max(threshold, 0.001)) * 50, 0),
            70
        )

        drift_component = min(abs(drift_pct) * 2, 20)

        prediction_component = 10 if pred_out_of_spec else 0

        risk_score = min(
            round(
                anomaly_component
                + drift_component
                + prediction_component,
                1
            ),
            100
        )

        if risk_score >= 75:
            risk_label = "Critical"
        elif risk_score >= 50:
            risk_label = "Elevated"
        elif risk_score >= 25:
            risk_label = "Moderate"
        else:
            risk_label = "Low"

        # ----------------------------------------------------
        # QA DECISION
        # ----------------------------------------------------

        if risk_label == "Critical":
            qa_decision = "REJECT"
            qa_reason = "Critical risk detected."

        elif anomaly_detected:
            qa_decision = "EXTEND"
            qa_reason = (
                "Anomaly detected; "
                "additional burn-in is recommended."
            )

        elif risk_label == "Elevated":
            qa_decision = "MONITOR"
            qa_reason = "Elevated risk; continue monitoring."

        else:
            qa_decision = "PASS"
            qa_reason = "Component passed the configured checks."

        # ----------------------------------------------------
        # QA RULES
        # ----------------------------------------------------

        qa_rules = []

        if anomaly["mahalanobis_anomaly"]:
            qa_rules.append({
                "rule": "Mahalanobis",
                "status": "FAIL",
                "message": "Mahalanobis threshold exceeded.",
            })
        else:
            qa_rules.append({
                "rule": "Mahalanobis",
                "status": "PASS",
                "message": "Mahalanobis distance is within threshold.",
            })

        if anomaly["lof_anomaly"]:
            qa_rules.append({
                "rule": "LOF",
                "status": "FAIL",
                "message": "LOF detected an anomaly.",
            })
        else:
            qa_rules.append({
                "rule": "LOF",
                "status": "PASS",
                "message": "LOF did not detect an anomaly.",
            })

        qa_rules.append({
            "rule": "168h Prediction",
            "status": "FAIL" if pred_out_of_spec else "PASS",
            "message": (
                f"Predicted 168h drift {pred_drift_pct:.1f}% exceeds "
                f"±{PRED_DRIFT_LIMIT_PCT}% limit."
                if pred_out_of_spec
                else
                f"Predicted 168h drift {pred_drift_pct:.1f}% "
                f"is within ±{PRED_DRIFT_LIMIT_PCT}% limit."
            ),
        })

        # ----------------------------------------------------
        # IMPORTANCE
        # ----------------------------------------------------

        importance = ml.get("feature_importance", [])

        if not importance:
            importance = [
                {"Feature": "Value_0h", "Importance": 0.30},
                {"Feature": "Value_24h", "Importance": 0.25},
                {"Feature": "Drift_0_24", "Importance": 0.25},
                {"Feature": "Percentage_Drift_0_24", "Importance": 0.20},
            ]

        # ----------------------------------------------------
        # POPULATION (skip rows with missing values)
        # ----------------------------------------------------

        population = []

        for cid, v0, v24 in zip(
            lot_data["Component_ID"],
            lot_data["Value_0h"],
            lot_data["Value_24h"],
        ):
            f0 = safe_float(v0)
            f24 = safe_float(v24)

            if f0 is None or f24 is None:
                continue

            population.append({
                "Component_ID": clean_id(cid),
                "Value_0h": f0,
                "Value_24h": f24,
            })

        # ----------------------------------------------------
        # TREND / HISTORY
        # ----------------------------------------------------

        trend = [
            {"Hour": 0, "Actual": value_0h, "Predicted": value_0h},
            {"Hour": 24, "Actual": value_24h, "Predicted": value_24h},
            {"Hour": 96, "Actual": value_96h, "Predicted": None},
            {"Hour": 168, "Actual": value_168h, "Predicted": predicted_168h},
        ]

        # Synthetic timestamps, only for plotting.
        history = [
            {"Time": "2000-01-01T00:00:00", "Value": value_0h},
            {"Time": "2000-01-02T00:00:00", "Value": value_24h},
            {"Time": "2000-01-05T00:00:00", "Value": value_96h},
            {"Time": "2000-01-08T00:00:00", "Value": value_168h},
        ]

        # ----------------------------------------------------
        # CONTRIBUTIONS
        # ----------------------------------------------------

        contributions = [
            {"Feature": "Mahalanobis", "Contribution": mahalanobis},
            {"Feature": "LOF", "Contribution": lof},
        ]

        # ----------------------------------------------------
        # WHY TEXT
        # ----------------------------------------------------

        why_text = (
            f"Component {component} from lot {lot} "
            f"was evaluated using the burn-in values, "
            f"Mahalanobis distance, LOF and XGBoost. "
            f"The 0h value was {value_0h:.3f}, "
            f"the 24h value was {value_24h:.3f}, "
            f"and the predicted 168h value was "
            f"{predicted_168h:.3f} "
            f"({pred_drift_pct:+.1f}% vs 0h)."
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {
            "status": "success",
            "component": component,
            "lot": lot,
            "time_window": time_window,

            "value_0h": value_0h,
            "value_24h": value_24h,
            "value_96h": value_96h,
            "value_168h": value_168h,

            "component_type": component_type,
            "ground_truth": ground_truth,

            "risk_score": risk_score,
            "risk_label": risk_label,

            "predicted_168h": round(predicted_168h, 3),
            "predicted_drift_pct": round(pred_drift_pct, 3),
            "threshold": round(threshold, 3),

            "anomaly_score": round(mahalanobis, 3),
            "anomaly_status": anomaly_status,
            "anomaly_note": (
                "Anomaly detected."
                if anomaly_detected
                else "No anomaly detected."
            ),

            "mahalanobis": round(mahalanobis, 3),
            "lof": round(lof, 3),

            "drift_pct": round(drift_pct, 3),
            "slope_pct": round(slope_pct, 3),

            "qa_decision": qa_decision,
            "qa_reason": qa_reason,
            "qa_rules": qa_rules,

            "population": population,
            "unit_xy": [value_0h, value_24h],
            "trend": trend,
            "history": history,
            "importance": importance,
            "contributions": contributions,
            "why_text": why_text,
        }

    except HTTPException:
        raise

    except Exception as e:
        print("ANALYZE ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )