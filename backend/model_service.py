import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "model")


# ============================================================
# LOAD SAVED MODELS
# ============================================================

print("Loading models...")


# Module A - Anomaly Detection
anomaly_scaler = joblib.load(
    os.path.join(MODEL_DIR, "anomaly_scaler.pkl")
)

lof_model = joblib.load(
    os.path.join(MODEL_DIR, "lof_model.pkl")
)

mahalanobis_mean = joblib.load(
    os.path.join(MODEL_DIR, "mahalanobis_mean.pkl")
)

mahalanobis_inverse_covariance = joblib.load(
    os.path.join(
        MODEL_DIR,
        "mahalanobis_inverse_covariance.pkl"
    )
)

mahalanobis_threshold = joblib.load(
    os.path.join(
        MODEL_DIR,
        "mahalanobis_threshold.pkl"
    )
)


# Module B - XGBoost
xgb_model = xgb.XGBRegressor()

xgb_model.load_model(
    os.path.join(
        MODEL_DIR,
        "xgboost_168h.json"
    )
)


print("All models loaded successfully.")


# ============================================================
# MAHALANOBIS FUNCTION
# ============================================================

def calculate_mahalanobis(X):

    difference = X - mahalanobis_mean

    distance = np.sqrt(
        np.sum(
            (difference @ mahalanobis_inverse_covariance)
            * difference,
            axis=1
        )
    )

    return distance


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_component(
    value_0h,
    value_24h,
    lot_mean_0h=None
):

    # --------------------------------------------------------
    # 1. FEATURE ENGINEERING
    # --------------------------------------------------------

    drift_0_24 = value_24h - value_0h

    if value_0h != 0:

        percentage_drift_0_24 = (
            (value_24h - value_0h)
            / value_0h
        ) * 100

    else:

        percentage_drift_0_24 = 0


    # Lot deviation
    if lot_mean_0h is None:
        lot_mean_0h = value_0h

    deviation_from_lot_0h = (
        value_0h - lot_mean_0h
    )


    # --------------------------------------------------------
    # 2. MODULE A - ANOMALY DETECTION
    # --------------------------------------------------------

    anomaly_features = pd.DataFrame([
        {
            "Value_0h": value_0h,
            "Value_24h": value_24h,
            "Drift_0_24": drift_0_24,
            "Percentage_Drift_0_24":
                percentage_drift_0_24,
            "Deviation_From_Lot_0h":
                deviation_from_lot_0h
        }
    ])


    X_anomaly = anomaly_features[
        [
            "Value_0h",
            "Value_24h",
            "Drift_0_24",
            "Percentage_Drift_0_24",
            "Deviation_From_Lot_0h"
        ]
    ]


    # Scale
    X_scaled = anomaly_scaler.transform(
        X_anomaly
    )


    # Mahalanobis
    mahalanobis_distance = calculate_mahalanobis(
        X_scaled
    )[0]

    mahalanobis_anomaly = (
        mahalanobis_distance
        > mahalanobis_threshold
    )


    # LOF
    lof_prediction = lof_model.predict(
        X_scaled
    )[0]

    lof_score = -lof_model.decision_function(
        X_scaled
    )[0]

    lof_anomaly = (
        lof_prediction == -1
    )


    # Combined anomaly
    anomaly_detected = (
        mahalanobis_anomaly
        or lof_anomaly
    )


    predicted_anomaly = (
        "Anomaly"
        if anomaly_detected
        else "Normal"
    )


    # --------------------------------------------------------
    # 3. MODULE B - XGBOOST
    # --------------------------------------------------------

    xgb_features = pd.DataFrame([
        {
            "Value_0h": value_0h,
            "Value_24h": value_24h,
            "Drift_0_24": drift_0_24,
            "Percentage_Drift_0_24":
                percentage_drift_0_24
        }
    ])


    predicted_168h = xgb_model.predict(
        xgb_features
    )[0]


    # --------------------------------------------------------
    # 4. RETURN RESULT
    # --------------------------------------------------------

    return {
        "anomaly_detection":{
            "prediction":predicted_anomaly,
            "anomaly_detected": bool(anomaly_detected),
            "mahalanobis_distance": float(mahalanobis_distance),
            "mahalanobis_anomaly": bool(mahalanobis_anomaly),
            "lof_score": float(lof_score),
            "lof_anomaly": bool(lof_anomaly),
            "threshold": float(mahalanobis_threshold)
        },
        "prediction_168h": {
            "predicted_value": float(predicted_168h)
        }
    }