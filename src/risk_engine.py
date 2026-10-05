import pandas as pd
import numpy as np
import os

# ==============================
# LOAD DATA
# ==============================

anomaly = pd.read_csv(
    "data/processed/final_anomaly_results.csv"
)

prediction = pd.read_csv(
    "data/processed/module_b_results.csv"
)

train = pd.read_csv(
    "data/processed/train.csv"
)

print("Anomaly data:", anomaly.shape)
print("Prediction data:", prediction.shape)


# ==============================
# MERGE RESULTS
# ==============================

results = anomaly.merge(
    prediction[
        [
            "Component_ID",
            "Value_0h",
            "Value_24h",
            "Predicted_168h",
            "Prediction_Error",
            "Absolute_Error"
        ]
    ],
    on="Component_ID",
    how="inner"
)

print("Combined data:", results.shape)


# ==============================
# CALCULATE PREDICTED DRIFT
# ==============================

results["Predicted_Drift_0_168"] = (
    results["Predicted_168h"] - results["Value_0h"]
)

results["Predicted_Slope"] = (
    results["Predicted_Drift_0_168"] / 168
)


# ==============================
# CALCULATE SAFETY SLOPE
# FROM NORMAL TRAINING DATA
# ==============================

normal_train = train[
    train["Ground_Truth"] == "Normal"
].copy()

normal_train["Normal_Drift_0_168"] = (
    normal_train["Value_168h"] -
    normal_train["Value_0h"]
)

normal_train["Normal_Slope"] = (
    normal_train["Normal_Drift_0_168"] / 168
)

# 95th percentile of normal degradation behaviour
safety_slope = normal_train["Normal_Slope"].quantile(0.95)

safety_drift = safety_slope * 168

print("\n========== SAFETY LIMIT ==========")
print(f"Safety slope : {safety_slope:.4f}")
print(f"Safety drift : {safety_drift:.4f}")


# ==============================
# RISK RATIO
# ==============================

results["Risk_Ratio"] = (
    results["Predicted_Slope"] / safety_slope
)


# ==============================
# RISK DECISION ENGINE
# ==============================

def classify_risk(row):

    # Confirmed anomaly at 96h
    if row["Anomaly_Status"] == "ANOMALY":
        return "REJECT"

    # Early warning or predicted unsafe drift
    if (
        row["Anomaly_Status"] == "EARLY_WARNING"
        or row["Risk_Ratio"] > 1.0
    ):
        return "EXTENDED_SCREENING"

    # Close to safety boundary
    if row["Risk_Ratio"] >= 0.85:
        return "MONITOR"

    return "PASS"


results["QA_Decision"] = results.apply(
    classify_risk,
    axis=1
)


# ==============================
# GENERATE EXPLANATION
# ==============================

def generate_reason(row):

    reasons = []

    if row["Anomaly_Status"] == "ANOMALY":
        reasons.append(
            "Anomalous behaviour detected during burn-in screening"
        )

    elif row["Anomaly_Status"] == "EARLY_WARNING":
        reasons.append(
            "Early anomaly warning detected at 24h"
        )

    if row["Risk_Ratio"] > 1.0:
        reasons.append(
            "Predicted degradation exceeds normal safety slope"
        )

    elif row["Risk_Ratio"] >= 0.85:
        reasons.append(
            "Predicted degradation is close to safety boundary"
        )

    if not reasons:
        reasons.append(
            "Component behaviour remains within expected limits"
        )

    return "; ".join(reasons)


results["QA_Reason"] = results.apply(
    generate_reason,
    axis=1
)


# ==============================
# SAVE RESULTS
# ==============================

os.makedirs(
    "data/processed",
    exist_ok=True
)

output_path = (
    "data/processed/risk_engine_results.csv"
)

results.to_csv(
    output_path,
    index=False
)


# ==============================
# DISPLAY RESULTS
# ==============================

print("\n========== QA DECISIONS ==========")

print(
    results["QA_Decision"]
    .value_counts()
)

print("\n========== SAMPLE RESULTS ==========")

print(
    results[
        [
            "Component_ID",
            "Anomaly_Status",
            "Predicted_168h",
            "Predicted_Drift_0_168",
            "Risk_Ratio",
            "QA_Decision",
            "QA_Reason"
        ]
    ].head(10).to_string(index=False)
)

print("\nRisk Engine results saved to:")
print(output_path)