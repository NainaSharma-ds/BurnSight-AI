import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# ==========================================
# LOAD TRAINING DATA
# ==========================================

train = pd.read_csv("data/processed/train.csv")

features = [
    "Value_0h",
    "Value_24h",
    "Drift_0_24",
    "Percentage_Drift_0_24"
]

X_train = train[features]
y_train = train["Value_168h"]

# ==========================================
# TRAIN XGBOOST MODEL
# ==========================================

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# SAFETY REFERENCE
# ==========================================

normal_data = train[
    train["Ground_Truth"] == "Normal"
]

normal_slope = (
    (normal_data["Value_168h"] - normal_data["Value_0h"])
    / 168
)

safety_slope = normal_slope.quantile(0.95)

# ==========================================
# WHAT-IF FUNCTION
# ==========================================

def what_if_prediction(
    value_0h,
    value_24h
):

    drift_0_24 = value_24h - value_0h

    percentage_drift = (
        drift_0_24 / value_0h
    ) * 100

    input_data = pd.DataFrame(
        [[
            value_0h,
            value_24h,
            drift_0_24,
            percentage_drift
        ]],
        columns=features
    )

    predicted_168h = model.predict(
        input_data
    )[0]

    predicted_drift = (
        predicted_168h - value_0h
    )

    predicted_slope = (
        predicted_drift / 168
    )

    risk_ratio = (
        predicted_slope / safety_slope
    )

    # ======================================
    # QA DECISION
    # ======================================

    if risk_ratio > 1:
        decision = "EXTENDED_SCREENING"

    elif risk_ratio >= 0.85:
        decision = "MONITOR"

    else:
        decision = "PASS"

    return {
        "Value_0h": value_0h,
        "Value_24h": value_24h,
        "Drift_0_24": drift_0_24,
        "Percentage_Drift_0_24": percentage_drift,
        "Predicted_168h": predicted_168h,
        "Predicted_Drift": predicted_drift,
        "Risk_Ratio": risk_ratio,
        "QA_Decision": decision
    }


# ==========================================
# TEST WHAT-IF SCENARIOS
# ==========================================

print("\n==========================================")
print("        BURNSIGHT AI WHAT-IF SIMULATOR")
print("==========================================")

# Scenario 1
result_1 = what_if_prediction(
    value_0h=10.0,
    value_24h=10.5
)

print("\nSCENARIO 1")
print("------------------------------------------")

for key, value in result_1.items():

    if isinstance(value, float):
        print(f"{key}: {value:.2f}")

    else:
        print(f"{key}: {value}")


# Scenario 2
result_2 = what_if_prediction(
    value_0h=10.0,
    value_24h=12.0
)

print("\nSCENARIO 2")
print("------------------------------------------")

for key, value in result_2.items():

    if isinstance(value, float):
        print(f"{key}: {value:.2f}")

    else:
        print(f"{key}: {value}")


print("\n==========================================")
print("What-If simulation completed successfully.")
print("==========================================")