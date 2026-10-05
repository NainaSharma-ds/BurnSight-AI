import pandas as pd

# Load Module A results
df = pd.read_csv(
    "data/processed/module_a_results.csv"
)

print("========== MODULE A EVALUATION ==========")


# ==========================================
# 1. MISSED ANOMALIES
# ==========================================

missed = df[
    (df["Ground_Truth"] == "Anomaly") &
    (df["Predicted_Anomaly"] == "Normal")
]

print("\n========== MISSED ANOMALIES ==========")

print(f"Total missed anomalies: {len(missed)}")

print(
    missed[
        [
            "Component_ID",
            "Lot_ID",
            "Component_Type",
            "Value_0h",
            "Value_24h",
            "Drift_0_24",
            "Percentage_Drift_0_24",
            "Mahalanobis_Distance",
            "LOF_Score"
        ]
    ].to_string(index=False)
)


# ==========================================
# 2. DETECTED ANOMALIES
# ==========================================

detected = df[
    (df["Ground_Truth"] == "Anomaly") &
    (df["Predicted_Anomaly"] == "Anomaly")
]

print("\n========== DETECTED ANOMALIES ==========")

print(
    detected["Component_Type"]
    .value_counts()
)


# ==========================================
# 3. MISSED ANOMALIES BY TYPE
# ==========================================

print("\n========== MISSED ANOMALIES BY TYPE ==========")

print(
    missed["Component_Type"]
    .value_counts()
)


# ==========================================
# 4. FALSE POSITIVES
# ==========================================

false_positive = df[
    (df["Ground_Truth"] == "Normal") &
    (df["Predicted_Anomaly"] == "Anomaly")
]

print("\n========== FALSE POSITIVES ==========")

print(f"False positives: {len(false_positive)}")

print(
    false_positive[
        [
            "Component_ID",
            "Lot_ID",
            "Component_Type",
            "Mahalanobis_Distance",
            "LOF_Score"
        ]
    ].to_string(index=False)
)


# ==========================================
# 5. COMPONENT TYPE PERFORMANCE
# ==========================================

print("\n========== COMPONENT TYPE DISTRIBUTION ==========")

print(
    df.groupby(
        ["Component_Type", "Ground_Truth"]
    ).size()
)


print("\n========== EVALUATION COMPLETE ==========")