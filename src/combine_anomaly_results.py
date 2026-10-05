import pandas as pd


# ==========================================
# LOAD RESULTS
# ==========================================

early = pd.read_csv(
    "data/processed/module_a_results.csv"
)

screening_96 = pd.read_csv(
    "data/processed/module_a_96h_results.csv"
)


# ==========================================
# SELECT IMPORTANT COLUMNS
# ==========================================

early = early[
    [
        "Component_ID",
        "Lot_ID",
        "Component_Type",
        "Ground_Truth",
        "Predicted_Anomaly",
        "Mahalanobis_Distance",
        "LOF_Score"
    ]
]

screening_96 = screening_96[
    [
        "Component_ID",
        "Predicted_Anomaly_96h",
        "Mahalanobis_96h",
        "LOF_96h"
    ]
]


# ==========================================
# MERGE
# ==========================================

final = early.merge(
    screening_96,
    on="Component_ID",
    how="left"
)


# ==========================================
# CREATE FINAL ANOMALY STATUS
# ==========================================

def final_status(row):

    if row["Predicted_Anomaly_96h"] == "Anomaly":
        return "ANOMALY"

    if row["Predicted_Anomaly"] == "Anomaly":
        return "EARLY_WARNING"

    return "NORMAL"


final["Anomaly_Status"] = final.apply(
    final_status,
    axis=1
)


# ==========================================
# SAVE
# ==========================================

final.to_csv(
    "data/processed/final_anomaly_results.csv",
    index=False
)


# ==========================================
# SUMMARY
# ==========================================

print("========== FINAL ANOMALY SUMMARY ==========")

print(
    final["Anomaly_Status"].value_counts()
)

print(
    "\nSaved: data/processed/final_anomaly_results.csv"
)