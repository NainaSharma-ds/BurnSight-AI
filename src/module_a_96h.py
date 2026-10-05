import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# 1. LOAD DATA
# ==========================================

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

print("Training:", train.shape)
print("Testing:", test.shape)


# ==========================================
# 2. CREATE 96h FEATURES
# ==========================================

for df in [train, test]:

    df["Drift_24_96"] = (
        df["Value_96h"] - df["Value_24h"]
    )

    df["Percentage_Drift_24_96"] = (
        df["Drift_24_96"]
        / df["Value_24h"]
    ) * 100


# ==========================================
# 3. LOT-RELATIVE 96h DEVIATION
# ==========================================

train_lot_mean = train.groupby(
    "Lot_ID"
)["Value_96h"].transform("mean")

test_lot_mean = test.groupby(
    "Lot_ID"
)["Value_96h"].transform("mean")

train["Deviation_From_Lot_96h"] = (
    train["Value_96h"] - train_lot_mean
)

test["Deviation_From_Lot_96h"] = (
    test["Value_96h"] - test_lot_mean
)


# ==========================================
# 4. FEATURES
# ==========================================

features = [
    "Value_24h",
    "Value_96h",
    "Drift_24_96",
    "Percentage_Drift_24_96",
    "Deviation_From_Lot_96h"
]


# ==========================================
# 5. NORMAL TRAINING DATA
# ==========================================

normal_train = train[
    train["Ground_Truth"] == "Normal"
]

X_normal = normal_train[features]

X_test = test[features]


# ==========================================
# 6. SCALE
# ==========================================

scaler = StandardScaler()

X_normal_scaled = scaler.fit_transform(X_normal)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# 7. MAHALANOBIS
# ==========================================

mean_vector = np.mean(
    X_normal_scaled,
    axis=0
)

covariance_matrix = np.cov(
    X_normal_scaled,
    rowvar=False
)

inverse_covariance = np.linalg.pinv(
    covariance_matrix
)


def mahalanobis(X):

    difference = X - mean_vector

    return np.sqrt(
        np.sum(
            (difference @ inverse_covariance)
            * difference,
            axis=1
        )
    )


train_distance = mahalanobis(
    X_normal_scaled
)

test_distance = mahalanobis(
    X_test_scaled
)


threshold = np.percentile(
    train_distance,
    97.5
)

mahalanobis_flag = (
    test_distance > threshold
)


# ==========================================
# 8. LOF
# ==========================================

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination="auto",
    novelty=True
)

lof.fit(X_normal_scaled)

lof_prediction = lof.predict(
    X_test_scaled
)

lof_score = -lof.decision_function(
    X_test_scaled
)

lof_flag = (
    lof_prediction == -1
)


# ==========================================
# 9. COMBINE
# ==========================================

anomaly_flag = (
    mahalanobis_flag |
    lof_flag
)


# ==========================================
# 10. RESULTS
# ==========================================

results = test.copy()

results["Mahalanobis_96h"] = test_distance

results["LOF_96h"] = lof_score

results["Mahalanobis_Anomaly_96h"] = (
    mahalanobis_flag
)

results["LOF_Anomaly_96h"] = (
    lof_flag
)

results["Anomaly_Detected_96h"] = (
    anomaly_flag
)

results["Predicted_Anomaly_96h"] = np.where(
    anomaly_flag,
    "Anomaly",
    "Normal"
)


# ==========================================
# 11. EVALUATION
# ==========================================

print("\n========== 96h MODULE A RESULTS ==========")

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        results["Ground_Truth"],
        results["Predicted_Anomaly_96h"],
        labels=["Normal", "Anomaly"]
    )
)

print("\nClassification Report:")

print(
    classification_report(
        results["Ground_Truth"],
        results["Predicted_Anomaly_96h"]
    )
)


# ==========================================
# 12. DETECTION BY TYPE
# ==========================================

print("\n========== DETECTION BY COMPONENT TYPE ==========")

print(
    pd.crosstab(
        results["Component_Type"],
        results["Predicted_Anomaly_96h"]
    )
)


# ==========================================
# 13. SAVE
# ==========================================

results.to_csv(
    "data/processed/module_a_96h_results.csv",
    index=False
)

print(
    "\nSaved: data/processed/module_a_96h_results.csv"
)

print("\n96h anomaly screening complete.")