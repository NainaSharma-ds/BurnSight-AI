import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. LOAD DATA
# ==========================================

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

print("Training data:", train.shape)
print("Test data:", test.shape)


# ==========================================
# 2. FEATURES FOR ANOMALY DETECTION
# ==========================================

features = [
    "Value_0h",
    "Value_24h",
    "Drift_0_24",
    "Percentage_Drift_0_24",
    "Deviation_From_Lot_0h"
]

X_train = train[features].copy()
X_test = test[features].copy()


# ==========================================
# 3. USE NORMAL TRAINING COMPONENTS
# ==========================================

normal_train = train[
    train["Ground_Truth"] == "Normal"
].copy()

X_normal = normal_train[features].copy()

print("\nNormal training components:", len(X_normal))


# ==========================================
# 4. SCALE FEATURES
# ==========================================

scaler = StandardScaler()

X_normal_scaled = scaler.fit_transform(X_normal)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. MAHALANOBIS DISTANCE
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


def mahalanobis_distance(X):

    difference = X - mean_vector

    distance = np.sqrt(
        np.sum(
            (difference @ inverse_covariance)
            * difference,
            axis=1
        )
    )

    return distance


train_mahalanobis = mahalanobis_distance(
    X_normal_scaled
)

test_mahalanobis = mahalanobis_distance(
    X_test_scaled
)


# ==========================================
# 6. MAHALANOBIS THRESHOLD
# ==========================================

mahalanobis_threshold = np.percentile(
    train_mahalanobis,
    97.5
)

test_mahalanobis_flag = (
    test_mahalanobis > mahalanobis_threshold
)


# ==========================================
# 7. LOCAL OUTLIER FACTOR
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
# 8. COMBINED ANOMALY DECISION
# ==========================================

combined_anomaly = (
    test_mahalanobis_flag |
    lof_flag
)


# ==========================================
# 9. CREATE RESULTS
# ==========================================

results = test.copy()

results["Mahalanobis_Distance"] = (
    test_mahalanobis
)

results["Mahalanobis_Anomaly"] = (
    test_mahalanobis_flag
)

results["LOF_Score"] = lof_score

results["LOF_Anomaly"] = lof_flag

results["Anomaly_Detected"] = (
    combined_anomaly
)


# ==========================================
# 10. CONVERT TO LABEL
# ==========================================

results["Predicted_Anomaly"] = np.where(
    results["Anomaly_Detected"],
    "Anomaly",
    "Normal"
)


# ==========================================
# 11. EVALUATION
# ==========================================

actual = results["Ground_Truth"]

predicted = results["Predicted_Anomaly"]

print("\n========== MODULE A RESULTS ==========")

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        actual,
        predicted,
        labels=["Normal", "Anomaly"]
    )
)

print("\nClassification Report:")

print(
    classification_report(
        actual,
        predicted
    )
)

print(
    "\nPrecision:",
    round(
        precision_score(
            actual,
            predicted,
            pos_label="Anomaly"
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            actual,
            predicted,
            pos_label="Anomaly"
        ),
        4
    )
)

print(
    "F1 Score:",
    round(
        f1_score(
            actual,
            predicted,
            pos_label="Anomaly"
        ),
        4
    )
)


# ==========================================
# 12. SAVE RESULTS
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

results.to_csv(
    "data/processed/module_a_results.csv",
    index=False
)


# ==========================================
# 13. SAVE MODEL INFORMATION
# ==========================================

os.makedirs(
    "model",
    exist_ok=True
)

import joblib

joblib.dump(
    scaler,
    "model/anomaly_scaler.pkl"
)

joblib.dump(
    lof,
    "model/lof_model.pkl"
)

joblib.dump(
    mean_vector,
    "model/mahalanobis_mean.pkl"
)

joblib.dump(
    inverse_covariance,
    "model/mahalanobis_inverse_covariance.pkl"
)

joblib.dump(
    mahalanobis_threshold,
    "model/mahalanobis_threshold.pkl"
)


print("\nResults saved:")
print("data/processed/module_a_results.csv")

print("\nModule A complete.")