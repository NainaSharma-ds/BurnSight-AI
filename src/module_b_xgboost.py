import pandas as pd
import numpy as np
import os

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

print("Training data:", train.shape)
print("Test data:", test.shape)


# ==========================================
# 2. FEATURES
# ==========================================

features = [
    "Value_0h",
    "Value_24h",
    "Drift_0_24",
    "Percentage_Drift_0_24"
]

target = "Value_168h"

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


# ==========================================
# 3. CREATE XGBOOST MODEL
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


# ==========================================
# 4. TRAIN
# ==========================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ==========================================
# 5. PREDICTION
# ==========================================

predicted_168h = model.predict(
    X_test
)


# ==========================================
# 6. EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    predicted_168h
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predicted_168h
    )
)

r2 = r2_score(
    y_test,
    predicted_168h
)


print("\n========== MODULE B RESULTS ==========")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ==========================================
# 7. CREATE RESULTS
# ==========================================

results = test.copy()

results["Predicted_168h"] = predicted_168h

results["Prediction_Error"] = (
    results["Value_168h"]
    - results["Predicted_168h"]
)

results["Absolute_Error"] = (
    results["Prediction_Error"]
    .abs()
)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

results.to_csv(
    "data/processed/module_b_results.csv",
    index=False
)


# ==========================================
# 9. SAVE MODEL
# ==========================================

os.makedirs(
    "model",
    exist_ok=True
)

model.save_model(
    "model/xgboost_168h.json"
)


# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")

print(
    importance.to_string(index=False)
)


print(
    "\nResults saved to:"
    "\ndata/processed/module_b_results.csv"
)

print(
    "\nModel saved to:"
    "\nmodel/xgboost_168h.json"
)

print("\nModule B complete.")