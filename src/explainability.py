import pandas as pd
import numpy as np
import os

from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error


# ==============================
# LOAD DATA
# ==============================

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

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


# ==============================
# TRAIN MODEL
# ==============================

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

print("Training XGBoost...")

model.fit(X_train, y_train)

print("Training complete.")


# ==============================
# PERMUTATION IMPORTANCE
# ==============================

print("\nCalculating permutation importance...")

importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=42
)


importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance.importances_mean
})


importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


# ==============================
# DISPLAY
# ==============================

print("\n========== PERMUTATION IMPORTANCE ==========")

print(
    importance_df.to_string(index=False)
)


# ==============================
# SAVE
# ==============================

os.makedirs(
    "data/processed",
    exist_ok=True
)

importance_df.to_csv(
    "data/processed/permutation_importance.csv",
    index=False
)


print("\nSaved to:")
print("data/processed/permutation_importance.csv")