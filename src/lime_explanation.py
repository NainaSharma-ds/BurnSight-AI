import pandas as pd
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer

# =========================
# LOAD DATA
# =========================

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

features = [
    "Value_0h",
    "Value_24h",
    "Drift_0_24",
    "Percentage_Drift_0_24"
]

X_train = train[features]
X_test = test[features]

# =========================
# TRAIN XGBOOST MODEL
# =========================

from xgboost import XGBRegressor

y_train = train["Value_168h"]

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

# =========================
# CREATE LIME EXPLAINER
# =========================

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=features,
    mode="regression",
    random_state=42
)

# =========================
# EXPLAIN TEST COMPONENTS
# =========================

all_explanations = []

for index in range(min(10, len(X_test))):

    instance = X_test.iloc[index].values

    explanation = explainer.explain_instance(
        instance,
        model.predict,
        num_features=4
    )

    component_id = test.iloc[index]["Component_ID"]

    predicted_value = model.predict(
        X_test.iloc[[index]]
    )[0]

    print("\n======================================")
    print("Component:", component_id)
    print("Predicted 168h Value:", round(predicted_value, 2))
    print("======================================")

    for feature, weight in explanation.as_list():
        print(f"{feature}: {weight:.4f}")

    for feature, weight in explanation.as_list():

        all_explanations.append({
            "Component_ID": component_id,
            "Predicted_168h": predicted_value,
            "Feature": feature,
            "LIME_Weight": weight
        })

# =========================
# SAVE RESULTS
# =========================

output = pd.DataFrame(all_explanations)

output_path = "data/processed/lime_explanations.csv"

output.to_csv(
    output_path,
    index=False
)

print("\n======================================")
print("LIME explanations saved to:")
print(output_path)
print("======================================")