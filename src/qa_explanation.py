import pandas as pd
import os


# ==============================
# LOAD RISK RESULTS
# ==============================

results = pd.read_csv(
    "data/processed/risk_engine_results.csv"
)


# ==============================
# EXPLANATION FUNCTION
# ==============================

def explain_component(row):

    explanation = []

    # 1. Anomaly status
    if row["Anomaly_Status"] == "ANOMALY":
        explanation.append(
            "Anomalous burn-in behaviour detected."
        )

    elif row["Anomaly_Status"] == "EARLY_WARNING":
        explanation.append(
            "Early anomaly warning detected at 24h."
        )

    else:
        explanation.append(
            "No anomaly detected during screening."
        )

    # 2. Predicted future behaviour
    explanation.append(
        f"Predicted 168h value: "
        f"{row['Predicted_168h']:.2f}"
    )

    # 3. Risk ratio
    explanation.append(
        f"Risk ratio: "
        f"{row['Risk_Ratio']:.2f}"
    )

    # 4. QA decision
    if row["QA_Decision"] == "REJECT":

        explanation.append(
            "QA decision: REJECT because abnormal "
            "behaviour or excessive predicted degradation "
            "was detected."
        )

    elif row["QA_Decision"] == "EXTENDED_SCREENING":

        explanation.append(
            "QA decision: EXTENDED SCREENING because "
            "predicted degradation exceeds the normal "
            "reference boundary."
        )

    elif row["QA_Decision"] == "MONITOR":

        explanation.append(
            "QA decision: MONITOR because the component "
            "is approaching the reference safety boundary."
        )

    else:

        explanation.append(
            "QA decision: PASS because behaviour remains "
            "within the expected range."
        )

    return " ".join(explanation)


# ==============================
# GENERATE EXPLANATIONS
# ==============================

results["Detailed_Explanation"] = results.apply(
    explain_component,
    axis=1
)


# ==============================
# SAVE
# ==============================

output_path = (
    "data/processed/qa_explanations.csv"
)

results.to_csv(
    output_path,
    index=False
)


# ==============================
# SHOW EXAMPLES
# ==============================

print("\n========== QA EXPLANATION EXAMPLES ==========")

print(
    results[
        [
            "Component_ID",
            "QA_Decision",
            "Detailed_Explanation"
        ]
    ].head(10).to_string(index=False)
)

print("\nSaved to:")
print(output_path)