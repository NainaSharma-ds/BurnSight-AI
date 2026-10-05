import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# LOAD PROCESSED DATA
# ==========================================

file_path = "data/processed/burn_in_cleaned.csv"

df = pd.read_csv(file_path)

print("Processed dataset loaded.")
print(f"Shape: {df.shape}")


# ==========================================
# 1. COMPONENT TYPE DISTRIBUTION
# ==========================================

print("\n========== COMPONENT TYPE DISTRIBUTION ==========")

print(
    df["Component_Type"]
    .value_counts()
)


# ==========================================
# 2. GROUND TRUTH DISTRIBUTION
# ==========================================

print("\n========== GROUND TRUTH DISTRIBUTION ==========")

print(
    df["Ground_Truth"]
    .value_counts()
)


# ==========================================
# 3. AVERAGE VALUES BY COMPONENT TYPE
# ==========================================

measurement_columns = [
    "Value_0h",
    "Value_24h",
    "Value_96h",
    "Value_168h"
]

print("\n========== AVERAGE VALUES BY COMPONENT TYPE ==========")

type_means = (
    df.groupby("Component_Type")[measurement_columns]
    .mean()
    .round(3)
)

print(type_means)


# ==========================================
# 4. MEDIAN VALUES BY COMPONENT TYPE
# ==========================================

print("\n========== MEDIAN VALUES BY COMPONENT TYPE ==========")

type_medians = (
    df.groupby("Component_Type")[measurement_columns]
    .median()
    .round(3)
)

print(type_medians)


# ==========================================
# 5. EARLY DRIFT BY COMPONENT TYPE
# ==========================================

print("\n========== EARLY DRIFT BY COMPONENT TYPE ==========")

drift_means = (
    df.groupby("Component_Type")[
        [
            "Drift_0_24",
            "Percentage_Drift_0_24"
        ]
    ]
    .mean()
    .round(3)
)

print(drift_means)


# ==========================================
# 6. CORRELATION WITH 168h
# ==========================================

print("\n========== CORRELATION WITH VALUE_168h ==========")

correlation_columns = [
    "Value_0h",
    "Value_24h",
    "Drift_0_24",
    "Percentage_Drift_0_24",
    "Value_96h"
]

correlations = (
    df[correlation_columns + ["Value_168h"]]
    .corr()["Value_168h"]
    .sort_values(ascending=False)
)

print(correlations)


# ==========================================
# 7. LOT STATISTICS
# ==========================================

print("\n========== LOT STATISTICS ==========")

lot_statistics = (
    df.groupby("Lot_ID")["Value_0h"]
    .agg(["mean", "std", "min", "max"])
    .round(3)
)

print(lot_statistics)


# ==========================================
# 8. SAVE SUMMARY
# ==========================================

type_means.to_csv(
    "data/processed/component_type_means.csv"
)

lot_statistics.to_csv(
    "data/processed/lot_statistics.csv"
)


# ==========================================
# 9. VISUALIZATION — COMPONENT BEHAVIOUR
# ==========================================

time_points = [
    "Value_0h",
    "Value_24h",
    "Value_96h",
    "Value_168h"
]

time_labels = [
    "0h",
    "24h",
    "96h",
    "168h"
]

plt.figure(figsize=(10, 6))

for component_type in df["Component_Type"].unique():

    averages = (
        df[df["Component_Type"] == component_type][time_points]
        .mean()
    )

    plt.plot(
        time_labels,
        averages.values,
        marker="o",
        label=component_type
    )

plt.title("Burn-In Behaviour by Component Type")

plt.xlabel("Burn-In Time")

plt.ylabel("Measurement Value")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "data/processed/component_behaviour.png",
    dpi=300
)

plt.show()


# ==========================================
# 10. 0h vs 168h SCATTER
# ==========================================

plt.figure(figsize=(8, 6))

for component_type in df["Component_Type"].unique():

    subset = df[
        df["Component_Type"] == component_type
    ]

    plt.scatter(
        subset["Value_24h"],
        subset["Value_168h"],
        label=component_type,
        alpha=0.6
    )

plt.xlabel("Value at 24h")

plt.ylabel("Value at 168h")

plt.title("Early Measurement vs 168h Measurement")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "data/processed/early_vs_168h.png",
    dpi=300
)

plt.show()


print("\n========== FEATURE ANALYSIS COMPLETE ==========")

print(
    "Analysis files saved inside data/processed/"
)