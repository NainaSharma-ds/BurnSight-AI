import pandas as pd

# ==============================
# LOAD DATASET
# ==============================

file_path = "data/raw/burn_in_raw.csv"

df = pd.read_csv(file_path)


# ==============================
# BASIC INFORMATION
# ==============================

print("\n========== DATASET INFO ==========")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ==============================
# COMPONENT TYPE DISTRIBUTION
# ==============================

print("\n========== COMPONENT TYPE DISTRIBUTION ==========")
print(df["Component_Type"].value_counts())


# ==============================
# GROUND TRUTH DISTRIBUTION
# ==============================

print("\n========== GROUND TRUTH DISTRIBUTION ==========")
print(df["Ground_Truth"].value_counts())


# ==============================
# LOT DISTRIBUTION
# ==============================

print("\n========== COMPONENTS PER LOT ==========")
print(df["Lot_ID"].value_counts().sort_index())


# ==============================
# MEASUREMENT STATISTICS
# ==============================

measurement_columns = [
    "Value_0h",
    "Value_24h",
    "Value_96h",
    "Value_168h"
]

print("\n========== MEASUREMENT STATISTICS ==========")
print(df[measurement_columns].describe())


# ==============================
# NEGATIVE VALUE CHECK
# ==============================

print("\n========== NEGATIVE VALUES ==========")

for column in measurement_columns:
    count = (df[column] < 0).sum()
    print(f"{column}: {count}")


# ==============================
# AVERAGE VALUE BY COMPONENT TYPE
# ==============================

print("\n========== AVERAGE VALUE BY COMPONENT TYPE ==========")

print(
    df.groupby("Component_Type")[measurement_columns]
    .mean()
    .round(3)
)


# ==============================
# GROUND TRUTH BY COMPONENT TYPE
# ==============================

print("\n========== COMPONENT TYPE vs GROUND TRUTH ==========")

print(
    pd.crosstab(
        df["Component_Type"],
        df["Ground_Truth"]
    )
)


# ==============================
# SAMPLE DATA
# ==============================

print("\n========== SAMPLE DATA ==========")
print(df.head(10))