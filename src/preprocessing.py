import pandas as pd
import os


# ==========================================
# 1. LOAD RAW DATA
# ==========================================

input_path = "data/raw/burn_in_raw.csv"

df = pd.read_csv(input_path)

print("Raw dataset loaded.")
print(f"Shape: {df.shape}")


# ==========================================
# 2. REMOVE DUPLICATES
# ==========================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print(f"Duplicate rows removed: {before - after}")


# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

missing_values = df.isnull().sum()

print("\nMissing values:")
print(missing_values)


# ==========================================
# 4. MEASUREMENT COLUMNS
# ==========================================

measurement_columns = [
    "Value_0h",
    "Value_24h",
    "Value_96h",
    "Value_168h"
]


# ==========================================
# 5. CHECK INVALID VALUES
# ==========================================

print("\nNegative value check:")

for column in measurement_columns:

    negative_count = (df[column] < 0).sum()

    print(
        f"{column}: {negative_count} negative values"
    )


# ==========================================
# 6. FEATURE ENGINEERING
# ==========================================

# Change between 0h and 24h

df["Drift_0_24"] = (
    df["Value_24h"] - df["Value_0h"]
)


# Percentage change between 0h and 24h

df["Percentage_Drift_0_24"] = (
    (df["Value_24h"] - df["Value_0h"])
    / df["Value_0h"]
) * 100


# Change between 24h and 96h

df["Drift_24_96"] = (
    df["Value_96h"] - df["Value_24h"]
)


# Change between 96h and 168h

df["Drift_96_168"] = (
    df["Value_168h"] - df["Value_96h"]
)


# Overall change from 0h to 168h
# Used mainly for analysis/evaluation,
# NOT as an early prediction feature.

df["Total_Drift_0_168"] = (
    df["Value_168h"] - df["Value_0h"]
)


# ==========================================
# 7. LOT-LEVEL FEATURES
# ==========================================

# Calculate lot mean at 0h

df["Lot_Mean_0h"] = (
    df.groupby("Lot_ID")["Value_0h"]
    .transform("mean")
)


# Calculate deviation from lot mean

df["Deviation_From_Lot_0h"] = (
    df["Value_0h"] - df["Lot_Mean_0h"]
)


# ==========================================
# 8. SAVE PROCESSED DATA
# ==========================================

output_directory = "data/processed"

os.makedirs(output_directory, exist_ok=True)

output_path = (
    f"{output_directory}/burn_in_cleaned.csv"
)

df.to_csv(output_path, index=False)


# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\n========== PREPROCESSING COMPLETE ==========")

print(f"Final shape: {df.shape}")

print("\nFinal columns:")

for column in df.columns:
    print(f"- {column}")

print(
    f"\nProcessed dataset saved to: {output_path}"
)