import pandas as pd
import os


# ==========================================
# 1. LOAD PROCESSED DATA
# ==========================================

input_path = "data/processed/burn_in_cleaned.csv"

df = pd.read_csv(input_path)

print("Dataset loaded.")
print(f"Total rows: {len(df)}")


# ==========================================
# 2. DEFINE TRAINING AND TEST LOTS
# ==========================================

train_lots = [
    "L01",
    "L02",
    "L03",
    "L04",
    "L05",
    "L06",
    "L07",
    "L08"
]

test_lots = [
    "L09",
    "L10"
]


# ==========================================
# 3. SPLIT DATA BY LOT
# ==========================================

train_df = df[df["Lot_ID"].isin(train_lots)].copy()

test_df = df[df["Lot_ID"].isin(test_lots)].copy()


# ==========================================
# 4. DISPLAY SPLIT INFORMATION
# ==========================================

print("\n========== TRAINING DATA ==========")

print(f"Rows: {len(train_df)}")

print(
    "Lots:",
    sorted(train_df["Lot_ID"].unique())
)


print("\n========== TEST DATA ==========")

print(f"Rows: {len(test_df)}")

print(
    "Lots:",
    sorted(test_df["Lot_ID"].unique())
)


# ==========================================
# 5. COMPONENT TYPE DISTRIBUTION
# ==========================================

print("\n========== TRAINING COMPONENT TYPES ==========")

print(
    train_df["Component_Type"]
    .value_counts()
)


print("\n========== TEST COMPONENT TYPES ==========")

print(
    test_df["Component_Type"]
    .value_counts()
)


# ==========================================
# 6. GROUND TRUTH DISTRIBUTION
# ==========================================

print("\n========== TRAINING GROUND TRUTH ==========")

print(
    train_df["Ground_Truth"]
    .value_counts()
)


print("\n========== TEST GROUND TRUTH ==========")

print(
    test_df["Ground_Truth"]
    .value_counts()
)


# ==========================================
# 7. CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ==========================================
# 8. SAVE TRAINING DATA
# ==========================================

train_path = (
    "data/processed/train.csv"
)

test_path = (
    "data/processed/test.csv"
)

train_df.to_csv(
    train_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


# ==========================================
# 9. FINAL MESSAGE
# ==========================================

print("\n========== SPLIT COMPLETE ==========")

print(
    f"Training data saved to: {train_path}"
)

print(
    f"Test data saved to: {test_path}"
)