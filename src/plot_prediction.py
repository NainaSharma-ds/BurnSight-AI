import pandas as pd
import matplotlib.pyplot as plt
import os

results = pd.read_csv("data/processed/module_b_results.csv")

plt.figure(figsize=(8, 6))

plt.scatter(
    results["Value_168h"],
    results["Predicted_168h"],
    alpha=0.7
)

min_val = min(
    results["Value_168h"].min(),
    results["Predicted_168h"].min()
)

max_val = max(
    results["Value_168h"].max(),
    results["Predicted_168h"].max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Actual Value at 168h")
plt.ylabel("Predicted Value at 168h")
plt.title("Actual vs Predicted 168h Component Value")

plt.tight_layout()

os.makedirs("data/processed", exist_ok=True)

plt.savefig(
    "data/processed/actual_vs_predicted_168h.png",
    dpi=300
)

plt.show()

print("Plot saved to:")
print("data/processed/actual_vs_predicted_168h.png")