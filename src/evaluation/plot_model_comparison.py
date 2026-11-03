
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "outputs" / "evaluation"
FIG_DIR = EVAL_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load patient-level evaluation results
comparison = pd.read_csv(
    EVAL_DIR / "patient_model_comparison.csv"
)

# Load ROC-AUC results
roc = pd.read_csv(
    EVAL_DIR / "roc_auc_comparison.csv"
)

# Match model names
roc["Model"] = roc["Model"].replace({
    "Random Forest": "Random_Forest"
})

# Combine evaluation metrics
df = comparison.merge(roc, on="Model", validate="one_to_one")

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "Macro F1",
    "Macro ROC-AUC"
]

# Plot grouped bars
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

for i, (_, row) in enumerate(df.iterrows()):
    values = [row[metric] for metric in metrics]

    bars = ax.bar(
        x + (i - (len(df) - 1) / 2) * width,
        values,
        width,
        label=row["Model"].replace("_", " ")
    )

    ax.bar_label(bars, fmt="%.2f", padding=3)

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Patient-Level Model Performance Comparison")
ax.legend()

plt.tight_layout()
plt.savefig(
    FIG_DIR / "model_comparison.png",
    dpi=300
)
plt.show()

print("Model comparison chart saved successfully!")
