"""Compare the saved patient-level metrics in a grouped bar chart."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parents[2] / "outputs/evaluation"
FIG_DIR = EVAL_DIR / "figures"


def main():
    plt.style.use("dark_background")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(EVAL_DIR / "patient_model_comparison.csv")
    roc = pd.read_csv(EVAL_DIR / "roc_auc_comparison.csv")
    roc["Model"] = roc["Model"].replace({"Random Forest": "Random_Forest"})
    scores = comparison.merge(roc, on="Model", validate="one_to_one")
    scores["Model"] = scores["Model"].str.replace("_", " ")
    columns = ["Accuracy", "Precision", "Recall", "Macro F1", "Macro ROC-AUC"]
    ax = scores.set_index("Model")[columns].T.plot.bar(
        figsize=(10, 6), rot=0, ylim=(0, 1.1), width=0.7,
    )
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(title="Patient-Level Model Performance Comparison", ylabel="Score", xlabel="")
    ax.legend(title=None)
    ax.figure.tight_layout()
    ax.figure.savefig(FIG_DIR / "model_comparison.png", dpi=300)
    plt.show()
    plt.close(ax.figure)


if __name__ == "__main__":
    main()
