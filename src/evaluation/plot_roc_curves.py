"""Plot patient ROC curves and save macro ROC-AUC scores."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import LABELS
from src.evaluation.aggregate_patients import PROBS
from src.evaluation.comparison import plot_patient_roc
from src.paths import FIGURES_DIR, METRICS_DIR

FIG_DIR = FIGURES_DIR / "evaluation"


def main():
    """Plot patient ROC curves and save macro ROC-AUC scores."""
    plt.style.use("dark_background")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name, filename in {"Random Forest": "Random_Forest", "CNN": "CNN"}.items():
        patients = pd.read_csv(METRICS_DIR / f"{filename}_patient_predictions.csv",
                               dtype={"subject_id": str})
        targets = patients["true_label"].to_numpy()[:, None] == np.array(LABELS)
        score = roc_auc_score(targets, patients[PROBS], average="macro")
        results.append({"Model": name, "Macro ROC-AUC": score})
        print(f"{name} Macro ROC-AUC: {score:.4f}")
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_patient_roc(ax, patients, f"{name} - Patient-Level ROC Curves")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{filename}_roc_curves.png", dpi=300)
        plt.show()
        plt.close(fig)
    pd.DataFrame(results).to_csv(METRICS_DIR / "roc_auc_comparison.csv", index=False)


if __name__ == "__main__":
    main()
