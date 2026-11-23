"""Plot confusion matrices from saved patient predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import LABELS
from src.paths import FIGURES_DIR, METRICS_DIR

FIG_DIR = FIGURES_DIR / "evaluation"


def main():
    """Plot confusion matrices from saved patient predictions."""
    plt.style.use("dark_background")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, filename in {"Random Forest": "Random_Forest", "CNN": "CNN"}.items():
        patients = pd.read_csv(METRICS_DIR / f"{filename}_patient_predictions.csv",
                               dtype={"subject_id": str})
        fig, ax = plt.subplots(figsize=(7, 5))
        ConfusionMatrixDisplay.from_predictions(
            patients["true_label"], patients["pred_label"], labels=LABELS,
            cmap="Blues", values_format="d", colorbar=False, ax=ax,
        )
        ax.set(title=f"{name} - Patient-Level Confusion Matrix",
               xlabel="Predicted Label", ylabel="True Label")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{filename}_confusion_matrix.png", dpi=300)
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
