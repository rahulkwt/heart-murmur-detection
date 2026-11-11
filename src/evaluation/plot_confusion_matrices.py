"""Plot confusion matrices from saved patient predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import LABELS

EVAL_DIR = Path(__file__).resolve().parents[2] / "outputs/evaluation"
FIG_DIR = EVAL_DIR / "figures"


def main():
    plt.style.use("dark_background")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, filename in {"Random Forest": "Random_Forest", "CNN": "CNN"}.items():
        patients = pd.read_csv(EVAL_DIR / f"{filename}_patient_predictions.csv",
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
