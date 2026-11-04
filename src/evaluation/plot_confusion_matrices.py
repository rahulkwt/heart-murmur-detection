
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

plt.style.use("dark_background")

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "outputs" / "evaluation"
FIG_DIR = EVAL_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from src.config import LABELS

models = {
    "Random Forest": "Random_Forest",
    "CNN": "CNN"
}

for model_name, filename in models.items():

    # Load patient-level predictions
    df = pd.read_csv(
        EVAL_DIR / f"{filename}_patient_predictions.csv",
        dtype={"subject_id": str}
    )

    # Calculate confusion matrix
    cm = confusion_matrix(
        df["true_label"],
        df["pred_label"],
        labels=LABELS
    )

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABELS
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    plt.title(f"{model_name} - Patient-Level Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    # Save figure
    plt.savefig(
        FIG_DIR / f"{filename}_confusion_matrix.png",
        dpi=300
    )

    plt.show()

print("Confusion matrix figures saved successfully!")
