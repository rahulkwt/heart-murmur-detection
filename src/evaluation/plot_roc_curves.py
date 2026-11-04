
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

plt.style.use("dark_background")

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "outputs" / "evaluation"
FIG_DIR = EVAL_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from src.config import LABELS
PROB_COLS = ["prob_present", "prob_absent", "prob_unknown"]

MODELS = {
    "Random Forest": "Random_Forest",
    "CNN": "CNN"
}

results = []

for model_name, filename in MODELS.items():

    # Load patient-level test predictions
    df = pd.read_csv(
        EVAL_DIR / f"{filename}_patient_predictions.csv",
        dtype={"subject_id": str}
    )

    y_true = label_binarize(df["true_label"], classes=LABELS)
    y_score = df[PROB_COLS].to_numpy()

    # Multiclass one-vs-rest ROC-AUC
    # Calculate one-vs-rest AUC for each class
    class_aucs = []

    for i, label in enumerate(LABELS):
        class_auc = roc_auc_score(
        y_true[:, i],
        y_score[:, i]
        )
        class_aucs.append(class_auc)

# Average the three class AUC scores
    macro_auc = np.mean(class_aucs)

    print(f"{model_name} Macro ROC-AUC: {macro_auc:.4f}")

    results.append({
        "Model": model_name,
        "Macro ROC-AUC": macro_auc
    })

    # Plot each class against the other two
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        class_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{label} (AUC = {class_auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", label="Random classifier")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} - Patient-Level ROC Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{filename}_roc_curves.png", dpi=300)
    plt.show()

pd.DataFrame(results).to_csv(
    EVAL_DIR / "roc_auc_comparison.csv",
    index=False
)

print("\nROC-AUC evaluation completed!")

