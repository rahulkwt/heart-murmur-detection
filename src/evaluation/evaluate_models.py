
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Project directories
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"
EVAL_DIR = OUTPUT_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

from src.config import LABELS
from src.evaluation.aggregate_patients import aggregate_patients, validate_windows

AGGREGATION_METHOD = "recording_mean"


def evaluate_model(model_name, filename):
    # Load saved predictions
    df = pd.read_csv(OUTPUT_DIR / filename, dtype={"subject_id": str})

    # Evaluate only the held-out test set
    test = df[df["split"] == "test"].copy()

    print(f"\n===== {model_name} =====")
    print("Test windows:", len(test))

    # True and predicted labels
    y_true = test["true_label"]
    y_pred = test["pred_label"]

    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Macro F1:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=LABELS,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    cm_df = pd.DataFrame(
        cm,
        index=LABELS,
        columns=LABELS
    )
    cm_df.to_csv(EVAL_DIR / f"{model_name}_confusion_matrix.csv")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": f1
    }


def evaluate_patient_level(model_name, filename):
    # Load predictions
    df = pd.read_csv(
        OUTPUT_DIR / filename,
        dtype={"subject_id": str}
    )

    validate_windows(df)
    test = df[df["split"] == "test"].copy()
    patients = aggregate_patients(test, AGGREGATION_METHOD)

    # Calculate evaluation metrics
    y_true = patients["true_label"]
    y_pred = patients["pred_label"]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    print(f"\n===== {model_name}: PATIENT LEVEL =====")
    print("Test patients:", len(patients))
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Macro F1:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=LABELS,
        zero_division=0
    ))

    # Save patient predictions
    patients.to_csv(
        EVAL_DIR / f"{model_name}_patient_predictions.csv",
        index=False
    )

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    pd.DataFrame(
        cm, index=LABELS, columns=LABELS
    ).to_csv(
        EVAL_DIR / f"{model_name}_patient_confusion_matrix.csv"
    )

    return {
        "Model": model_name,
        "Aggregation": AGGREGATION_METHOD,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": f1
    }


def main():
    results = []

    results.append(evaluate_patient_level(
        "Random_Forest",
        "rf_window_predictions.csv"
    ))

    results.append(evaluate_patient_level(
        "CNN",
        "cnn_window_predictions.csv"
    ))

    comparison = pd.DataFrame(results)

    print("\n===== PATIENT-LEVEL MODEL COMPARISON =====")
    print(comparison.to_string(index=False))

    comparison.to_csv(
        EVAL_DIR / "patient_model_comparison.csv",
        index=False
    )

    print("\nPatient-level evaluation completed!")


if __name__ == "__main__":
    main()
