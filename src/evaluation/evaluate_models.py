"""Evaluate saved window or patient predictions on the held-out test split."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from src.config import LABELS
from src.evaluation.aggregate_patients import aggregate_patients, validate_windows
from src.paths import METRICS_DIR, PREDICTIONS_DIR

ROOT = Path(__file__).resolve().parents[2]
AGGREGATION_METHOD = "recording_mean"


def report_scores(model_name, predictions, suffix=""):
    """Print scores and save a confusion matrix for either prediction level."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    actual, predicted = predictions["true_label"], predictions["pred_label"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=LABELS, average="macro", zero_division=0,
    )
    scores = {
        "Model": model_name, "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision, "Recall": recall, "Macro F1": f1,
    }
    print(f"\n{model_name}: {len(predictions)} {'patients' if suffix else 'windows'}")
    for name in ["Accuracy", "Precision", "Recall", "Macro F1"]:
        print(f"{name}: {scores[name]:.4f}")
    print(classification_report(actual, predicted, labels=LABELS, zero_division=0))
    matrix = confusion_matrix(actual, predicted, labels=LABELS)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
        METRICS_DIR / f"{model_name}{suffix}_confusion_matrix.csv"
    )
    return scores


def evaluate_model(model_name, filename):
    """Evaluate window-level predictions on the test split."""
    windows = pd.read_csv(PREDICTIONS_DIR / filename, dtype={"subject_id": str})
    return report_scores(model_name, windows[windows["split"].eq("test")])


def evaluate_patient_level(model_name, filename):
    """Aggregate test predictions by patient and report their scores."""
    windows = pd.read_csv(PREDICTIONS_DIR / filename, dtype={"subject_id": str})
    validate_windows(windows)
    patients = aggregate_patients(windows[windows["split"].eq("test")], AGGREGATION_METHOD)
    scores = report_scores(model_name, patients, suffix="_patient")
    patients.to_csv(METRICS_DIR / f"{model_name}_patient_predictions.csv", index=False)
    return {"Model": model_name, "Aggregation": AGGREGATION_METHOD, **scores}


def main():
    """Evaluate the saved Random Forest and CNN predictions."""
    models = {"Random_Forest": "random_forest_windows.csv", "CNN": "cnn_windows.csv"}
    results = [evaluate_patient_level(name, filename)
               for name, filename in models.items()]
    comparison = pd.DataFrame(results)
    comparison.to_csv(METRICS_DIR / "patient_model_comparison.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
