"""Compare patient aggregation on validation data, without retraining models.

Run from the project root: python -m src.evaluation.aggregate_patients
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import LABELS

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs"
PROBS = [f"prob_{label.lower()}" for label in LABELS]
KEYS = ["subject_id", "recording_id", "window_idx"]


def validate_windows(windows):
    """Reject ambiguous identities, inconsistent labels and invalid probabilities."""
    required = KEYS + ["split", "true_label"] + PROBS
    if windows[required].isna().any().any():
        raise ValueError("Missing window identity, label, split or probability")
    if windows.duplicated(KEYS).any():
        raise ValueError("Duplicate window identity")
    if not windows["true_label"].isin(LABELS).all():
        raise ValueError("Unexpected label")
    if not windows["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Unexpected split")
    consistency = windows.groupby("subject_id")[["split", "true_label"]].nunique()
    if (consistency != 1).any().any():
        raise ValueError("A patient has multiple labels or splits")
    values = windows[PROBS].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or (values > 1).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid class probabilities")


def aggregate_patients(windows, method):
    """Return one prediction per patient and split; ties use LABELS order.

    window_mean gives every window equal weight. recording_mean first averages
    windows within each recording, then gives every recording equal weight.
    """
    validate_windows(windows)
    group = ["subject_id", "split", "true_label"]
    if method == "window_mean":
        probabilities = windows.groupby(group, sort=True)[PROBS].mean()
    elif method == "recording_mean":
        recordings = windows.groupby(group + ["recording_id"], sort=True)[PROBS].mean()
        probabilities = recordings.groupby(level=group)[PROBS].mean()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    result = probabilities.reset_index()
    counts = windows.groupby(group).agg(
        n_windows=("window_idx", "size"), n_recordings=("recording_id", "nunique")
    ).reset_index()
    result = result.merge(counts, on=group, validate="one_to_one")
    result["pred_label"] = np.asarray(LABELS)[result[PROBS].to_numpy().argmax(axis=1)]
    result.insert(0, "method", method)
    return result


def metrics(patients):
    precision, recall, f1, support = precision_recall_fscore_support(
        patients["true_label"], patients["pred_label"], labels=LABELS, zero_division=0
    )
    result = {"n_subjects": len(patients),
              "accuracy": accuracy_score(patients["true_label"], patients["pred_label"]),
              "macro_f1": f1.mean()}
    for index, label in enumerate(LABELS):
        for name, values in [("precision", precision), ("recall", recall),
                             ("f1", f1), ("support", support)]:
            result[f"{name}_{label.lower()}"] = values[index]
    return result


def main():
    destination = OUTPUT / "patient_aggregation"
    destination.mkdir(parents=True, exist_ok=True)
    summaries = []
    reference = None
    for model in ["rf", "cnn"]:
        windows = pd.read_csv(OUTPUT / f"{model}_window_predictions.csv",
                              dtype={"subject_id": str, "recording_id": str})
        validate_windows(windows)
        # Inspect identities across splits for integrity; evaluate validation only.
        identities = windows[KEYS + ["split", "true_label"]].sort_values(KEYS).reset_index(drop=True)
        if reference is not None and not identities.equals(reference):
            raise ValueError("Models do not have identical windows, labels and splits")
        reference = identities
        validation = windows.loc[windows["split"] == "val"]
        if validation.empty:
            raise ValueError("No validation windows")
        for method in ["window_mean", "recording_mean"]:
            patients = aggregate_patients(validation, method)
            patients.to_csv(destination / f"{model}_{method}_val_patients.csv", index=False)
            summaries.append({"model": model, "method": method, "split": "val",
                              **metrics(patients)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(destination / "validation_comparison.csv", index=False)
    print(summary[["model", "method", "n_subjects", "accuracy", "macro_f1",
                   "recall_present", "f1_unknown"]].round(4).to_string(index=False))
    print(f"\nSaved validation patient predictions and comparison to {destination}")


if __name__ == "__main__":
    main()
