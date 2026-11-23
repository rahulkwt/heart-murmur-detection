"""Calculate a validation-only subject-level diagnostic."""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import LABELS
from src.paths import METRICS_DIR, PREDICTIONS_DIR


MODELS = {
    "Random Forest": PREDICTIONS_DIR / "random_forest_windows.csv",
    "CNN": PREDICTIONS_DIR / "cnn_windows.csv",
}
SPLIT = "val"
PROB_COLUMNS = [f"prob_{label.lower()}" for label in LABELS]


def subject_predictions(windows):
    """Average window probabilities and choose one class per subject."""
    subjects = windows.groupby("subject_id").agg(
        true_label=("true_label", "first"),
        **{column: (column, "mean") for column in PROB_COLUMNS},
    )
    best_column = subjects[PROB_COLUMNS].idxmax(axis=1)
    subjects["pred_label"] = best_column.map(dict(zip(PROB_COLUMNS, LABELS)))
    return subjects


def subject_metrics(subjects):
    """Calculate subject-level accuracy and F1 scores."""
    per_class = f1_score(subjects["true_label"], subjects["pred_label"],
                         labels=LABELS, average=None, zero_division=0)
    return {
        "n_subjects": len(subjects),
        "accuracy": accuracy_score(subjects["true_label"], subjects["pred_label"]),
        "macro_f1": per_class.mean(),
        **{f"f1_{label.lower()}": f1 for label, f1 in zip(LABELS, per_class)},
    }


def main():
    """Write and print the validation subject-level comparison."""
    rows = []
    for model_name, path in MODELS.items():
        windows = pd.read_csv(path, dtype={"subject_id": str})
        windows = windows[windows["split"] == SPLIT]

        assert (windows.groupby("subject_id")["true_label"].nunique() == 1).all()

        metrics = subject_metrics(subject_predictions(windows))
        rows.append({"model": model_name, "split": SPLIT, **metrics})

    results = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS_DIR / "subject_level_val_check.csv", index=False)

    print("Subject-level check on the VALIDATION split (diagnostic only,")
    print("not the final subject-level evaluation)\n")
    print(results.round(4).to_string(index=False))
    print(f"\nSaved to {METRICS_DIR / 'subject_level_val_check.csv'}")


if __name__ == "__main__":
    main()
