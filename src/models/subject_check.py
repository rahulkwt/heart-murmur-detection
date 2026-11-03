"""
Quick subject-level sanity check on the VALIDATION split only.

For each validation subject, average the window probabilities over all of
the subject's windows and recording locations, and pick the class with the
highest average. Then print macro-F1, accuracy and per-class F1 for the
Random Forest and the CNN.

This is only a diagnostic, not the final subject-level evaluation. The test
split is never read here.

Run from the project root (after train_rf.py and train_cnn.py):
    python -m src.models.subject_check
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.models.data_utils import OUTPUT_DIR, LABELS


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODELS = {
    "Random Forest": OUTPUT_DIR / "rf_window_predictions.csv",
    "CNN": OUTPUT_DIR / "cnn_window_predictions.csv",
}
SPLIT = "val"   # validation only; do not change this to "test"
PROB_COLUMNS = [f"prob_{label.lower()}" for label in LABELS]


# Average each subject's window probabilities and pick the most likely class.
# Returns one row per subject with its true and predicted label.
def subject_predictions(windows):
    subjects = windows.groupby("subject_id").agg(
        true_label=("true_label", "first"),
        **{column: (column, "mean") for column in PROB_COLUMNS},
    )
    best_column = subjects[PROB_COLUMNS].idxmax(axis=1)
    subjects["pred_label"] = best_column.map(dict(zip(PROB_COLUMNS, LABELS)))
    return subjects


# Compute accuracy, macro-F1 and per-class F1 from subject-level predictions.
def subject_metrics(subjects):
    per_class = f1_score(subjects["true_label"], subjects["pred_label"],
                         labels=LABELS, average=None, zero_division=0)
    return {
        "n_subjects": len(subjects),
        "accuracy": accuracy_score(subjects["true_label"], subjects["pred_label"]),
        "macro_f1": per_class.mean(),
        **{f"f1_{label.lower()}": f1 for label, f1 in zip(LABELS, per_class)},
    }


def main():
    rows = []
    for model_name, path in MODELS.items():
        windows = pd.read_csv(path, dtype={"subject_id": str})
        windows = windows[windows["split"] == SPLIT]

        # Each subject must belong to one split with one label
        assert (windows.groupby("subject_id")["true_label"].nunique() == 1).all()

        metrics = subject_metrics(subject_predictions(windows))
        rows.append({"model": model_name, "split": SPLIT, **metrics})

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "subject_level_val_check.csv", index=False)

    print("Subject-level check on the VALIDATION split (diagnostic only,")
    print("not the final subject-level evaluation)\n")
    print(results.round(4).to_string(index=False))
    print(f"\nSaved to {OUTPUT_DIR / 'subject_level_val_check.csv'}")


if __name__ == "__main__":
    main()
