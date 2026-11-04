"""
Train a Random Forest on MFCC summary features, one prediction per window.

The hyperparameters are tuned on the validation set only. The test set is
used only to write the final predictions.

Run from the project root:
    python -m src.models.train_rf
"""

import itertools

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.config import LABELS
from src.models.data_utils import (
    OUTPUT_DIR, SEED, set_seeds, load_data, encode_labels,
    make_predictions_table, save_label_mapping,
)


# ---------------------------------------------------------------------------
# CONFIG: hyperparameter grid (every combination is tried)
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [10, None],
    # Larger leaves stop the trees from memorizing the training windows
    "min_samples_leaf": [5, 20, 50],
}
N_JOBS = -1     # use all CPU cores


# Train one Random Forest with the given settings on the training windows.
# class_weight="balanced" gives the rare classes (Present, Unknown) more weight.
def train_forest(x_train, y_train, params):
    model = RandomForestClassifier(
        **params,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    model.fit(x_train, y_train)
    return model


# Try every combination in PARAM_GRID and keep the one with the best
# validation macro-F1. Returns the best model and a table of all results.
def tune_forest(x_train, y_train, x_val, y_val):
    names = list(PARAM_GRID)
    combinations = list(itertools.product(*PARAM_GRID.values()))
    results = []
    best_model, best_f1 = None, -1.0

    for number, values in enumerate(combinations, start=1):
        params = dict(zip(names, values))
        model = train_forest(x_train, y_train, params)
        val_f1 = f1_score(y_val, model.predict(x_val), average="macro",
                          labels=range(len(LABELS)))
        # Train macro-F1 is only reported to show overfitting (train >> val);
        # the choice of settings uses validation macro-F1 alone
        train_f1 = f1_score(y_train, model.predict(x_train), average="macro",
                            labels=range(len(LABELS)))
        results.append({**params, "train_macro_f1": train_f1, "val_macro_f1": val_f1})
        print(f"  [{number}/{len(combinations)}] {params} -> "
              f"train macro-F1 {train_f1:.4f}, val macro-F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_model, best_f1 = model, val_f1

    return best_model, pd.DataFrame(results)


def main():
    set_seeds()

    # Step 1: load the windows and their RF features
    windows, rf_features, _ = load_data()
    labels = encode_labels(windows["true_label"])
    is_train = (windows["split"] == "train").to_numpy()
    is_val = (windows["split"] == "val").to_numpy()

    # Step 2: tune on train -> validation (the test set is not touched here)
    print("\nTuning Random Forest on the validation set...")
    model, results = tune_forest(
        rf_features[is_train], labels[is_train],
        rf_features[is_val], labels[is_val],
    )
    best = results.loc[results["val_macro_f1"].idxmax()]

    # Step 3: predict every window (train, val and test) with the best model
    print("\nWriting predictions for all splits...")
    probabilities = model.predict_proba(rf_features)
    predictions = make_predictions_table(windows, probabilities)

    # Step 4: save outputs for evaluation
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "rf_window_predictions.csv", index=False)
    results.to_csv(OUTPUT_DIR / "rf_tuning_results.csv", index=False)
    joblib.dump(model, OUTPUT_DIR / "rf_model.joblib", compress=3)
    save_label_mapping()

    print(f"\nBest settings: { {name: model.get_params()[name] for name in PARAM_GRID} }")
    print(f"Random Forest validation macro-F1 (window level): {best['val_macro_f1']:.4f}")
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
