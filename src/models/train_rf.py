"""Train a Random Forest on MFCC summary features."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.config import LABELS
from src.models.data_utils import (
    SEED, set_seeds, load_data, encode_labels,
    make_predictions_table, save_label_mapping,
)
from src.models.patient_validation import patient_macro_f1
from src.paths import METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR


PARAMETERS = [
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5},
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 20},
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 50},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 20},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 50},
    {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 5},
    {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 20},
    {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 50},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 5},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 20},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 50},
]


def tune_forest(x_train, y_train, x_val, y_val, val_windows=None,
                selection_mode="window"):
    """Tune the forest and return the best model and trial scores."""
    results = []
    best_model = None
    best_f1 = -1

    for number, params in enumerate(PARAMETERS, start=1):
        model = RandomForestClassifier(
            **params, class_weight="balanced", random_state=SEED, n_jobs=-1
        )
        model.fit(x_train, y_train)

        if selection_mode == "patient":
            val_f1, _ = patient_macro_f1(val_windows, model.predict_proba(x_val))
        else:
            val_f1 = f1_score(y_val, model.predict(x_val), average="macro",
                              labels=range(len(LABELS)))

        train_f1 = f1_score(y_train, model.predict(x_train), average="macro",
                            labels=range(len(LABELS)))
        results.append({**params, "train_macro_f1": train_f1,
                        "val_macro_f1": val_f1,
                        "selection": selection_mode})
        print(f"  [{number}/{len(PARAMETERS)}] {params} -> "
              f"train macro-F1 {train_f1:.4f}, val macro-F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_model, best_f1 = model, val_f1

    return best_model, pd.DataFrame(results)


def main(label_mode="patient", selection_mode="window", output_dir=None):
    """Tune the forest, save the best model, and write its predictions."""
    set_seeds()

    windows, rf_features, _ = load_data(label_mode=label_mode)
    labels = encode_labels(windows["training_label"])
    is_train = (windows["split"] == "train").to_numpy()
    is_val = (windows["split"] == "val").to_numpy()

    print("\nTuning Random Forest on the validation set...")
    model, results = tune_forest(
        rf_features[is_train], labels[is_train],
        rf_features[is_val], labels[is_val],
        val_windows=windows.loc[is_val].reset_index(drop=True),
        selection_mode=selection_mode,
    )
    print("\nWriting predictions for all splits...")
    probabilities = model.predict_proba(rf_features)
    predictions = make_predictions_table(windows, probabilities)

    if output_dir:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        prediction_path = destination / "rf_window_predictions.csv"
        metrics_path = destination / "rf_tuning_results.csv"
        model_path = destination / "rf_model.joblib"
        mapping_path = destination / "label_mapping.json"
        metadata_path = destination / "rf_metadata.json"
    else:
        destination = PREDICTIONS_DIR.parent
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        prediction_path = PREDICTIONS_DIR / "random_forest_windows.csv"
        metrics_path = METRICS_DIR / "random_forest_tuning.csv"
        model_path = MODELS_DIR / "random_forest.joblib"
        mapping_path = MODELS_DIR / "label_mapping.json"
        metadata_path = MODELS_DIR / "random_forest_metadata.json"
    predictions.to_csv(prediction_path, index=False)
    results.to_csv(metrics_path, index=False)
    joblib.dump(model, model_path, compress=3)
    save_label_mapping(mapping_path.parent)
    metadata = {
        "variant": destination.name,
        "label_mode": label_mode,
        "seed": SEED,
        "class_order": LABELS,
        "aggregation": "recording_mean" if selection_mode == "patient" else None,
        "selection_criterion": "patient_macro_f1" if selection_mode == "patient" else "window_macro_f1",
        "selected_settings": {
            name: model.get_params()[name]
            for name in ["n_estimators", "max_depth", "min_samples_leaf"]
        },
        "best_validation_macro_f1": float(results["val_macro_f1"].max()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("\nBest settings:", metadata["selected_settings"])
    print(f"Random Forest validation macro-F1 ({selection_mode} level): "
          f"{metadata['best_validation_macro_f1']:.4f}")
    print(f"Saved outputs to {prediction_path.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-mode", choices=["patient", "recording"], default="patient")
    parser.add_argument("--selection-mode", choices=["window", "patient"], default="window")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    main(label_mode=args.label_mode, selection_mode=args.selection_mode,
         output_dir=args.output_dir)
