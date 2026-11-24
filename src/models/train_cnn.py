"""Train a small CNN on log-Mel spectrograms and save window predictions."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.config import LABELS
from src.models.data_utils import (
    SEED, set_seeds, load_data, encode_labels,
    make_predictions_table, save_label_mapping,
)
from src.models.patient_validation import PatientMacroF1Callback, patient_macro_f1
from src.paths import METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR


MAX_EPOCHS = 40
PATIENCE = 6

BASE = {"learning_rate": 1e-3, "batch_size": 32, "filters": (16, 32, 64), "dropout": 0.3}
SEARCH = [
    BASE,
    {**BASE, "learning_rate": 3e-4},
    {**BASE, "batch_size": 64},
    {**BASE, "filters": (32, 64, 128)},
    {**BASE, "dropout": 0.5},
]


def set_all_seeds():
    """Set the random seeds used by Python, NumPy, and TensorFlow."""
    set_seeds(SEED)
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()


def build_cnn(input_shape, x_train, filters, dropout, learning_rate):
    """Build and compile the convolutional classifier."""
    normalize = keras.layers.Normalization(axis=None)
    normalize.adapt(x_train)

    model = keras.Sequential([keras.Input(shape=input_shape), normalize])
    for n_filters in filters:
        model.add(keras.layers.Conv2D(n_filters, 3, padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D(2))

    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(len(LABELS), activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn(params, x_train, y_train, x_val, y_val, class_weight,
              val_windows=None, patient_selection=False):
    """Train one CNN and restore its best validation weights."""
    model = build_cnn(x_train.shape[1:], x_train, params["filters"],
                      params["dropout"], params["learning_rate"])

    callbacks = []
    if patient_selection:
        patient_metric = PatientMacroF1Callback(val_windows, x_val)
        callbacks.append(patient_metric)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_patient_macro_f1", mode="max", patience=PATIENCE,
            restore_best_weights=True,
        )
    else:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        )
    callbacks.append(early_stopping)
    history = model.fit(
        x_train, keras.utils.to_categorical(y_train, len(LABELS)),
        validation_data=(x_val, keras.utils.to_categorical(y_val, len(LABELS))),
        epochs=MAX_EPOCHS,
        batch_size=params["batch_size"],
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    history = pd.DataFrame(history.history)
    history.insert(0, "epoch", range(1, len(history) + 1))
    return model, history


def main(label_mode="patient", selection_mode="window", output_dir=None,
         save_diagnostics=True, output_prefix="cnn"):
    """Tune the CNN, save the best model, and write its predictions."""
    set_all_seeds()
    print("GPUs found:", tf.config.list_physical_devices("GPU") or "none (using CPU)")

    windows, _, log_mels = load_data(label_mode=label_mode)
    x_all = log_mels[..., np.newaxis]
    labels = encode_labels(windows["training_label"])
    is_train = (windows["split"] == "train").to_numpy()
    is_val = (windows["split"] == "val").to_numpy()
    x_train, y_train = x_all[is_train], labels[is_train]
    x_val, y_val = x_all[is_val], labels[is_val]

    weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=y_train)
    class_weight = dict(enumerate(weights))
    print("Class weights:", {LABELS[i]: round(float(w), 2) for i, w in class_weight.items()})

    results, all_histories = [], []
    best_model, best_history, best_f1 = None, None, -1.0

    for trial, params in enumerate(SEARCH, start=1):
        print(f"\n=== Trial {trial}/{len(SEARCH)}: {params} ===")
        keras.utils.set_random_seed(SEED)
        start = time.time()
        model, history = train_cnn(
            params, x_train, y_train, x_val, y_val, class_weight,
            val_windows=windows.loc[is_val].reset_index(drop=True),
            patient_selection=selection_mode == "patient",
        )

        val_probabilities = model.predict(x_val, batch_size=256, verbose=0)
        if selection_mode == "patient":
            val_f1, _ = patient_macro_f1(
                windows.loc[is_val].reset_index(drop=True), val_probabilities
            )
            best_epoch = int(history.loc[history["val_patient_macro_f1"].idxmax(), "epoch"])
        else:
            val_pred = val_probabilities.argmax(axis=1)
            val_f1 = f1_score(y_val, val_pred, average="macro", labels=range(len(LABELS)))
            best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
        print(f"Trial {trial}: val macro-F1 {val_f1:.4f} (best epoch {best_epoch}, "
              f"{time.time() - start:.0f} s)")

        results.append({"trial": trial, **params, "filters": str(params["filters"]),
                        "epochs_run": len(history), "best_epoch": best_epoch,
                        "val_macro_f1": val_f1,
                        "selection": selection_mode})
        all_histories.append(history.assign(trial=trial))

        if val_f1 > best_f1:
            best_model, best_history, best_f1 = model, history, val_f1

    results = pd.DataFrame(results)
    best = results.loc[results["val_macro_f1"].idxmax()]

    print("\nWriting predictions for all splits...")
    probabilities = best_model.predict(x_all, batch_size=256, verbose=0)
    predictions = make_predictions_table(windows, probabilities)

    result_root = Path(output_dir) if output_dir else MODELS_DIR.parent
    model_dir = result_root / "models"
    prediction_dir = result_root / "predictions"
    metrics_dir = result_root / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{output_prefix}_windows.csv"
    model_path = model_dir / f"{output_prefix}.keras"
    mapping_path = model_dir / "label_mapping.json"
    metadata_path = model_dir / f"{output_prefix}_metadata.json"
    history_path = metrics_dir / f"{output_prefix}_training_history.csv"
    tuning_history_path = metrics_dir / f"{output_prefix}_tuning_histories.csv"
    tuning_results_path = metrics_dir / f"{output_prefix}_tuning_results.csv"
    predictions.to_csv(prediction_path, index=False)
    if save_diagnostics:
        best_history.to_csv(history_path, index=False)
        pd.concat(all_histories).to_csv(tuning_history_path, index=False)
        results.to_csv(tuning_results_path, index=False)
    best_model.save(model_path)
    save_label_mapping(mapping_path.parent)
    metadata = {
        "label_mode": label_mode,
        "output_prefix": output_prefix,
        "selection": selection_mode,
        "selection_metric": "patient_macro_f1" if selection_mode == "patient" else "window_macro_f1",
        "best_epoch": int(best["best_epoch"]),
        "best_validation_macro_f1": float(best_f1),
        "class_order": LABELS,
        "aggregation": "recording_mean" if selection_mode == "patient" else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("\n" + results.to_string(index=False))
    print(f"\nBest trial: {int(best['trial'])}")
    print(f"CNN validation macro-F1 ({selection_mode} level): {best_f1:.4f}")
    print(f"Saved outputs to {result_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-mode", choices=["patient", "recording"], default="patient")
    parser.add_argument("--selection-mode", choices=["window", "patient"], default="window")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    main(args.label_mode, args.selection_mode, args.output_dir)
