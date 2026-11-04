"""
Train a small CNN on log-Mel spectrograms, one prediction per window.

A few hyperparameter settings are compared on the validation set. Early
stopping also uses the validation set. The test set is used only to write
the final predictions.

Runs on CPU, and uses a GPU automatically if TensorFlow can see one (Colab).

Run from the project root:
    python -m src.models.train_cnn
"""

import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.config import LABELS
from src.models.data_utils import (
    OUTPUT_DIR, SEED, set_seeds, load_data, encode_labels,
    make_predictions_table, save_label_mapping,
)


# ---------------------------------------------------------------------------
# CONFIG: training settings and the small hyperparameter search
# ---------------------------------------------------------------------------
MAX_EPOCHS = 40         # upper limit; early stopping usually ends sooner
PATIENCE = 6            # stop after this many epochs without val_loss improving

# Start from a base setting and change one thing at a time. This keeps the
# search small enough for a CPU and makes the effect of each change clear.
BASE = {"learning_rate": 1e-3, "batch_size": 32, "filters": (16, 32, 64), "dropout": 0.3}
SEARCH = [
    BASE,
    {**BASE, "learning_rate": 3e-4},
    {**BASE, "batch_size": 64},
    {**BASE, "filters": (32, 64, 128)},
    {**BASE, "dropout": 0.5},
]


# Seed Python, NumPy and TensorFlow, and ask TensorFlow for deterministic
# operations, so that running the script twice gives the same model.
def set_all_seeds():
    set_seeds(SEED)
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()


# Build the CNN. The first layer standardizes the spectrograms with the mean
# and variance of the training data, so the saved model takes raw log-Mel
# input. Each block is Conv2D -> MaxPooling, which halves the image size.
def build_cnn(input_shape, x_train, filters, dropout, learning_rate):
    normalize = keras.layers.Normalization(axis=None)
    normalize.adapt(x_train)

    model = keras.Sequential([keras.Input(shape=input_shape), normalize])
    for n_filters in filters:
        model.add(keras.layers.Conv2D(n_filters, 3, padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D(2))

    # Average each feature map to one number (few weights, less overfitting)
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


# Train one CNN with early stopping on validation loss. The weights from the
# epoch with the lowest validation loss are restored at the end.
def train_cnn(params, x_train, y_train, x_val, y_val, class_weight):
    model = build_cnn(x_train.shape[1:], x_train, params["filters"],
                      params["dropout"], params["learning_rate"])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )
    history = model.fit(
        x_train, keras.utils.to_categorical(y_train, len(LABELS)),
        validation_data=(x_val, keras.utils.to_categorical(y_val, len(LABELS))),
        epochs=MAX_EPOCHS,
        batch_size=params["batch_size"],
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=2,
    )

    history = pd.DataFrame(history.history)
    history.insert(0, "epoch", range(1, len(history) + 1))
    return model, history


def main():
    set_all_seeds()
    print("GPUs found:", tf.config.list_physical_devices("GPU") or "none (using CPU)")

    # Step 1: load the windows and their log-Mel spectrograms,
    # adding a channel axis -> (n_windows, n_mels, time, 1)
    windows, _, log_mels = load_data()
    x_all = log_mels[..., np.newaxis]
    labels = encode_labels(windows["true_label"])
    is_train = (windows["split"] == "train").to_numpy()
    is_val = (windows["split"] == "val").to_numpy()
    x_train, y_train = x_all[is_train], labels[is_train]
    x_val, y_val = x_all[is_val], labels[is_val]

    # Step 2: class weights so the rare classes count as much as Absent
    weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=y_train)
    class_weight = dict(enumerate(weights))
    print("Class weights:", {LABELS[i]: round(float(w), 2) for i, w in class_weight.items()})

    # Step 3: try each setting, keep the one with the best validation macro-F1
    results, all_histories = [], []
    best_model, best_history, best_f1 = None, None, -1.0

    for trial, params in enumerate(SEARCH, start=1):
        print(f"\n=== Trial {trial}/{len(SEARCH)}: {params} ===")
        keras.utils.set_random_seed(SEED)   # same starting weights for each trial
        start = time.time()
        model, history = train_cnn(params, x_train, y_train, x_val, y_val, class_weight)

        val_pred = model.predict(x_val, batch_size=256, verbose=0).argmax(axis=1)
        val_f1 = f1_score(y_val, val_pred, average="macro", labels=range(len(LABELS)))
        best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
        print(f"Trial {trial}: val macro-F1 {val_f1:.4f} (best epoch {best_epoch}, "
              f"{time.time() - start:.0f} s)")

        results.append({"trial": trial, **params, "filters": str(params["filters"]),
                        "epochs_run": len(history), "best_epoch": best_epoch,
                        "val_macro_f1": val_f1})
        all_histories.append(history.assign(trial=trial))

        if val_f1 > best_f1:
            best_model, best_history, best_f1 = model, history, val_f1

    results = pd.DataFrame(results)
    best = results.loc[results["val_macro_f1"].idxmax()]

    # Step 4: predict every window (train, val and test) with the best model
    print("\nWriting predictions for all splits...")
    probabilities = best_model.predict(x_all, batch_size=256, verbose=0)
    predictions = make_predictions_table(windows, probabilities)

    # Step 5: save outputs for evaluation
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "cnn_window_predictions.csv", index=False)
    best_history[["epoch", "loss", "val_loss", "accuracy", "val_accuracy"]].to_csv(
        OUTPUT_DIR / "cnn_training_history.csv", index=False)
    pd.concat(all_histories).to_csv(OUTPUT_DIR / "cnn_tuning_histories.csv", index=False)
    results.to_csv(OUTPUT_DIR / "cnn_tuning_results.csv", index=False)
    best_model.save(OUTPUT_DIR / "cnn_model.keras")
    save_label_mapping()

    print("\n" + results.to_string(index=False))
    print(f"\nBest trial: {int(best['trial'])}")
    print(f"CNN validation macro-F1 (window level): {best_f1:.4f}")
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
