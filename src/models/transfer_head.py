"""Small reusable classifier head for frozen transfer-learning features."""

import numpy as np

from src.config import LABELS
from src.models.data_utils import encode_labels
from src.models.patient_validation import PatientMacroF1Callback


def train_transfer_head(windows, features, patient_selection=True):
    """Train a classifier head and select its restored epoch by patient F1."""
    from tensorflow import keras
    from sklearn.utils.class_weight import compute_class_weight

    train = windows["split"].eq("train").to_numpy()
    val = windows["split"].eq("val").to_numpy()
    y = encode_labels(windows["training_label"])
    classes = np.arange(len(LABELS))
    weights = compute_class_weight("balanced", classes=classes, y=y[train])

    normalize = keras.layers.Normalization()
    normalize.adapt(features[train])
    classifier = keras.Sequential([
        keras.Input(shape=(features.shape[1],)),
        normalize,
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(LABELS), activation="softmax"),
    ])
    classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    callbacks = []
    if patient_selection:
        callbacks.append(PatientMacroF1Callback(windows.loc[val].reset_index(drop=True), features[val]))
        monitor = "val_patient_macro_f1"
        mode = "max"
    else:
        monitor = "val_loss"
        mode = "min"
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor=monitor, mode=mode, patience=6, restore_best_weights=True
    ))
    history = classifier.fit(
        features[train], y[train],
        validation_data=(features[val], y[val]),
        class_weight=dict(enumerate(weights)),
        epochs=40, batch_size=32, verbose=2, callbacks=callbacks,
    )
    return classifier, history
