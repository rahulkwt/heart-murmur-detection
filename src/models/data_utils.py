"""
Shared data loading for train_rf.py and train_cnn.py.

Builds one table of audio windows (with subject, recording, location, label
and split for every window) plus the MFCC and log-Mel features for those
windows. Both models use exactly the same windows, so their results can be
compared fairly.

The features are cached in data/processed/model_cache/ because computing
them for ~19,000 windows takes several minutes on a CPU.

Run once on its own to build the cache:
    python -m src.models.data_utils
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the project's preprocessing (resample, band-pass, normalize, window)
from src.preprocessing.audio_preprocessing import iter_recording_windows
from src.config import LABELS, LABEL_TO_INDEX, WINDOW_SECONDS, MIN_VALID_SECONDS
from src.models import features


ROOT = Path(__file__).resolve().parents[2]
SPLITS_CSV = ROOT / "data" / "processed" / "recording_splits.csv"
CACHE_DIR = ROOT / "data" / "processed" / "model_cache"
OUTPUT_DIR = ROOT / "outputs"

SEED = 42

# The repo calls the validation split "validation"; outputs use "val"
SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}


# Set the Python and NumPy random seeds so results can be reproduced.
# (train_cnn.py also sets the TensorFlow seed.)
def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


# Cut every recording into windows and compute both kinds of features.
# Returns the window table, the RF feature matrix and the log-Mel array.
def build_features():
    recordings = pd.read_csv(SPLITS_CSV, dtype={"patient_id": str})
    print(f"Building features for {len(recordings)} recordings...")

    rows, rf_features, log_mels = [], [], []
    for window, sample_rate, metadata in iter_recording_windows(recordings):
        valid_samples = metadata["valid_samples"]
        rf_features.append(features.extract_rf_features(window[:valid_samples], sample_rate))
        log_mels.append(features.extract_log_mel(window, sample_rate))
        rows.append({
            "subject_id": metadata["patient_id"],
            "recording_id": Path(metadata["recording"]).stem,
            "location": metadata["location"],
            "window_idx": metadata["window_idx"],
            "start_seconds": metadata["start_seconds"],
            "valid_seconds": valid_samples / sample_rate,
            "split": SPLIT_NAMES[metadata["split"]],
            "true_label": metadata["murmur_label"],
        })

    return pd.DataFrame(rows), np.stack(rf_features), np.stack(log_mels)


# Settings that change which windows exist or what their features are.
def cache_settings():
    return {
        "labels": LABELS,
        "window_seconds": WINDOW_SECONDS,
        "min_valid_seconds": MIN_VALID_SECONDS,
        "features": features.feature_settings(),
    }


# Load the windows and features from the cache, building the cache first if
# it is missing or was made with different settings.
def load_data():
    settings_file = CACHE_DIR / "settings.json"
    windows_file = CACHE_DIR / "windows.csv"
    rf_file = CACHE_DIR / "rf_features.npy"
    mel_file = CACHE_DIR / "log_mel.npy"

    cache_ok = (
        settings_file.exists() and windows_file.exists()
        and rf_file.exists() and mel_file.exists()
        and json.loads(settings_file.read_text()) == cache_settings()
    )

    if cache_ok:
        print(f"Loading cached features from {CACHE_DIR}")
        windows = pd.read_csv(windows_file, dtype={"subject_id": str})
        rf_features = np.load(rf_file)
        log_mels = np.load(mel_file).astype(np.float32)
    else:
        windows, rf_features, log_mels = build_features()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        windows.to_csv(windows_file, index=False)
        np.save(rf_file, rf_features)
        # float16 halves the file size; the precision loss (~0.03 dB) is tiny
        np.save(mel_file, log_mels.astype(np.float16))
        settings_file.write_text(json.dumps(cache_settings(), indent=2))
        print(f"Saved feature cache to {CACHE_DIR}")

    check_data(windows, rf_features, log_mels)
    return windows, rf_features, log_mels


# Sanity checks: no NaNs, matching lengths, consistent shapes, and no subject
# that appears in more than one split.
def check_data(windows, rf_features, log_mels):
    assert len(windows) == len(rf_features) == len(log_mels), "Feature/window count mismatch"
    assert np.isfinite(rf_features).all(), "NaN or inf in RF features"
    assert np.isfinite(log_mels).all(), "NaN or inf in log-Mel spectrograms"
    assert set(windows["true_label"]) <= set(LABELS), "Unexpected label"

    splits_per_subject = windows.groupby("subject_id")["split"].nunique()
    assert (splits_per_subject == 1).all(), "A subject appears in more than one split"

    print(f"Windows: {len(windows)} | RF features: {rf_features.shape[1]} "
          f"| log-Mel shape: {log_mels.shape[1:]}")
    print(pd.crosstab(windows["split"], windows["true_label"])
          .reindex(index=["train", "val", "test"], columns=LABELS))


# Convert label strings to integer class indexes (Present=0, Absent=1, Unknown=2).
def encode_labels(labels):
    return np.array([LABEL_TO_INDEX[label] for label in labels])


# Build the output table for one model: one row per window with the true
# label, predicted label and the probability of each class.
def make_predictions_table(windows, probabilities):
    table = windows[["subject_id", "recording_id", "location", "window_idx",
                     "split", "true_label"]].copy()
    table["pred_label"] = [LABELS[i] for i in probabilities.argmax(axis=1)]
    for i, label in enumerate(LABELS):
        table[f"prob_{label.lower()}"] = probabilities[:, i]
    return table


# Save the class-name-to-index mapping used by both models.
def save_label_mapping():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "label_mapping.json").write_text(json.dumps(LABEL_TO_INDEX, indent=2))


if __name__ == "__main__":
    load_data()
