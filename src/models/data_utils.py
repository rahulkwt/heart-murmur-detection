"""Load shared window labels and audio features for the models."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.audio_preprocessing import iter_recording_windows
from src.config import LABELS, LABEL_TO_INDEX, WINDOW_SECONDS, MIN_VALID_SECONDS
from src.paths import METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, RESULTS_DIR
from src.models import features


ROOT = Path(__file__).resolve().parents[2]
SPLITS_CSV = ROOT / "data" / "processed" / "recording_splits.csv"
CACHE_DIR = ROOT / "data" / "processed" / "model_cache"
OUTPUT_DIR = RESULTS_DIR

SEED = 42

SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}


def set_seeds(seed=SEED):
    """Set the Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def build_features(label_mode="patient"):
    """Build window metadata, Random Forest features, and log-Mel inputs."""
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
            "patient_label": metadata["patient_label"],
            "recording_label": metadata["recording_label"],
            "training_label": metadata["patient_label"] if label_mode == "patient"
            else metadata["recording_label"],
            "true_label": metadata["patient_label"],
        })

    return pd.DataFrame(rows), np.stack(rf_features), np.stack(log_mels)


def cache_settings(label_mode="patient"):
    """Return the settings that identify a feature cache."""
    settings = {
        "labels": LABELS,
        "window_seconds": WINDOW_SECONDS,
        "min_valid_seconds": MIN_VALID_SECONDS,
        "features": features.feature_settings(),
    }
    if label_mode != "patient":
        settings["label_mode"] = label_mode
        settings["source_label_columns"] = ["patient_label", "recording_label"]
    return settings


def load_data(label_mode="patient", cache_dir=None):
    """Load cached features and set the explicit target used for training."""
    cache_path = Path(cache_dir) if cache_dir else (
        CACHE_DIR if label_mode == "patient" else CACHE_DIR.with_name("model_cache_location_aware")
    )
    settings_file = cache_path / "settings.json"
    windows_file = cache_path / "windows.csv"
    rf_file = cache_path / "rf_features.npy"
    mel_file = cache_path / "log_mel.npy"

    cache_ok = (
        settings_file.exists() and windows_file.exists()
        and rf_file.exists() and mel_file.exists()
        and json.loads(settings_file.read_text()) == cache_settings(label_mode)
    )

    if cache_ok:
        print(f"Loading cached features from {cache_path}")
        windows = pd.read_csv(windows_file, dtype={"subject_id": str})
        rf_features = np.load(rf_file)
        log_mels = np.load(mel_file).astype(np.float32)
    else:
        windows, rf_features, log_mels = build_features(label_mode)
        cache_path.mkdir(parents=True, exist_ok=True)
        windows.to_csv(windows_file, index=False)
        np.save(rf_file, rf_features)
        np.save(mel_file, log_mels.astype(np.float16))
        settings_file.write_text(json.dumps(cache_settings(label_mode), indent=2))
        print(f"Saved feature cache to {cache_path}")

    if "patient_label" not in windows:
        windows["patient_label"] = windows["true_label"]
    if "recording_label" not in windows:
        windows["recording_label"] = windows["patient_label"]
    windows["training_label"] = windows[
        "patient_label" if label_mode == "patient" else "recording_label"
    ]

    check_data(windows, rf_features, log_mels)
    return windows, rf_features, log_mels


def check_data(windows, rf_features, log_mels):
    """Print the feature shapes and label counts used by the models."""
    print(f"Windows: {len(windows)} | RF features: {rf_features.shape[1]} "
          f"| log-Mel shape: {log_mels.shape[1:]}")
    print(pd.crosstab(windows["split"], windows["true_label"])
          .reindex(index=["train", "val", "test"], columns=LABELS))


def encode_labels(labels):
    """Convert class names to the integer indexes used by the models."""
    return np.array([LABEL_TO_INDEX[label] for label in labels])


def make_predictions_table(windows, probabilities):
    """Combine window metadata with predicted labels and class probabilities."""
    probabilities = np.asarray(probabilities)
    columns = ["subject_id", "recording_id", "location", "window_idx",
               "split", "true_label"]
    for optional in ["patient_label", "recording_label", "training_label"]:
        if optional in windows:
            columns.append(optional)
    table = windows[columns].copy()
    table["pred_label"] = [LABELS[i] for i in probabilities.argmax(axis=1)]
    for i, label in enumerate(LABELS):
        table[f"prob_{label.lower()}"] = probabilities[:, i]
    return table


def save_label_mapping(output_dir=None):
    """Save the class-to-index mapping beside model outputs."""
    destination = Path(output_dir) if output_dir else OUTPUT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "label_mapping.json").write_text(
        json.dumps(LABEL_TO_INDEX, indent=2)
    )


if __name__ == "__main__":
    load_data()
