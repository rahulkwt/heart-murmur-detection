"""Frozen CardioPHON embeddings for the project's four-second audio windows.

Architecture and 4 kHz frontend follow the official release:
https://git.lih.lu/vdespotovic/cardiophon (revision a3bf9709).
This is an audio-only transfer experiment, without quality filtering or
demographic fusion; it does not reproduce the full published system.
"""

import importlib.util
from pathlib import Path
from urllib.request import urlretrieve


REVISION = "a3bf9709"
BASE_URL = (
    "https://git.lih.lu/vdespotovic/cardiophon/-/raw/"
    f"{REVISION}/heart_sound_classification"
)
ASSETS = {
    "models.py": "byol_a2/models.py",
    "pretrained_physionet_2022_4kHz.pth":
        "checkpoints/pretrained_physionet_2022_4kHz.pth",
}


def load_encoder(asset_dir):
    """Download the pinned official encoder and load its weights strictly."""
    import torch

    # Small four-second inputs are faster without intra-op thread overhead.
    torch.set_num_threads(1)
    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    for filename, relative in ASSETS.items():
        target = asset_dir / filename
        if not target.is_file():
            partial = target.with_suffix(target.suffix + ".download")
            urlretrieve(f"{BASE_URL}/{relative}", partial)
            partial.replace(target)
    spec = importlib.util.spec_from_file_location(
        "_official_cardiophon_models", asset_dir / "models.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = module.AudioNTT2022(n_mels=64, d=3072)
    state = torch.load(
        asset_dir / "pretrained_physionet_2022_4kHz.pth",
        map_location="cpu", weights_only=True,
    )
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def extract_embeddings(recordings, asset_dir, reference):
    """Return matched window metadata and frozen 3072-dimensional embeddings.

    Log-mel normalization is fitted on training windows only, then reused for
    validation and test. The frontend uses the official config_4k.yaml values.
    """
    import numpy as np
    import pandas as pd
    import torch
    from nnAudio.features import MelSpectrogram

    from src.evaluation.aggregate_patients import KEYS
    from src.models.data_utils import SPLIT_NAMES
    from src.preprocessing.audio_preprocessing import iter_recording_windows

    encoder = load_encoder(asset_dir)
    frontend = MelSpectrogram(
        sr=4000, n_fft=512, win_length=512, hop_length=160,
        n_mels=64, fmin=60, fmax=2000, center=True, power=2,
        verbose=False,
    ).eval()

    def log_mel(window, rate):
        if rate != 4000:
            raise ValueError(f"CardioPHON expects the project's 4 kHz windows, got {rate}")
        audio = torch.as_tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            return (frontend(audio) + torch.finfo(torch.float32).eps).log()

    total, squares, count = 0.0, 0.0, 0
    training = recordings.loc[recordings["split"].eq("train")]
    for window, rate, _ in iter_recording_windows(training):
        mel = log_mel(window, rate).double()
        total += mel.sum().item()
        squares += mel.square().sum().item()
        count += mel.numel()
    if count == 0:
        raise ValueError("No training windows for CardioPHON normalization")
    mean = total / count
    std = np.sqrt(max(squares / count - mean ** 2, 0))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid CardioPHON training normalization statistics")

    rows, embeddings = [], []
    for i, (window, rate, meta) in enumerate(iter_recording_windows(recordings), 1):
        mel = (log_mel(window, rate) - mean) / std
        with torch.inference_mode():
            embedding = encoder(mel.unsqueeze(0)).cpu().numpy()[0]
        embeddings.append(embedding)
        rows.append({
            "subject_id": str(meta["patient_id"]),
            "recording_id": Path(meta["recording"]).stem,
            "location": meta["location"], "window_idx": meta["window_idx"],
            "split": SPLIT_NAMES[meta["split"]],
            "true_label": meta["murmur_label"],
        })
        if i % 1000 == 0:
            print(f"CardioPHON: extracted {i:,} windows")
    windows = pd.DataFrame(rows)
    identity = KEYS + ["split", "true_label"]
    pd.testing.assert_frame_equal(
        reference[identity].sort_values(KEYS).reset_index(drop=True),
        windows[identity].sort_values(KEYS).reset_index(drop=True),
    )
    features = np.stack(embeddings)
    if features.shape != (len(windows), 3072) or not np.isfinite(features).all():
        raise ValueError("Invalid CardioPHON embedding shape or values")
    return windows, features, {"mean": mean, "std": float(std), "revision": REVISION}
