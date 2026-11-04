"""
Preprocess audio files and save the processed windows and their index.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, resample_poly

from src.config import LABELS, SAMPLE_RATE, WINDOW_SECONDS, MIN_VALID_SECONDS


ROOT = Path(__file__).resolve().parents[2]


def preprocess_audio(file_path):
    audio, sample_rate = sf.read(file_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio - audio.mean()

    if sample_rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sample_rate)
        sample_rate = SAMPLE_RATE

    sos = butter(4, [25, 800], btype="bandpass",
                 fs=sample_rate, output="sos")
    audio = sosfiltfilt(sos, audio)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio, sample_rate


def create_windows(audio, sample_rate, window_seconds=WINDOW_SECONDS):
    window_size = int(sample_rate * window_seconds)
    windows = []

    for start in range(0, len(audio), window_size):
        window = audio[start:start + window_size]
        valid_samples = len(window)

        if valid_samples < window_size:
            window = np.pad(window, (0, window_size - valid_samples))

        windows.append((window, start, valid_samples))

    return windows


def iter_recording_windows(recordings):
    """Yield processed windows and metadata, keeping all three classes."""
    if not recordings["murmur_label"].isin(LABELS).all():
        raise ValueError("Missing or unsupported murmur label")

    for _, recording in recordings.iterrows():
        audio, sample_rate = preprocess_audio(ROOT / recording["audio_path"])
        for window_idx, (window, start, valid_samples) in enumerate(
            create_windows(audio, sample_rate)
        ):
            if valid_samples < MIN_VALID_SECONDS * sample_rate:
                continue
            metadata = {
                "patient_id": recording["patient_id"],
                "recording": recording["recording"],
                "location": recording["location"],
                "murmur_label": recording["murmur_label"],
                "split": recording["split"],
                "window_idx": window_idx,
                "start_seconds": start / sample_rate,
                "valid_samples": valid_samples,
            }
            yield window, sample_rate, metadata


def prepare_dataset():
    output_dir = ROOT / "data" / "processed"
    recordings = pd.read_csv(
        output_dir / "recording_splits.csv", dtype={"patient_id": str}
    )
    rows = []
    for window, sample_rate, metadata in iter_recording_windows(recordings):
        folder = output_dir / "windows" / metadata["split"]
        folder.mkdir(parents=True, exist_ok=True)
        recording_id = Path(metadata["recording"]).stem
        filename = f"{metadata['patient_id']}_{recording_id}_{metadata['window_idx']}.wav"
        window_path = folder / filename
        sf.write(window_path, window, sample_rate, subtype="FLOAT")
        rows.append({**metadata, "audio_path": window_path.relative_to(ROOT).as_posix()})

    pd.DataFrame(rows).to_csv(output_dir / "window_index.csv", index=False)
    print("Recordings processed:", len(recordings))
    print("Windows saved:", len(rows))


if __name__ == "__main__":
    prepare_dataset()
