"""
Preprocess audio files and save the processed windows and their index.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, resample_poly


ROOT = Path(__file__).resolve().parents[2]


def preprocess_audio(file_path):
    audio, sample_rate = sf.read(file_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio - audio.mean()

    if sample_rate != 4000:
        audio = resample_poly(audio, 4000, sample_rate)
        sample_rate = 4000

    sos = butter(4, [25, 800], btype="bandpass",
                 fs=sample_rate, output="sos")
    audio = sosfiltfilt(sos, audio)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio, sample_rate


def create_windows(audio, sample_rate, window_seconds=4):
    window_size = int(sample_rate * window_seconds)
    windows = []

    for start in range(0, len(audio), window_size):
        window = audio[start:start + window_size]
        valid_samples = len(window)

        if valid_samples < window_size:
            window = np.pad(window, (0, window_size - valid_samples))

        windows.append((window, start, valid_samples))

    return windows


def prepare_dataset():
    output_dir = ROOT / "data" / "processed"
    recordings = pd.read_csv(
        output_dir / "recording_splits.csv",
        dtype={"patient_id": str}
    )

    recordings = recordings[
    recordings["murmur_label"].isin(["Present", "Absent"])
    ].copy()
    rows = []

    for recording_number, (_, recording) in enumerate(recordings.iterrows()):
        audio, sample_rate = preprocess_audio(ROOT / recording["audio_path"])
        windows = create_windows(audio, sample_rate)

        folder = output_dir / "windows" / recording["split"]
        folder.mkdir(parents=True, exist_ok=True)

        for window_number, (window, start, valid_samples) in enumerate(windows):
            filename = f"recording_{recording_number}_window_{window_number}.wav"
            window_path = folder / filename
            sf.write(window_path, window, sample_rate, subtype="FLOAT")

            rows.append({
                "patient_id": recording["patient_id"],
                "recording": recording["recording"],
                "location": recording["location"],
                "murmur_label": recording["murmur_label"],
                "split": recording["split"],
                "start_seconds": start / sample_rate,
                "valid_samples": valid_samples,
                "audio_path": window_path.relative_to(ROOT).as_posix()
            })

    window_index = pd.DataFrame(rows)
    window_index.to_csv(output_dir / "window_index.csv", index=False)

    print("Recordings processed:", len(recordings))
    print("Windows saved:", len(window_index))


if __name__ == "__main__":
    prepare_dataset()
