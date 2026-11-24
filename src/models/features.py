"""
Feature extraction for the models.

Two kinds of features are computed for every audio window:
  1. A fixed-length vector of MFCC summary statistics (for the Random Forest).
  2. A log-Mel spectrogram image of shape (N_MELS, time) (for the CNN).

MFCCs are computed once per window and reused for the delta statistics.
"""

import numpy as np
import librosa

from src.config import SAMPLE_RATE

N_MFCC = 13
RF_N_FFT = 1024
RF_HOP_LENGTH = 256

N_MELS = 64
N_FFT = 512
HOP_LENGTH = 128
FMIN = 25
FMAX = 800
TOP_DB = 80


RF_FEATURE_NAMES = (
    [f"mfcc_mean_{i}" for i in range(1, N_MFCC + 1)]
    + [f"mfcc_std_{i}" for i in range(1, N_MFCC + 1)]
    + [f"delta_mfcc_mean_{i}" for i in range(1, N_MFCC + 1)]
    + [f"delta_mfcc_std_{i}" for i in range(1, N_MFCC + 1)]
    + ["rms_mean", "rms_std", "zcr_mean", "zcr_std"]
)


def extract_rf_features(audio, sample_rate=SAMPLE_RATE):
    """Return MFCC, delta-MFCC, energy, and zero-crossing features."""
    audio = np.asarray(audio, dtype=np.float32)

    mfccs = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=N_MFCC,
        n_fft=RF_N_FFT, hop_length=RF_HOP_LENGTH
    )
    mfcc_stats = np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)])
    deltas = librosa.feature.delta(mfccs)
    delta_stats = np.concatenate([deltas.mean(axis=1), deltas.std(axis=1)])

    rms = librosa.feature.rms(
        y=audio, frame_length=RF_N_FFT, hop_length=RF_HOP_LENGTH
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=RF_N_FFT, hop_length=RF_HOP_LENGTH
    )[0]
    energy_stats = np.array([rms.mean(), rms.std(), zcr.mean(), zcr.std()])

    return np.concatenate([mfcc_stats, delta_stats, energy_stats]).astype(np.float32)


def extract_log_mel(audio, sample_rate=SAMPLE_RATE):
    """Return a log-Mel spectrogram for one audio window."""
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
    return log_mel.astype(np.float32)


def feature_settings():
    """Return the settings that identify the extracted features."""
    return {
        "sample_rate": SAMPLE_RATE, "n_mfcc": N_MFCC,
        "rf_n_fft": RF_N_FFT, "rf_hop_length": RF_HOP_LENGTH,
        "n_mels": N_MELS, "n_fft": N_FFT, "hop_length": HOP_LENGTH,
        "fmin": FMIN, "fmax": FMAX, "top_db": TOP_DB,
        "rf_feature_names": RF_FEATURE_NAMES,
    }
