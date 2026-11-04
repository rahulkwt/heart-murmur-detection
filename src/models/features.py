"""
Feature extraction for the models.

Two kinds of features are computed for every audio window:
  1. A fixed-length vector of MFCC summary statistics (for the Random Forest).
  2. A log-Mel spectrogram image of shape (N_MELS, time) (for the CNN).

All feature code lives in this one file. To use a different MFCC or log-Mel
implementation, change the import below; nothing else needs to change.
"""

import numpy as np
import librosa

from src.config import SAMPLE_RATE

# MFCC mean + std (26 values) from the shared project code.
from src.features.extract_features import extract_mfcc_features


# ---------------------------------------------------------------------------
# CONFIG: feature settings (change here, then delete the cache to rebuild)
# ---------------------------------------------------------------------------
# Random Forest features
N_MFCC = 13             # number of MFCC coefficients
RF_N_FFT = 1024         # same frame settings as extract_mfcc_features()
RF_HOP_LENGTH = 256

# CNN log-Mel spectrogram
N_MELS = 64             # number of Mel frequency bands (image height)
N_FFT = 512             # FFT size (128 ms at 4 kHz)
HOP_LENGTH = 128        # step between frames (32 ms at 4 kHz)
FMIN = 25               # audio is band-passed to 25-800 Hz, so higher
FMAX = 800              # Mel bands would carry no information
TOP_DB = 80             # clip the spectrogram 80 dB below its loudest point


# Names of the Random Forest features, in the same order as the vector
# returned by extract_rf_features(). Saved next to the cached features.
RF_FEATURE_NAMES = (
    [f"mfcc_mean_{i}" for i in range(1, N_MFCC + 1)]
    + [f"mfcc_std_{i}" for i in range(1, N_MFCC + 1)]
    + [f"delta_mfcc_mean_{i}" for i in range(1, N_MFCC + 1)]
    + [f"delta_mfcc_std_{i}" for i in range(1, N_MFCC + 1)]
    + ["rms_mean", "rms_std", "zcr_mean", "zcr_std"]
)


# Build the Random Forest feature vector for one window.
# Only the real (non-padded) part of the window should be passed in, so the
# zero padding at the end of a recording does not distort the statistics.
def extract_rf_features(audio, sample_rate=SAMPLE_RATE):
    audio = np.asarray(audio, dtype=np.float32)

    # 26 values: mean and std of 13 MFCCs over time (shared project code)
    mfcc_stats = extract_mfcc_features(audio, sample_rate, n_mfcc=N_MFCC)

    # Delta MFCCs describe how the spectrum changes from frame to frame,
    # which helps capture the timing of heart sounds and murmurs
    mfccs = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=N_MFCC,
        n_fft=RF_N_FFT, hop_length=RF_HOP_LENGTH
    )
    deltas = librosa.feature.delta(mfccs)
    delta_stats = np.concatenate([deltas.mean(axis=1), deltas.std(axis=1)])

    # Loudness (RMS energy) and noisiness (zero-crossing rate) over time
    rms = librosa.feature.rms(
        y=audio, frame_length=RF_N_FFT, hop_length=RF_HOP_LENGTH
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=RF_N_FFT, hop_length=RF_HOP_LENGTH
    )[0]
    energy_stats = np.array([rms.mean(), rms.std(), zcr.mean(), zcr.std()])

    return np.concatenate([mfcc_stats, delta_stats, energy_stats]).astype(np.float32)


# Build the log-Mel spectrogram (in decibels) for one full, fixed-length window.
# A 4 s window (16,000 samples) gives shape (64, 126).
def extract_log_mel(audio, sample_rate=SAMPLE_RATE):
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


# Settings that change the feature values. Stored with the cache so that
# old cached features are rebuilt automatically when a setting changes.
def feature_settings():
    return {
        "sample_rate": SAMPLE_RATE, "n_mfcc": N_MFCC,
        "rf_n_fft": RF_N_FFT, "rf_hop_length": RF_HOP_LENGTH,
        "n_mels": N_MELS, "n_fft": N_FFT, "hop_length": HOP_LENGTH,
        "fmin": FMIN, "fmax": FMAX, "top_db": TOP_DB,
        "rf_feature_names": RF_FEATURE_NAMES,
    }
