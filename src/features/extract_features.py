
"""Feature extraction for heart sound recordings."""

import numpy as np
import librosa


def extract_mfcc_features(audio, sample_rate, n_mfcc=13):
    """Return the mean and standard deviation of each MFCC over time."""
    mfccs = librosa.feature.mfcc(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=1024,
        hop_length=256
    )

    return np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)])
