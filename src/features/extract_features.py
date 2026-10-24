
# Feature extraction for heart sound recordings

import numpy as np
import librosa


def extract_mfcc_features(audio, sample_rate, n_mfcc=13):
    """
    Extract MFCC features from a preprocessed recording.

    Returns a fixed-length numerical feature vector.
    """

    # Extract MFCC coefficients
    mfccs = librosa.feature.mfcc(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=1024,
        hop_length=256
    )

    # Calculate mean and standard deviation
    # of each MFCC coefficient across time
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    # Combine the features into one vector
    features = np.concatenate([mfcc_mean, mfcc_std])

    return features
