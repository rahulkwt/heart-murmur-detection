
# Audio preprocessing functions for heart murmur detection

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


def preprocess_audio(file_path):
    """
    Load and preprocess a heart sound recording.
    Returns the processed audio and sample rate.
    """

    # Load audio
    audio, sample_rate = sf.read(file_path)

    # Convert stereo to mono if necessary
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Apply bandpass filter (25-800 Hz)
    sos = butter(
        4,
        [25, 800],
        btype="bandpass",
        fs=sample_rate,
        output="sos"
    )
    audio = sosfiltfilt(sos, audio)

    # Normalize using peak amplitude
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio, sample_rate
