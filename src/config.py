"""Settings shared by preprocessing, models and evaluation."""

LABELS = ["Present", "Absent", "Unknown"]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
SAMPLE_RATE = 4000
WINDOW_SECONDS = 4
MIN_VALID_SECONDS = 1.0
