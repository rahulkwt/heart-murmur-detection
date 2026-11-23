"""Shared patient-level validation helpers for experimental model selection."""

from sklearn.metrics import f1_score
from tensorflow import keras

from src.config import LABELS
from src.evaluation.aggregate_patients import aggregate_patients
from src.models.data_utils import make_predictions_table


def patient_macro_f1(windows, probabilities, split="val"):
    """Score probabilities after recording-mean patient aggregation."""
    predictions = make_predictions_table(windows, probabilities)
    patients = aggregate_patients(predictions[predictions["split"].eq(split)],
                                   "recording_mean")
    score = f1_score(patients["true_label"], patients["pred_label"],
                     labels=LABELS, average="macro", zero_division=0)
    return float(score), patients


class PatientMacroF1Callback(keras.callbacks.Callback):
    """Keras callback that records patient-level validation macro-F1."""

    def __init__(self, validation_windows, validation_inputs, batch_size=256):
        """Store the validation windows and model inputs."""
        super().__init__()
        self.validation_windows = validation_windows.reset_index(drop=True)
        self.validation_inputs = validation_inputs
        self.batch_size = batch_size
        self.scores = []

    def on_epoch_end(self, epoch, logs=None):
        """Calculate and record patient-level validation macro-F1."""
        if logs is None:
            logs = {}
        probabilities = self.model.predict(
            self.validation_inputs, batch_size=self.batch_size, verbose=0
        )
        score, _ = patient_macro_f1(self.validation_windows, probabilities)
        logs["val_patient_macro_f1"] = score
        self.scores.append(score)

