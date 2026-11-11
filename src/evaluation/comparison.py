"""Patient comparisons and ROC plots shared by the analysis notebooks."""

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve

from src.config import LABELS
from src.evaluation.aggregate_patients import KEYS, PROBS, aggregate_patients, metrics


def window_index(table):
    """Sort window identities so predictions from different models can be checked."""
    return table[KEYS + ["split", "true_label"]].sort_values(KEYS).reset_index(drop=True)


def compare_models(predictions, split):
    """Return a score table and patient predictions for one split."""
    rows, by_model = [], {}
    for name, windows in predictions.items():
        patients = aggregate_patients(windows[windows["split"].eq(split)], "recording_mean")
        by_model[name] = patients
        scores = metrics(patients)
        targets = patients["true_label"].to_numpy()[:, None] == np.array(LABELS)
        rows.append({
            "Model": name, "Patients": scores["n_subjects"],
            "Accuracy": scores["accuracy"], "Macro-F1": scores["macro_f1"],
            "Present recall": scores["recall_present"],
            "Macro ROC-AUC": roc_auc_score(targets, patients[PROBS], average="macro"),
        })
    return pd.DataFrame(rows).set_index("Model"), by_model


def plot_patient_roc(ax, patients, title, show_macro=False):
    """Draw the three one-versus-rest ROC curves on an existing subplot."""
    class_aucs = []
    for label, column in zip(LABELS, PROBS):
        fpr, tpr, _ = roc_curve(patients["true_label"].eq(label), patients[column])
        score = auc(fpr, tpr)
        class_aucs.append(score)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {score:.3f})")
    if show_macro:
        title += f"\nMacro ROC-AUC = {np.mean(class_aucs):.3f}"
    ax.plot([0, 1], [0, 1], color="lightgray", linestyle="--", label="Random classifier")
    ax.set(title=title, xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
