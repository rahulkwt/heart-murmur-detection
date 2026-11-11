"""Create a stratified 70/15/15 patient split, keeping recordings together."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[2] / "data/processed"


def split_patients():
    patients = pd.read_csv(DATA_DIR / "patient_index.csv", dtype={"patient_id": str})
    recordings = pd.read_csv(DATA_DIR / "recording_index.csv", dtype={"patient_id": str})
    if not patients["patient_id"].is_unique:
        raise ValueError("Each patient must appear exactly once.")
    if not patients["murmur_label"].isin(["Present", "Absent", "Unknown"]).all():
        raise ValueError("Missing or invalid murmur labels.")

    train, remaining = train_test_split(
        patients, test_size=0.30, stratify=patients["murmur_label"], random_state=42,
    )
    validation, test = train_test_split(
        remaining, test_size=0.50, stratify=remaining["murmur_label"], random_state=42,
    )
    patient_splits = pd.concat([
        train.assign(split="train"), validation.assign(split="validation"),
        test.assign(split="test"),
    ], ignore_index=True)
    recording_splits = recordings.merge(
        patient_splits[["patient_id", "split"]], on="patient_id",
        how="left", validate="many_to_one",
    )
    assert patient_splits["patient_id"].is_unique, "A patient appears in multiple splits"
    assert recording_splits["split"].notna().all(), "A recording has no patient split"
    patient_splits.to_csv(DATA_DIR / "patient_splits.csv", index=False)
    recording_splits.to_csv(DATA_DIR / "recording_splits.csv", index=False)
    print("Patients per split and murmur label:")
    print(pd.crosstab(patient_splits["split"], patient_splits["murmur_label"]))
    print("\nRecordings per split:")
    print(recording_splits["split"].value_counts())
    print("\nSaved patient_splits.csv and recording_splits.csv")


if __name__ == "__main__":
    split_patients()
