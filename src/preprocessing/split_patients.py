# Reads the indexes, splits the patients into train, validation, and test sets, and saves the splits to `data/processed` folder.
# Important to note that the splits are done at the patient level, so all recordings for a given patient will be in the same split.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"


def split_patients():
    patients = pd.read_csv(
        DATA_DIR / "patient_index.csv",
        dtype={"patient_id": str}
    )

    recordings = pd.read_csv(
        DATA_DIR / "recording_index.csv",
        dtype={"patient_id": str}
    )

    if not patients["patient_id"].is_unique:
        raise ValueError("Each patient must appear exactly once.")

    if not patients["murmur_label"].isin(
        ["Present", "Absent", "Unknown"]
    ).all():
        raise ValueError("Missing or invalid murmur labels.")

    train, remaining = train_test_split(
        patients,
        test_size=0.30,
        stratify=patients["murmur_label"],
        random_state=42
    )

    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        stratify=remaining["murmur_label"],
        random_state=42
    )

    train = train.assign(split="train")
    validation = validation.assign(split="validation")
    test = test.assign(split="test")

    patient_splits = pd.concat(
        [train, validation, test],
        ignore_index=True
    )

    recording_splits = recordings.merge(
        patient_splits[["patient_id", "split"]],
        on="patient_id",
        how="left",
        validate="many_to_one"
    )

    # Check that no patient appears in multiple groups.
    train_ids = set(train["patient_id"])
    validation_ids = set(validation["patient_id"])
    test_ids = set(test["patient_id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert recording_splits["split"].notna().all()

    patient_splits.to_csv(
        DATA_DIR / "patient_splits.csv", index=False
    )

    recording_splits.to_csv(
        DATA_DIR / "recording_splits.csv", index=False
    )

    print("Patients per split and murmur label:")
    print(pd.crosstab(
        patient_splits["split"],
        patient_splits["murmur_label"]
    ))

    print("\nRecordings per split:")
    print(recording_splits["split"].value_counts())

    print("\nSaved patient_splits.csv and recording_splits.csv")


if __name__ == "__main__":
    split_patients()
