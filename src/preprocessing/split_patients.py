# Add patient-level splits to the existing indexes for binary classification.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"


def split_patients():
    patients = pd.read_csv(DATA_DIR / "patient_index.csv", dtype={"patient_id": str})
    recordings = pd.read_csv(DATA_DIR / "recording_index.csv", dtype={"patient_id": str})

    if not patients["patient_id"].is_unique:
        raise ValueError("Each patient must appear exactly once.")
    if not patients["murmur_label"].isin(["Present", "Absent", "Unknown"]).all():
        raise ValueError("Missing or invalid patient labels.")
    if not recordings["patient_id"].isin(patients["patient_id"]).all():
        raise ValueError("Recording has no matching patient.")
    labels = patients.set_index("patient_id")["murmur_label"]
    if not recordings["murmur_label"].eq(recordings["patient_id"].map(labels)).all():
        raise ValueError("Recording and patient labels disagree.")

    known = patients[patients["murmur_label"].isin(["Present", "Absent"])]
    train, remaining = train_test_split(
        known, test_size=0.30, stratify=known["murmur_label"], random_state=42
    )
    validation, test = train_test_split(
        remaining, test_size=0.50,
        stratify=remaining["murmur_label"], random_state=42
    )

    patients["split"] = "excluded"
    for name, group in [("train", train), ("validation", validation), ("test", test)]:
        patients.loc[group.index, "split"] = name

    split_lookup = patients.set_index("patient_id")["split"]
    recordings["split"] = recordings["patient_id"].map(split_lookup)

    groups = [set(group["patient_id"]) for group in [train, validation, test]]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("A patient appears in multiple splits.")
    if recordings["split"].isna().any():
        raise ValueError("A recording has no split assignment.")

    patients.to_csv(DATA_DIR / "patient_index.csv", index=False)
    recordings.to_csv(DATA_DIR / "recording_index.csv", index=False)

    print("Patients per split and label:")
    print(pd.crosstab(patients["split"], patients["murmur_label"]).to_string())
    print("\nRecordings per split:")
    print(recordings["split"].value_counts().to_string())
    print("\nUpdated patient_index.csv and recording_index.csv")


if __name__ == "__main__":
    split_patients()
