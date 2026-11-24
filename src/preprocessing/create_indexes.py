"""Create patient and recording indexes from the raw dataset."""


from pathlib import Path

import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "training_data"
OUTPUT_DIR = ROOT / "data" / "processed"


def normalize_location(value):
    """Normalize the dataset's four valve locations and phonocardiogram lead."""
    value = value.strip()
    if value.upper() == "PHC":
        return "Phc"
    return value.upper()


def parse_murmur_locations(value):
    """Return the normalized locations from the patient record."""
    value = value.strip()
    if not value or value.lower() in {"nan", "none", "unknown"}:
        return []

    locations = [normalize_location(part) for part in value.replace(",", "+").split("+")]
    return locations


def read_patient_annotation(patient_file):
    """Read the patient and location annotations from one record."""
    lines = patient_file.read_text().splitlines()
    annotations = {}
    for line in lines:
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            annotations[key.strip()] = value.strip()

    return lines, annotations["Murmur"], parse_murmur_locations(
        annotations.get("Murmur locations", "")
    )


def create_indexes():
    """Create patient and recording indexes from the raw annotation files."""
    patient_files = sorted(DATA_DIR.rglob("*.txt"))

    recording_rows = []
    patient_rows = []

    for patient_file in patient_files:
        patient_id = patient_file.stem
        raw_lines = patient_file.read_text().splitlines()
        if not raw_lines or not any(line.startswith("#Murmur:") for line in raw_lines):
            continue
        lines, label, murmur_locations = read_patient_annotation(patient_file)

        patient_rows.append({
            "patient_id": patient_id,
            "patient_label": label,
            "murmur_label": label,
            "murmur_locations": "+".join(murmur_locations or []),
        })

        for line in lines[1:]:
            if not line.strip() or line.startswith("#"):
                continue

            parts = line.split()
            audio_name = next(
                (part for part in parts if part.lower().endswith(".wav")),
                None
            )

            if audio_name is None:
                continue

            recording_location = normalize_location(parts[0])
            if label == "Present" and recording_location not in murmur_locations:
                recording_label = "Absent"
            else:
                recording_label = label

            audio_path = patient_file.parent / audio_name
            audio_info = sf.info(audio_path)

            recording_rows.append({
                "patient_id": patient_id,
                "recording": audio_path.name,
                "location": recording_location,
                "patient_label": label,
                "recording_label": recording_label,
                "training_label": recording_label,
                "murmur_label": label,
                "murmur_locations": "+".join(murmur_locations or []),
                "sample_rate": audio_info.samplerate,
                "duration_seconds": audio_info.duration,
                "channels": audio_info.channels,
                "audio_path": audio_path.relative_to(ROOT).as_posix()
            })

    recordings = pd.DataFrame(recording_rows)
    patients = pd.DataFrame(patient_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recordings.to_csv(
        OUTPUT_DIR / "recording_index.csv", index=False
    )
    patients.to_csv(
        OUTPUT_DIR / "patient_index.csv", index=False
    )

    print(f"Patients: {len(patients)}")
    print(f"Recordings: {len(recordings)}")
    print("\nPatients per murmur label:")
    print(patients["murmur_label"].value_counts())
    print(f"\nIndexes saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    create_indexes()
