# Uses the data in `data/raw/training_data` folder and saves the recording and patient indexes to `data/processed` folder.
# 

from pathlib import Path

import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "training_data"
OUTPUT_DIR = ROOT / "data" / "processed"


def create_indexes():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    patient_files = sorted(DATA_DIR.rglob("*.txt"))

    recording_rows = []
    patient_rows = []
    issues = []

    for patient_file in patient_files:
        patient_id = patient_file.stem
        lines = patient_file.read_text().splitlines()

        # Skip text files that are not patient records.
        if not lines or not any(
            line.startswith("#Murmur:") for line in lines
        ):
            continue

        label = next(
            line.split(":", 1)[1].strip()
            for line in lines
            if line.startswith("#Murmur:")
        )

        if label not in {"Present", "Absent", "Unknown"}:
            issues.append(f"{patient_file.name}: invalid label '{label}'")
            continue

        patient_rows.append({
            "patient_id": patient_id,
            "murmur_label": label
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

            audio_path = patient_file.parent / audio_name

            if not audio_path.exists():
                issues.append(f"{audio_name}: missing audio file")
                continue

            try:
                audio_info = sf.info(audio_path)
            except Exception as error:
                issues.append(f"{audio_name}: {error}")
                continue

            recording_rows.append({
                "patient_id": patient_id,
                "recording": audio_path.name,
                "location": parts[0],
                "murmur_label": label,
                "sample_rate": audio_info.samplerate,
                "duration_seconds": audio_info.duration,
                "channels": audio_info.channels,
                "audio_path": audio_path.relative_to(ROOT).as_posix()
            })

    if issues:
        for issue in issues:
            print(issue)
        raise ValueError("Resolve the issues above before creating indexes.")

    if not recording_rows:
        raise ValueError("No recordings found. Check the dataset folder.")

    recordings = pd.DataFrame(recording_rows)
    patients = pd.DataFrame(patient_rows)

    if not patients["patient_id"].is_unique:
        raise ValueError("Duplicate patient IDs found.")

    if not recordings["audio_path"].is_unique:
        raise ValueError("Duplicate recording paths found.")

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