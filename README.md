# Heart Murmur Detection from Phonocardiogram Recordings

Classifying heart murmurs as Present, Absent or Unknown using a CNN and a Random Forest baseline on the CirCor DigiScope dataset.

## Team

- Mikkel Rosendahl
- Sadhana Bastola
- Rahul Sunil

## Folder Structure

```
data/
  raw/            # Original CirCor .wav, .txt, .tsv files (not committed)
  processed/      # Preprocessed/windowed audio and extracted features (not committed)
notebooks/        # Exploratory Jupyter notebooks
src/
  preprocessing/  # Audio normalization, windowing, subject-level train/val/test split
  features/       # MFCC extraction, log-Mel spectrogram extraction
  models/         # Random Forest baseline, CNN model
  evaluation/     # Accuracy, precision, recall, macro-F1, confusion matrix
reports/          # Final report and figures
```

## Setup

```bash
git clone https://github.com/rahulkwt/heart-murmur-detection.git
cd heart-murmur-detection
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Dataset

This project uses the [CirCor DigiScope Phonocardiogram Dataset v1.0.3](https://physionet.org/content/circor-heart-sound/1.0.3/). The dataset is **not included** in this repository due to size — download it from PhysioNet and place the raw `training_data` folder in `data/raw/`.

## Preprocessing

Both models use the same preprocessing and windows from
`src/preprocessing/audio_preprocessing.py`. Settings and label order are in
`src/config.py`. Audio is resampled to 4 kHz, filtered and normalized, then
split into 4-second windows. Windows with less than 1 second of real audio
are dropped. All three classes are kept.

To save processed audio windows, run `python -m src.preprocessing.audio_preprocessing`.
To build features directly for both models, run `python -m src.models.data_utils`.
Both commands use the same window-building function.

## Evaluation

Run `python -m src.evaluation.aggregate_patients` to compare aggregation on
validation data. Final test evaluation uses equal recording weights; see
`outputs/README.md` for the evaluation and plotting commands.
