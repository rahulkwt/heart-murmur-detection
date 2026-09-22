# Heart Murmur Detection from Phonocardiogram Recordings

Classifying heart murmurs from phonocardiogram (PCG) recordings using a CNN vs. a Random Forest baseline, trained on the CirCor DigiScope dataset.

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
pip install -r requirements.txt
```

## Dataset

This project uses the [CirCor DigiScope Phonocardiogram Dataset v1.0.3](https://physionet.org/content/circor-heart-sound/1.0.3/). The dataset is **not included** in this repository due to size — download it from PhysioNet and place the raw files in `data/raw/`.
