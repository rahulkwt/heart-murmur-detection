# Heart Murmur Detection from Phonocardiogram Recordings

Classifying heart murmurs as Present, Absent or Unknown using Random Forest, a CNN, and frozen YAMNet/CardioPHON encoders on the CirCor DigiScope dataset.

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
python -m pip install -r requirements.txt -r requirements-cardiophon.txt
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

## Training and evaluation

Run these commands from the project root after placing the dataset in `data/raw/training_data/`:

```bash
python -m src.preprocessing.create_indexes
python -m src.preprocessing.split_patients
python -m src.models.data_utils
python -m src.models.train_rf
python -m src.models.train_cnn
python -m src.models.make_figures
python -m src.evaluation.aggregate_patients
python -m src.evaluation.evaluate_models
python -m src.evaluation.plot_confusion_matrices
python -m src.evaluation.plot_roc_curves
python -m src.evaluation.plot_model_comparison
```

Skip training when the saved models and predictions are already available.
Final evaluation gives each recording equal weight within its patient.

## Notebooks

Open Jupyter from the project root and select the environment where the requirements
are installed. Run cells from top to bottom. Notebooks 01–02 explore the data;
03 uses saved RF/CNN outputs; 04 compares noise augmentation; 05 trains the
YAMNet and CardioPHON transfer classifiers; 06 adds Logistic Regression and
compares it with CardioPHON. Run 05 before 06, and rerun 03's ROC section after
06 to include Logistic Regression. The first transfer run downloads pretrained
models and can take a while on CPU.

Results go to [outputs/](outputs/README.md); notebook figures go to
[reports/figures/](reports/figures/). CardioPHON uses a frozen audio encoder
with a new classifier, without demographic fusion or quality filtering;
see the [official release](https://git.lih.lu/vdespotovic/cardiophon).
