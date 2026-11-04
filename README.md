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

## Running the pipeline

After setup and downloading the dataset, run these commands from the project
root in the order shown. Activate the environment where you installed the
requirements first.

```bash
# 1. Create the patient/recording indexes and split patients into train/validation/test.
python -m src.preprocessing.create_indexes
python -m src.preprocessing.split_patients

# 2. Export processed audio windows as WAV files (optional).
python -m src.preprocessing.audio_preprocessing

# 3. Build the shared feature cache, then train and tune both models.
python -m src.models.data_utils
python -m src.models.train_rf
python -m src.models.train_cnn

# 4. Plot training behavior and check validation predictions.
python -m src.models.make_figures
python -m src.models.subject_check  # Optional validation diagnostic.
python -m src.evaluation.aggregate_patients

# 5. Evaluate test patients, then generate the evaluation figures.
python -m src.evaluation.evaluate_models
python -m src.evaluation.plot_confusion_matrices
python -m src.evaluation.plot_roc_curves
python -m src.evaluation.plot_model_comparison
```

The WAV export in step 2 is optional because feature extraction preprocesses
the original recordings directly. The subject-level check in step 4 is also
optional; the aggregation comparison evaluates patient-level rules on
validation data before final test evaluation. Run the ROC plotting command
before the model comparison command, which reads its ROC-AUC results.

Feature caches and exported audio are saved in `data/processed/`. Trained
models, tuning results and window predictions are saved in `outputs/`.
Training figures are saved in `outputs/figures/`; patient-level metrics and
evaluation figures are saved in `outputs/evaluation/`.

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

## Notebooks and report figures

After running the pipeline, execute the first three notebooks in order to refresh
their saved outputs and generate figures in `reports/figures/`:

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/01_data_exploration.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/02_audio_exploration.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/03_model_exploration.ipynb
```

The notebooks cover dataset exploration, audio exploration and model
exploration, respectively. The model notebook reads the saved models and
predictions without retraining and saves all eight of its figures in
`reports/figures/`.

All six notebooks save their plots under `reports/figures/`, including each audio
preprocessing example and the noise augmentation, transfer learning, and
additional model analysis charts.
Notebook 06 groups its figures in `reports/figures/additional_analysis/`.
It also saves Logistic Regression predictions and refreshes `patient_roc_curves.png`
with all three models. Notebook 03 reuses these predictions on subsequent runs.
It also saves `rf_vs_logistic_regression_roc_curves.png` with class and macro AUCs.
Notebook 05 saves `cnn_vs_transfer_learning_roc_curves.png` for the original CNN
and YAMNet transfer model, using the same held-out test patients.
