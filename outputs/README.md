# Model outputs (Random Forest and CNN)

Window-level outputs of the two murmur models, for subject-level aggregation
and evaluation (Sadhana). The models are a Random Forest on MFCC summary
features and a CNN on log-Mel spectrograms.

## How to reproduce

Put the CirCor v1.0.3 `training_data/` folder in `data/raw/training_data/`.
Then run these commands from the project root, in order. Times are for an
8-core laptop CPU with no GPU.

```bash
pip install -r requirements.txt
python src/preprocessing/create_indexes.py   # patient/recording index      (~10 s)
python src/preprocessing/split_patients.py   # subject-level split, seed 42 (~5 s)
python -m src.models.data_utils              # windows + feature cache      (~6.5 min)
python -m src.models.train_rf                # Random Forest grid search    (~5 min)
python -m src.models.train_cnn               # CNN search, 5 settings       (~55 min; much faster on a GPU)
python -m src.models.make_figures            # overfitting plots            (~5 s)
python -m src.models.subject_check           # validation diagnostic        (~2 s)
```

The feature cache is written to `data/processed/model_cache/`, which git
ignores. Seeds: 42 everywhere. The CNN uses deterministic TensorFlow
operations. Results are reproducible on the same machine and versions, but
may differ slightly on other hardware.

## Windows

- **What a window is.** Recordings are preprocessed with the project's
  `preprocess_audio()`: 4 kHz, 25–800 Hz band-pass, and peak normalization per
  recording. They are then cut into **4-second windows** with `create_windows()`,
  and the last window of each recording is zero-padded.
- **Dropped windows.** Windows with **less than 1 s of real audio are dropped**:
  820 of 19,651, which leaves **18,831**.
- **Same windows for both models.** Both models use exactly the same windows,
  in the same row order.

| split | windows kept: Present / Absent / Unknown | dropped: Present / Absent / Unknown |
|---|---|---|
| train | 2,627 / 10,039 / 671 | 106 / 435 / 26 |
| val | 486 / 2,092 / 132 | 22 / 98 / 3 |
| test | 465 / 2,211 / 108 | 27 / 101 / 2 |

## Labels and splits

- **3 classes:** Present, Absent and Unknown. These come from the subject's
  `#Murmur:` field. `label_mapping.json` gives `{"Present": 0, "Absent": 1, "Unknown": 2}`,
  and the probability columns follow that order.
- **Note:** teammate commit `d9cf714` changed `prepare_dataset()` to drop Unknown.
  My code does not edit that file. Instead, `src/models/data_utils.py` calls
  `preprocess_audio()` and `create_windows()` directly and builds the windows in
  memory, including Unknown.
- **Split names.** The split is the subject-level split from `split_patients.py`
  (70/15/15, stratified, seed 42). No subject is in more than one split.
  **The CSVs use `train`, `val`, `test`**; the repo's files call the middle split
  `validation`.

## Files

| File | Contents |
|---|---|
| `rf_window_predictions.csv` | Random Forest predictions, one row per window, all splits |
| `cnn_window_predictions.csv` | CNN predictions, one row per window, all splits |
| `cnn_training_history.csv` | Per-epoch loss/accuracy of the selected CNN |
| `rf_model.joblib` | Trained Random Forest (`joblib.load(...)`) |
| `cnn_model.keras` | Trained CNN (`keras.models.load_model(...)`). Takes raw log-Mel input of shape `(64, 126, 1)`; it standardizes internally. |
| `label_mapping.json` | Class name → index |
| `rf_tuning_results.csv` | Train and validation macro-F1 (window level) for every RF grid setting; an empty `max_depth` means None (no limit) |
| `cnn_tuning_results.csv` | Validation macro-F1 for every CNN setting |
| `cnn_tuning_histories.csv` | Per-epoch history of every CNN trial (column `trial`) |
| `subject_level_val_check.csv` | Quick subject-level diagnostic on **validation only** (see below) |
| `figures/cnn_loss_curves.png`, `figures/cnn_accuracy_curves.png` | Train vs val curves of the selected CNN; dashed line = best epoch |
| `figures/rf_train_vs_val.png` | Train vs val macro-F1 for each RF setting |

### `*_window_predictions.csv` columns

| Column | Meaning |
|---|---|
| `subject_id` | Patient ID (read as a string: `dtype={"subject_id": str}`) |
| `recording_id` | Recording file name without `.wav`, e.g. `13918_AV`, `50782_MV_1` |
| `location` | Auscultation location: AV, PV, TV, MV or Phc |
| `window_idx` | Window position in the recording (0 = 0–4 s, 1 = 4–8 s, ...). Dropped windows leave a gap at the end. |
| `split` | `train`, `val` or `test` |
| `true_label` | Subject-level label: Present, Absent or Unknown |
| `pred_label` | Class with the highest probability |
| `prob_present`, `prob_absent`, `prob_unknown` | Class probabilities (sum to 1) |

### `cnn_training_history.csv` columns

`epoch` (from 1), `loss`, `val_loss`, `accuracy`, `val_accuracy`. Training loss
uses class weights; validation loss does not.

### `subject_level_val_check.csv`

This is a **quick diagnostic on the validation split only, not the official
result.** Each subject's window probabilities are averaged over all its windows
and locations, and the subject gets the class with the highest average. The
columns are `model`, `split`, `n_subjects`, `accuracy`, `macro_f1`,
`f1_present`, `f1_absent` and `f1_unknown`. **Sadhana's evaluation is the
official subject-level result.** No metric was computed on the test split.

## Models and tuning

- **Random Forest.** Input: 56 features per window. These are the mean and std
  of 13 MFCCs (the project's `extract_mfcc_features()`), the mean and std of 13
  delta-MFCCs, and RMS and zero-crossing-rate mean/std, computed on the real
  (unpadded) audio. Uses `class_weight="balanced"`. The setting was chosen by
  validation macro-F1.
  - **The grid was widened once after seeing validation results.** The first
    grid (`max_depth` None/20, `min_samples_leaf` 1/5) overfit. The final grid
    is `n_estimators` 200/500, `max_depth` 10/None, `min_samples_leaf` 5/20/50.
    Both rounds used validation data only.
- **CNN.** Input: log-Mel spectrogram (64 Mel bands, 25–800 Hz, n_fft=512,
  hop=128, dB) of shape `(64, 126, 1)`. The network has 3 × (Conv2D → MaxPool),
  global average pooling, dropout, Dense(64), dropout and a 3-way softmax.
  Training uses Adam, categorical cross-entropy and class weights.
  - **Search:** 5 settings, each changing one of learning rate, batch size,
    filters and dropout from a base. The best was chosen by validation macro-F1.
  - **Early stopping:** on `val_loss` with **patience 6**, restoring the best
    epoch's weights.
- **Test set:** used only to write the final predictions. Never used for
  tuning, early stopping or any metric here.

## Known limitations

- **Window labels are subject labels.** Every window gets its subject's label,
  but a murmur may be audible at only some locations. Many "Present" windows
  may therefore contain no audible murmur, so the window-level labels are noisy.
- **Unknown is tiny.** Only 10 subjects are Unknown in val and 10 in test
  (48 in train), so Unknown scores are unstable.
- **Single seed.** All results come from one seed (42) and one split. There are
  no error bars.
- **Train predictions are optimistic.** The train-split predictions come from
  models fit on those windows. Use val and test for evaluation.
