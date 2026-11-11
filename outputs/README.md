# Model outputs

Saved models, predictions, and evaluation results. Setup and run instructions
are in the [main README](../README.md).

| Files or folder | Contents |
|---|---|
| `*_window_predictions.csv` | RF, CNN, YAMNet, and CardioPHON probabilities for each audio window |
| `rf_model.joblib`, `cnn_model.keras` | Trained baseline models |
| `cardiophon_classifier.keras`, `cardiophon_metadata.json` | Transfer classifier and encoder/normalization settings |
| `*_training_history.csv`, `*_tuning_*.csv` | Training curves and validation tuning results |
| `transfer_learning_*_comparison.csv` | Patient-level RF, CNN, YAMNet, and CardioPHON scores |
| `evaluation/` | Baseline test predictions, metrics, and figures |
| `patient_aggregation/` | Validation comparisons of patient aggregation methods |
| `additional_analysis/` | Logistic Regression predictions and extended model comparisons |
| `figures/` | Baseline training plots |

Prediction rows contain patient and recording IDs, window index, location,
split, true/predicted labels, and three class probabilities. Class order is
**Present, Absent, Unknown**, as recorded in `label_mapping.json`; splits are
`train`, `val`, and `test`.

All models use the same four-second windows and patient splits. Final patient
predictions average windows within recordings, then give each recording equal
weight. Test data is reserved for evaluation. Unknown has only ten test
patients, so its scores should be interpreted cautiously.

Notebook figures are saved in [reports/figures](../reports/figures/), including
[additional analysis](../reports/figures/additional_analysis/).
