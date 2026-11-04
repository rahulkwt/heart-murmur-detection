# Patient-level aggregation

Run from the project root:

```bash
python -m src.evaluation.aggregate_patients
```

This uses the saved RF and CNN window probabilities; no retraining or raw
audio is needed. Outputs contain validation patients only. Test metrics
are not computed. Input identities, labels, splits and probabilities are
checked, including matching windows between models.

Two methods are compared:

- `window_mean`: average all window probabilities for each patient.
- `recording_mean`: average windows within each recording, then average
  recording probabilities for each patient. Longer recordings therefore
  do not receive extra weight. Each recording has equal weight; this does
  not imply equal weight per auscultation location if a location has
  multiple recordings.

The highest mean probability determines the patient label. Exact ties
resolve in the fixed order Present, Absent, Unknown. All three classes are
included in macro-F1; undefined per-class metrics are reported as zero.

`validation_comparison.csv` contains accuracy, macro-F1, and per-class
precision, recall, F1 and support. The four `*_val_patients.csv` files
contain patient probabilities, predictions, true labels, and recording
and window counts. Patient IDs should be read as strings.

On the supplied 141 validation patients:

| Model | Method | Accuracy | Macro-F1 |
|---|---|---:|---:|
| RF | window_mean | 0.7730 | 0.5998 |
| RF | recording_mean | 0.7801 | 0.6186 |
| CNN | window_mean | 0.8723 | 0.7733 |
| CNN | recording_mean | 0.8794 | 0.7842 |

Recommendation: use `recording_mean` for both models for final evaluation.
It improves validation macro-F1 for both and removes recording-duration
weighting. These small improvements on a single validation split are not
evidence of statistical significance. Freeze the aggregation rule before
computing final test metrics. The models themselves were originally tuned
using window-level validation macro-F1 and have not been retuned here.
