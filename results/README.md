# Results

Generated files are grouped by type:

| Folder | Contents |
|---|---|
| `models/` | Trained models and model metadata |
| `predictions/` | Window-level model probabilities and labels |
| `metrics/` | Tuning results, patient predictions, comparisons, and JSON summaries |
| `figures/` | Figures grouped by notebook or evaluation task |

All models use the same four-second windows and patient splits. Patient-level
predictions average windows within each recording, then give each recording
equal weight. Class order is **Present, Absent, Unknown**.

The location-aware experiment uses recording-level training labels, selects the CNN by patient-level validation macro-F1, and saves its model, predictions, and test summary alongside the other results with the `location_aware_cnn` prefix.
