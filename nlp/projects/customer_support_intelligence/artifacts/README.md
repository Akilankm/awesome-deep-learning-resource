# Reference-run artifacts

This directory keeps compact, human-readable results from the validated **ready-made-data** execution.

## Versioned reference results

- `metrics.json`
- `retrieval_metrics.csv`
- `entity_extraction_metrics.csv`
- `eda_class_distribution.svg`
- `eda_mean_token_length.svg`
- `model_comparison.svg`
- `confusion_matrix.svg`
- `slice_analysis.svg`
- `retrieval_metrics.svg`

These SVGs make the key results visible directly on GitHub. They correspond to the reference metrics documented in the notebook.

## Regenerated locally

Executing the notebook recreates the wider runtime artifact set, including the serialized model, cleaned/split data, predictions, errors, tuning/model-comparison tables, monitoring snapshot and Matplotlib PNG/SVG plots.

Binary model artifacts are intentionally generated locally rather than committed.

Current reference classifier:
- test accuracy: **0.8913**
- test macro-F1: **0.8896**
- held-out errors: **10**
