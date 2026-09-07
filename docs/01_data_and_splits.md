# Data, EDA, Leakage, and Train/Validation/Test

Training data teaches parameters; validation data supports model-selection decisions; test data is the final unbiased estimate after development is frozen. Leakage occurs when information unavailable at prediction time or evaluation-only information influences training or selection.

Document provenance, schema, label definition, missingness, duplicates, class balance, outliers, timestamps, sampling strategy and split IDs. Stratify classification splits when class balance matters. Preserve deterministic seeds while recognizing that one seed is not uncertainty estimation.

## Learning questions
- What assumption does this stage make?
- What can fail silently?
- What evidence would you monitor?
- How does this affect business or production behavior?
