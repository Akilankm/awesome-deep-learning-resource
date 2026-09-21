# End-to-end Jena climate forecasting
Raw multivariate time-series data → quality checks → chronological split → train-only normalization → windowing → naive baseline → SimpleRNN/LSTM/GRU → validation selection → untouched test evaluation → saved Keras model → reload/inference → drift checks.
