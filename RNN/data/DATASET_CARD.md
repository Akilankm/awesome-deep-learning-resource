# Dataset card — Jena Climate

The flagship RNN project uses the Jena Climate time-series dataset distributed through TensorFlow/Keras educational examples. The raw observations were recorded at the Max Planck Institute for Biogeochemistry weather station in Jena, Germany at 10-minute intervals.

The project downloads the canonical compressed CSV with `tf.keras.utils.get_file`, downsamples it to hourly resolution for a CPU-friendly educational experiment, and predicts future temperature from recent multivariate weather history.

**Important split contract:** all train/validation/test boundaries are chronological. Normalization statistics are learned on training data only.
