# ANN on MNIST — Manual NumPy

This branch is a **self-contained executable ANN course**. It uses the same official MNIST dataset, split contract, architecture and evaluation philosophy as the PyTorch and TensorFlow branches so learners compare implementation mechanics rather than different experiments.

## Reproduce from a fresh clone
```bash
conda env create -f environment.yml
conda activate ann-mnist-manual
jupyter lab
```
Run `notebooks/00_environment_and_learning_map.ipynb` through `12_production_monitoring_drift_retraining_business.ipynb` in numerical order. For a clean non-interactive verification identical to CI:
```bash
make verify
```

## Data contract
- official MNIST 28×28 archive
- development subset sampled only from official training data
- deterministic stratified **5,000 train / 1,000 validation**
- deterministic **1,000 final test** examples sampled only from the official test archive
- `float32`, 784 flattened features, normalized to `[0,1]`
- primary MLP: **784 → 64 → 10**
- representation lab: **784 → 2 → 10**

## What this branch teaches
Data provenance and quality, leakage-safe splitting, preprocessing, mini-batching, neuron math, architecture and parameter counts, activation functions, initialization, forward propagation, logits/softmax, cross-entropy, chain rule, backpropagation, finite-difference gradient checks, SGD/Momentum/Adam concepts, full training/validation loops, validation-based hyperparameter selection, failure-mode diagnostics, representation learning, final test evaluation, error analysis, calibration, serialization/reload, single and batch inference, latency/throughput, production contracts, drift, monitoring, retraining and business impact.

## Quality contract
Every code cell has a substantial preceding technical explanation. The learner-facing notebooks are canonical source. CI clears outputs, executes every notebook from zero, validates syntax/documentation/execution counts/rendered outputs, rejects runtime errors, uploads the executed artifacts and commits the rendered notebooks back to this orphan branch.
