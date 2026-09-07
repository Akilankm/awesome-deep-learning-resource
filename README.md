# MNIST End-to-End Reference — PyTorch + TensorFlow

This branch is intentionally compact: **two self-contained notebooks, one complete lifecycle in each framework**.

## Notebooks

- `notebooks/01_mnist_e2e_pytorch.ipynb`
- `notebooks/02_mnist_e2e_tensorflow.ipynb`

Both use the same official MNIST `mnist.npz` source and the same development contract:

- 50,000 training samples
- 10,000 validation samples (stratified from the original 60,000 training set)
- 10,000 untouched official test samples
- 28×28 grayscale → 784 normalized float features
- ANN: 784 → 128 → 64 → 10
- 5 epochs
- Adam optimizer

## Lifecycle covered in both notebooks

`data load → data quality → train/validation/test → preprocessing → batching → model → training → validation → learning curves → final test → precision/recall/F1 → confusion matrix → error analysis → save/reload → inference`

## Create the environment

```bash
conda env create -f environment.yml
conda activate ann-mnist-e2e
jupyter lab
```

Open `notebooks/` and run either notebook top-to-bottom in a fresh kernel.

## Verify exactly as CI does

```bash
python scripts/execute_notebooks.py
python scripts/validate_notebooks.py
```

GitHub Actions executes both notebooks from cleared outputs and commits the rendered results back to this branch.
