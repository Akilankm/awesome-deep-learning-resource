# ANN on MNIST — PyTorch

> **Goal:** take a learner from "I know the ANN formula" to "I can reason about, train, debug, evaluate, serialize, and operate an ANN system."

This is a **true orphan branch**. It is intentionally independent of `master` and the other MNIST tracks. The branch is self-contained: environment definition, executable notebooks, CI validation, and rendered notebook outputs all live here.

## What makes this resource different

- the official 28×28 MNIST dataset is used throughout
- the same deterministic teaching subset is used across the manual, PyTorch, and TensorFlow branches
- every important framework call is mapped back to the mathematics
- the central training loop is explicit rather than hidden behind a one-line training API
- code is written for inspection: shapes, gradients, parameter updates, metrics, and artifacts are exposed
- notebooks are committed **after execution**, so GitHub shows the results before you run anything locally
- CI clears every output and re-executes the full curriculum to prove reproducibility
- business and production consequences are explained alongside the mathematics and code

## Environment

Tested contract:

- Python 3.11
- NumPy 1.26.4
- Matplotlib 3.9.2
- pandas 2.2.3
- scikit-learn 1.5.2
- JupyterLab 4.3.4
- PyTorch CPU 2.5.1

### Recommended: Conda

```bash
git clone https://github.com/Akilankm/awesome-deep-learning-resource.git
cd awesome-deep-learning-resource
git checkout ann-mnist-pytorch

conda env create -f environment.yml
conda activate ann-mnist-pytorch
jupyter lab
```

Then open `notebooks/00_environment_and_learning_map.ipynb` and continue in numerical order.

### Verify the whole branch from the command line

```bash
make verify
```

`make verify` rebuilds the canonical notebook source, executes all notebooks from cleared outputs, and runs the same validation checks used by CI.

## Complete learning path

1. `00_environment_and_learning_map.ipynb` — environment, architecture, mental model, reproducibility
2. `01_mnist_data_and_preprocessing.ipynb` — raw MNIST, shape/dtype/range, flattening, inference contract
3. `02_neuron_tensor_math.ipynb` — one neuron, weights, bias, activation, feature contributions
4. `03_forward_propagation_end_to_end.ipynb` — full tensor flow from 784 pixels to ten logits
5. `04_activations_initialization_gradient_flow.ipynb` — ReLU, saturation, initialization, gradient diagnostics
6. `05_softmax_cross_entropy_loss.ipynb` — logits, probabilities, cross-entropy, confidence
7. `06_backpropagation_and_autodiff.ipynb` — manual derivative, finite difference, framework autodiff, optimizer update
8. `07_end_to_end_training_loop.ipynb` — full training/evaluation loop, mini-batches, optimizer, metrics, model structure
9. `08_representation_learning_visualized.ipynb` — 2D bottleneck and changing hidden geometry
10. `09_evaluation_error_analysis_calibration.ipynb` — confusion matrix, high-confidence mistakes, error analysis
11. `10_inference_serialization_reload.ipynb` — save, reload, reproducible inference artifact
12. `11_production_business_monitoring.ipynb` — drift, monitoring, SLA, governance, business impact

## Shared experiment contract

- official MNIST archive
- deterministic balanced teaching subset: 5,000 training / 1,000 test images
- normalization: integer pixels `[0,255]` → floating-point `[0,1]`
- dense classifier: `784 → 64 → 10`
- representation lab: `784 → 2 → 10`
- fixed seed for repeatable teaching runs

## Definition of done

A notebook is accepted only when it has valid notebook structure, substantial technical explanation, compilable Python, non-null execution counts, persisted outputs, required visual outputs, zero execution errors, and the expected framework-specific training semantics.

The notebooks under `notebooks/` are the **primary learning artifacts**. Files under `scripts/` exist only to reproduce and verify them.
