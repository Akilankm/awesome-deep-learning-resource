# ANN on MNIST — Manual NumPy

This is a **true orphan branch**: it has its own zero-parent history and is never intended to merge into `master`.

## Purpose
Learn artificial neural networks from first principles on the official MNIST dataset with no PyTorch or TensorFlow hiding the learning mechanics. The notebooks keep the core ANN logic directly visible: weighted sums, activations, softmax, cross-entropy, gradients, manual backpropagation, parameter updates, representation learning, inference, debugging, and business decision trade-offs.

## Shared experiment contract
This branch is deliberately comparable with `ann-mnist-pytorch` and `ann-mnist-tensorflow`:
- official MNIST 28×28 handwritten digits
- deterministic balanced teaching subset: 5,000 train / 1,000 test
- normalized 784-pixel input
- primary classifier: 784 → 64 → 10
- representation lab: 784 → 2 → 10
- same learning objectives and evaluation views

## Learning path
1. `00_mnist_learning_map.ipynb`
2. `01_pixels_to_neurons.ipynb`
3. `02_forward_propagation.ipynb`
4. `03_loss_gradients_backprop.ipynb`
5. `04_training_and_representation_learning.ipynb`
6. `05_inference_debugging_business.ipynb`

CI generates the canonical notebooks, executes every code cell from a cleared state, validates outputs, and commits the **rendered notebooks with results** back to this orphan branch.

## Conda
```bash
conda env create -f environment.yml
conda activate ann-mnist-manual
jupyter lab
```

## Quality contract
A notebook is accepted only if it has valid nbformat, substantial explanatory Markdown, compilable code, execution counts on every code cell, persisted outputs, and zero error outputs.
