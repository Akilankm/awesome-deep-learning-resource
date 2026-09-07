# ANN on MNIST — PyTorch

This is a **true orphan branch** with a zero-parent history, independent of `master` and the other MNIST tracks.

## Purpose
Learn the same ANN mechanics as the manual branch, but mapped explicitly into PyTorch tensors, `nn.Module`, autograd, optimizer steps, representation learning, inference, debugging, and production decisioning.

The notebooks intentionally avoid turning PyTorch into a black box: scalar/vector equations are shown first, gradients are inspected directly, and framework abstractions are tied back to the same mathematics used in `ann-mnist-manual`.

## Shared experiment contract
- official MNIST 28×28 handwritten digits
- deterministic balanced teaching subset: 5,000 train / 1,000 test
- normalized 784-pixel input
- primary classifier: 784 → 64 → 10
- representation lab: 784 → 2 → 10
- same learning objectives and evaluation views as the manual and TensorFlow branches

## Learning path
1. `00_mnist_learning_map.ipynb`
2. `01_pixels_to_neurons.ipynb`
3. `02_forward_propagation.ipynb`
4. `03_loss_gradients_backprop.ipynb`
5. `04_training_and_representation_learning.ipynb`
6. `05_inference_debugging_business.ipynb`

CI generates the notebooks, executes every code cell from a clean state, validates outputs, and commits the rendered notebooks back to this orphan branch.

## Conda
```bash
conda env create -f environment.yml
conda activate ann-mnist-pytorch
jupyter lab
```

## Quality contract
Valid notebook schema, substantial explanation, compilable code, non-null execution counts, persisted outputs, and zero notebook error outputs are required.
