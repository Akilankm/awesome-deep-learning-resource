# ANN on MNIST — TensorFlow / Keras

This is a **true orphan branch** with its own zero-parent history. It is intentionally independent from `master`, `ann-mnist-manual`, and `ann-mnist-pytorch`.

## Purpose
Learn the same ANN problem on the same MNIST data while mapping the mathematics into TensorFlow tensors, `GradientTape`, Keras layers, explicit optimization steps, representation learning, inference, debugging, and production decisioning.

The learning path starts below `model.fit()` so TensorFlow is not treated as a black box. The notebooks show the primitive tensor operations and gradients before introducing framework abstractions.

## Shared experiment contract
- official MNIST 28×28 handwritten digits
- deterministic balanced teaching subset: 5,000 train / 1,000 test
- normalized 784-pixel input
- primary classifier: 784 → 64 → 10
- representation lab: 784 → 2 → 10
- same learning objectives and evaluation views as the manual and PyTorch branches

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
conda activate ann-mnist-tensorflow
jupyter lab
```

## Quality contract
Valid notebook schema, substantial explanation, compilable code, non-null execution counts, persisted outputs, and zero notebook error outputs are required.
