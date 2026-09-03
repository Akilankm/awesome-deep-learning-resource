# Artificial Neural Networks — Visual Learning Track

This section builds neural networks from first principles and then connects the mathematics to PyTorch.

## Learning path

| # | Notebook | Core idea | Visual focus |
|---|---|---|---|
| 01 | `01_artificial_neuron.ipynb` | neuron, weights, bias, activation | slope, shifts, decision boundaries |
| 02 | `02_feed_forward_network.ipynb` | layers and forward propagation | activation flow and representation space |
| 03 | `03_loss_and_learning_objective.ipynb` | MSE and binary cross entropy | loss curves and 3D loss surfaces |
| 04 | `04_gradients_and_gradient_descent.ipynb` | derivatives and optimization | animated descent and learning-rate behavior |
| 05 | `05_backpropagation_from_scratch.ipynb` | chain rule through networks | computational graph + gradient check |
| 06 | `06_train_mlp_from_scratch.ipynb` | complete training loop | XOR and learned nonlinear boundary |
| 07 | `07_pytorch_autograd_and_modules.ipynb` | autograd and `nn.Module` | framework mapping to first principles |
| 08 | `08_training_dynamics_and_failure_modes.ipynb` | practical optimization failures | saturation, gradient flow, overfitting |

## Philosophy

These notebooks deliberately expose the core logic. The important calculations are kept in the notebook instead of hidden behind helper modules so the learner can execute cells in order and inspect every intermediate value.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

For interactive `ipywidgets`, use JupyterLab or a notebook environment with widget support.

## Recommended order

Run the notebooks from `01` through `08`. The first six require only NumPy/Matplotlib/IPython/ipywidgets. Notebook 07 additionally uses PyTorch.
