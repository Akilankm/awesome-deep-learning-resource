# Artificial Neural Networks — Visual Learning Track

> A first-principles, executable path from **one artificial neuron** to **backpropagation, learned representations, real data, and PyTorch**.

The notebooks are intentionally ordered. Each one introduces only the abstractions needed for the next step.

## Core learning path

| # | Notebook | Core idea | What you should see |
|---:|---|---|---|
| 01 | [Artificial Neuron](./01_artificial_neuron.ipynb) | weights, bias, activation | slope, translation, decision boundaries |
| 02 | [Feed-Forward Network](./02_feed_forward_network.ipynb) | layers and forward propagation | activation flow and representation space |
| 03 | [Loss and Learning Objective](./03_loss_and_learning_objective.ipynb) | MSE and binary cross-entropy | loss curves and a 3D parameter-loss surface |
| 04 | [Gradients and Gradient Descent](./04_gradients_and_gradient_descent.ipynb) | derivatives and optimization | downhill parameter updates and learning-rate behavior |
| 05 | [Backpropagation From Scratch](./05_backpropagation_from_scratch.ipynb) | chain rule through a network | forward graph, backward gradients, gradient checking |
| 06 | [Train an MLP From Scratch](./06_train_mlp_from_scratch.ipynb) | complete learning loop | learned nonlinear decision boundary |
| 07 | [PyTorch Autograd and Modules](./07_pytorch_autograd_and_modules.ipynb) | autograd and `nn.Module` | framework mapping to first principles |
| 08 | [Training Dynamics and Failure Modes](./08_training_dynamics_and_failure_modes.ipynb) | optimization pathologies | saturation, gradient flow, overfitting |

## Real-data visual extension

These notebooks use **scikit-learn's real handwritten digits dataset** so the ideas are no longer only synthetic.

| # | Notebook | What it teaches visually |
|---:|---|---|
| 10 | [Real Data Walkthrough — Handwritten Digits](./10_real_data_digits_walkthrough.ipynb) | real image → interpretable features → weighted contributions → neuron activation |
| 11 | [Representation Learning on Real Digits](./11_representation_learning_on_real_digits.ipynb) | the same samples physically move in a 2D hidden representation as backprop changes weights |

The rendered teaching assets used by these notebooks live in [`visual_learning/`](./visual_learning/).

## The mental model

The entire track revolves around one computational loop:

$$
\mathbf{x}
\rightarrow
\mathbf{z}=W\mathbf{x}+\mathbf{b}
\rightarrow
\mathbf{a}=\phi(\mathbf{z})
\rightarrow
\hat{y}
\rightarrow
\mathcal{L}
$$

Training adds the reverse path:

$$
\mathcal{L}
\xrightarrow{\text{backprop}}
\nabla_W\mathcal{L},\nabla_b\mathcal{L}
\xrightarrow{\text{optimizer}}
W',b'
$$

> **Core principle:** forward propagation tells us what the network currently represents; backpropagation changes the parameters, which changes the representation seen on the next forward pass.

## Notebook philosophy

These are not thin wrappers around hidden utility modules.

- Core mathematics stays **inside the notebook**.
- Intermediate tensors and scalar values are inspectable.
- Manual calculations are mapped to code.
- Visuals are paired with the exact quantities that produced them.
- The real-data extension uses rendered SVG teaching assets so GitHub itself remains useful without first running Jupyter.
- Framework code is introduced only after the same mechanism is understood manually.
- Math uses GitHub/Jupyter-safe `$...$` and `$$...$$` delimiters consistently.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r artificial-neural-networks/requirements.txt
jupyter lab
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

For interactive controls in the earlier notebooks, use JupyterLab or another environment with `ipywidgets` support.
