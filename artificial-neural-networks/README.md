# Artificial Neural Networks — Visual Learning Track

> A first-principles, executable path from **one artificial neuron** to **backpropagation, learned representations, and PyTorch**.

The notebooks are intentionally ordered. Each one introduces only the abstractions needed for the next step.

## Learning path

| # | Notebook | Core idea | What you should see |
|---:|---|---|---|
| 01 | [Artificial Neuron](./01_artificial_neuron.ipynb) | weights, bias, activation | slope, translation, decision boundaries |
| 02 | [Feed-Forward Network](./02_feed_forward_network.ipynb) | layers and forward propagation | activation flow and representation space |
| 03 | [Loss and Learning Objective](./03_loss_and_learning_objective.ipynb) | MSE and binary cross-entropy | loss curves and a 3D parameter-loss surface |
| 04 | [Gradients and Gradient Descent](./04_gradients_and_gradient_descent.ipynb) | derivatives and optimization | downhill parameter updates and learning-rate behavior |
| 05 | [Backpropagation From Scratch](./05_backpropagation_from_scratch.ipynb) | chain rule through a network | forward graph, backward gradients, gradient checking |
| 06 | [Train an MLP From Scratch](./06_train_mlp_from_scratch.ipynb) | complete learning loop | XOR and a learned nonlinear decision boundary |
| 07 | [PyTorch Autograd and Modules](./07_pytorch_autograd_and_modules.ipynb) | autograd and `nn.Module` | mapping framework abstractions back to first principles |
| 08 | [Training Dynamics and Failure Modes](./08_training_dynamics_and_failure_modes.ipynb) | optimization pathologies | saturation, vanishing/exploding gradients, overfitting |

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

> **Core principle:** forward propagation tells us what the network currently represents; backpropagation tells us how its parameters must change to represent the target better.

## Notebook philosophy

These are not thin wrappers around hidden utility modules.

- Core mathematics stays **inside the notebook**.
- Intermediate tensors and scalar values are inspectable.
- Visuals are paired with the exact equations that produced them.
- Framework code is introduced only after the same mechanism is implemented manually.
- Math uses GitHub/Jupyter-safe `$...$` and `$$...$$` delimiters consistently.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r artificial-neural-networks/requirements.txt
jupyter lab
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

For the interactive controls, run the notebooks in JupyterLab or another environment with `ipywidgets` support.

## Recommended order

Start with **01** and run through **08** sequentially. The conceptual dependency is deliberate:

```text
neuron
  ↓
feed-forward network
  ↓
loss
  ↓
gradients
  ↓
backpropagation
  ↓
complete MLP training
  ↓
PyTorch abstraction
  ↓
training dynamics
```

The first six notebooks expose the learning mechanism directly with NumPy. Notebook 07 maps the same mechanics to PyTorch. Notebook 08 shows why mathematically valid networks can still train poorly.
