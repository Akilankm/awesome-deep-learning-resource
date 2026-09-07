# Executed ANN Learning Notebooks — PyTorch

These `.ipynb` files are intentionally committed **with outputs**. GitHub should render the explanations, printed values, training metrics, and plots directly.

## How to study

Do not jump immediately to the training notebook. The sequence is designed so each abstraction is earned:

`data → neuron → forward propagation → activation/initialization → loss → gradients/backprop → training loop → representation learning → evaluation → inference → production`

Each notebook combines four views:

1. **Mathematics** — what is being computed.
2. **Code** — how PyTorch expresses that computation.
3. **Visual evidence** — shapes, plots, gradients, representations, mistakes, and metrics.
4. **Engineering/business inference** — why the behavior matters in an actual system.

## Run locally

From the repository root, create/activate the Conda environment described in `../environment.yml`, start JupyterLab, and run notebooks in numerical order. The notebooks download and cache the official `mnist.npz` into `data/` automatically when needed.

## Important reproducibility rule

CI clears every committed output before execution. Therefore a green workflow means these rendered outputs were reproduced from the code—not merely carried forward from a previous run.
