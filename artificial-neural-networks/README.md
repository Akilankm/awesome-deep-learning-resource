# Artificial Neural Networks — Gold-Standard Visual Learning Track

> **Objective:** make artificial neural networks mechanically transparent, visually intuitive, mathematically rigorous, executable, and connected to production/business decisions.

The notebooks use **real handwritten digit data** for the core mechanics. Each notebook contains the calculations inside the notebook rather than hiding the learning logic behind utility modules.

## Learning path

| # | Notebook | Main question |
|---:|---|---|
| 00 | [Visual roadmap + business context](./00_ann_visual_roadmap_and_business_context.ipynb) | How do all ANN concepts fit together? |
| 01 | [Artificial neuron](./01_artificial_neuron.ipynb) | What exactly do weight, bias, and activation do? |
| 02 | [Feed-forward network](./02_feed_forward_network.ipynb) | What happens to one real sample layer by layer? |
| 03 | [Loss](./03_loss_and_learning_objective.ipynb) | How is “wrong” converted into a scalar objective? |
| 04 | [Gradients + gradient descent](./04_gradients_and_gradient_descent.ipynb) | How does a parameter know which direction to move? |
| 05 | [Backpropagation from scratch](./05_backpropagation_from_scratch.ipynb) | How does loss sensitivity reach every weight? |
| 06 | [Train MLP from scratch](./06_train_mlp_from_scratch.ipynb) | How do forward/backward/update form a real training loop? |
| 07 | [Representation learning on real digits](./07_representation_learning_on_real_digits.ipynb) | How do hidden coordinates change during learning? |
| 08 | [Training dynamics + failure modes](./08_training_dynamics_and_failure_modes.ipynb) | Why can mathematically correct networks still train badly? |
| 09 | [PyTorch equivalence](./09_pytorch_autograd_and_modules.ipynb) | What does autograd automate? |
| 10 | [Inference + debugging](./10_inference_and_model_debugging.ipynb) | What happens after training, one unseen sample at a time? |
| 11 | [Business decision lab](./11_business_decision_lab.ipynb) | How does a score become an operational action and KPI? |

## Learning design

Every core notebook follows:

**intuition → real data → manual arithmetic → vectorized code → visual → what affects what → inference/insight → production/business connection**

## Reproducible environment

Recommended Python: **3.11**.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r artificial-neural-networks/requirements-notebook.txt
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"
jupyter lab
```

Or with Conda/Mamba:

```bash
conda env create -f artificial-neural-networks/environment.yml
conda activate ann-gold
```

## Validate and execute exactly like CI

```bash
python scripts/validate_notebooks.py
python scripts/execute_notebooks.py --in-place
python scripts/validate_notebooks.py --require-executed
```

The GitHub workflow performs these same checks. On pushes to the feature branch it commits executed notebook outputs back to GitHub, so the notebook preview contains the figures/tables/results instead of source-only cells.

## Static + interactive learning

GitHub renders committed static outputs. Notebook 01 additionally contains an optional `ipywidgets` live lab when opened in Jupyter. Static plots are always present so the educational content never depends on widget rendering.
