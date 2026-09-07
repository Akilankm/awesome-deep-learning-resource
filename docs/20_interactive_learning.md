# Interactive Learning: GitHub Static View vs VS Code/Jupyter Lab

This repository deliberately supports **two complementary learning modes from the same notebook**.

## 1. GitHub mode — static, reviewable, zero setup

When a learner opens an executed `.ipynb` directly on GitHub, the committed notebook remains useful without running any code:

- technical Markdown and equations are visible;
- code is visible and reviewable;
- execution counts are persisted;
- numeric results, tables and metrics are persisted;
- Matplotlib/static figures remain visible;
- CI has already executed and validated the notebook from a cleared state.

GitHub does not execute arbitrary notebook JavaScript, so Plotly interaction itself is not the contract of the GitHub viewer. The committed static plots are the durable fallback.

## 2. VS Code / Jupyter mode — executable interactive laboratory

After cloning one implementation branch, create its pinned Conda environment and open the notebooks locally:

```bash
conda env create -f environment.yml
conda activate ann-mnist-manual   # or ann-mnist-pytorch / ann-mnist-tensorflow
code .
```

Open a notebook, select the matching Conda kernel and choose **Run All**.

Cells headed **Interactive Plotly lab — clone + run locally** use Plotly's `plotly_mimetype` renderer, which is supported by VS Code notebooks and JupyterLab. The learner can then:

- hover exact values;
- zoom and pan;
- rotate 3D scenes;
- click legends to isolate traces;
- move parameter sliders;
- animate training epochs;
- inspect individual MNIST samples and hidden coordinates.

## Where interactivity is used

The implementation branches intentionally add interaction only where it improves technical understanding.

| Notebook | Interactive learning objective |
|---|---|
| `03_neuron_linear_algebra_architecture.ipynb` | Rotate the affine neuron surface and change weights/bias. |
| `04_activations_initialization_gradient_flow.ipynb` | Compare activation functions and local derivatives; inspect saturation and gradient flow. |
| `05_forward_logits_softmax.ipynb` | Change softmax temperature while keeping logits fixed to separate ranking from confidence. |
| `07_optimizers_end_to_end_training.ipynb` | Rotate a 3D loss landscape and compare gradient-descent trajectories for different learning rates. |
| `09_representation_learning_visualized.ipynb` | Animate the same 1,500 MNIST samples through hidden space across epochs and hover sample-level coordinates. |
| `10_test_evaluation_error_calibration.ipynb` | Hover calibration bins and inspect predicted confidence versus empirical accuracy. |
| `12_production_monitoring_drift_retraining_business.ipynb` | Move a drift-severity slider and inspect how the MNIST input distribution shifts. |

## Why the CI runner suppresses the JavaScript display

The notebook still constructs every Plotly figure during GitHub Actions, which catches Python/API errors. In CI it prints a confirmation instead of asking a headless runner to display an interactive JavaScript surface. Locally, the exact same cell calls:

```python
fig.show(renderer="plotly_mimetype")
```

This preserves three properties at once:

1. **CI reproducibility** — interactive figure construction is executed and validated.
2. **GitHub readability** — static plots and explanations remain visible.
3. **Local exploration** — the cloned notebook becomes fully interactive in VS Code/Jupyter.

## Learning principle

Interactivity is not decorative. Each control exposes a causal mechanism:

**parameter → internal computation → model behavior → engineering consequence → business consequence**.

A learner should be able to use the interactive view to form a hypothesis, inspect the numerical effect, and then return to the mathematics/code to explain why the effect occurred.
