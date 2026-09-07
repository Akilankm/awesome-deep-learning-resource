from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path.cwd()
NOTEBOOKS = ROOT / "notebooks"
MARKER = "### Interactive Plotly lab — clone + run locally"


def _id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _cell(kind: str, source: str, key: str) -> dict:
    cell = {
        "cell_type": kind,
        "id": _id(key),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def append_lab(filename: str, title: str, explanation: str, code: str) -> None:
    path = NOTEBOOKS / filename
    nb = json.loads(path.read_text())
    text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    if MARKER in text and title in text:
        print(filename, "already enhanced")
        return
    markdown = f"""{MARKER}\n\n## {title}\n\n> **GitHub vs local behavior.** GitHub keeps the committed static plots, tables, metrics and explanations visible. **Clone this branch, create `environment.yml`, open the notebook in VS Code/Jupyter, select the Conda kernel, and run the cell below for full Plotly interactivity**: hover, zoom, pan, 3D rotation, legend selection, sliders and animation.\n\n{explanation}\n\n**Production/business interpretation.** The interaction is an inspection instrument, not decoration: change or inspect a technical quantity and connect it to model behavior, operational risk or downstream decision quality.\n"""
    key = filename + title
    nb["cells"].extend(
        [_cell("markdown", markdown, "md-" + key), _cell("code", code, "code-" + key)]
    )
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(filename, "enhanced")


def ensure_dependencies() -> None:
    req = ROOT / "requirements.txt"
    text = req.read_text()
    if "plotly==7.0.0" not in text:
        req.write_text(text.rstrip() + "\nplotly==7.0.0\n")
    env = ROOT / "environment.yml"
    text = env.read_text()
    if "plotly==7.0.0" not in text:
        lines = text.splitlines()
        insert_at = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("      - "):
                insert_at = i + 1
                break
        lines.insert(insert_at, "      - plotly==7.0.0")
        env.write_text("\n".join(lines) + "\n")


def update_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text()
    if "## Interactive Plotly labs" in text:
        return
    branch = os.environ.get("GITHUB_REF_NAME", "<this-branch>")
    addition = f"""\n\n## Interactive Plotly labs\n\nGitHub intentionally remains a **static, reviewable learning surface**. The committed plots, metrics and explanations are readable without executing code. Concept-heavy notebooks also contain marked Plotly labs.\n\n```bash\nconda env create -f environment.yml\nconda activate {branch}\ncode .\n```\n\nOpen `notebooks/` in VS Code, select the Conda kernel and **Run All**. Cells marked **Interactive Plotly lab — clone + run locally** support hover, zoom, pan, 3D rotation, sliders, animation and point-level inspection. Plotly's `plotly_mimetype` renderer is supported by VS Code notebooks and JupyterLab.\n\nThe same notebook therefore has two valid modes: **GitHub = persisted static learning artifact; VS Code/Jupyter = executable interactive laboratory.**\n"""
    path.write_text(text.rstrip() + addition + "\n")


NEURON = """import os\nimport numpy as np\nimport plotly.graph_objects as go\n\ngrid = np.linspace(-2.0, 2.0, 55)\nx1, x2 = np.meshgrid(grid, grid)\nconfigs = [("w=(0.4,-0.8), b=0.1", 0.4, -0.8, 0.1), ("w=(1.0,-0.3), b=0.1", 1.0, -0.3, 0.1), ("w=(-0.6,0.9), b=-0.2", -0.6, 0.9, -0.2)]\nsurfaces = [w1 * x1 + w2 * x2 + b for _, w1, w2, b in configs]\nfig = go.Figure(data=[go.Surface(x=x1, y=x2, z=surfaces[0], colorscale="Viridis", colorbar=dict(title="z = w·x + b"), hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<br>z=%{z:.3f}<extra></extra>")], frames=[go.Frame(name=str(i), data=[go.Surface(x=x1, y=x2, z=s, colorscale="Viridis", showscale=True)]) for i, s in enumerate(surfaces)])\nfig.update_layout(title="Interactive affine neuron geometry", scene=dict(xaxis_title="input x1", yaxis_title="input x2", zaxis_title="pre-activation z"), sliders=[dict(currentvalue=dict(prefix="parameter setting: "), steps=[dict(label=label, method="animate", args=[[str(i)], dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0))]) for i, (label, *_rest) in enumerate(configs)])])\nif os.getenv("GITHUB_ACTIONS") == "true":\n    print("Interactive neuron surface built. Run locally to rotate, hover and move the parameter slider.")\nelse:\n    fig.show(renderer="plotly_mimetype")\n"""

ACTIVATIONS = """import os\nimport numpy as np\nimport plotly.graph_objects as go\n\nx = np.linspace(-6, 6, 500)\ns = 1.0 / (1.0 + np.exp(-x))\nfunctions = {"Sigmoid": (s, s * (1 - s)), "Tanh": (np.tanh(x), 1 - np.tanh(x) ** 2), "ReLU": (np.maximum(x, 0), (x > 0).astype(float)), "Leaky ReLU": (np.where(x > 0, x, 0.1 * x), np.where(x > 0, 1.0, 0.1))}\nfig = go.Figure()\nfor name, (y, dy) in functions.items():\n    visible = name == "Sigmoid"\n    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{name} activation", visible=visible, hovertemplate="z=%{x:.2f}<br>a=%{y:.4f}<extra></extra>"))\n    fig.add_trace(go.Scatter(x=x, y=dy, mode="lines", name=f"{name} derivative", line=dict(dash="dash"), visible=visible, hovertemplate="z=%{x:.2f}<br>da/dz=%{y:.4f}<extra></extra>"))\nnames = list(functions)\nbuttons = []\nfor idx, name in enumerate(names):\n    visible = [False] * (2 * len(names)); visible[2 * idx] = visible[2 * idx + 1] = True\n    buttons.append(dict(label=name, method="update", args=[dict(visible=visible), dict(title=f"{name}: activation and local gradient")]))\nfig.update_layout(title="Sigmoid: activation and local gradient", xaxis_title="pre-activation z", yaxis_title="value", hovermode="x unified", updatemenus=[dict(buttons=buttons, x=1.02, y=1.0)])\nif os.getenv("GITHUB_ACTIONS") == "true":\n    print("Interactive activation explorer built. Run locally to inspect saturation and local gradients.")\nelse:\n    fig.show(renderer="plotly_mimetype")\n"""

SOFTMAX = """import os\nimport numpy as np\nimport plotly.graph_objects as go\n\nlogits_demo = np.array([3.2, 1.7, 0.8, -0.4, 2.1])\ntemperatures = [0.25, 0.5, 1.0, 2.0, 4.0]\ndef softmax_t(logits, temperature):\n    scaled = logits / temperature; exp = np.exp(scaled - scaled.max()); return exp / exp.sum()\nprobabilities = [softmax_t(logits_demo, t) for t in temperatures]\nclasses = [f"class {i}" for i in range(len(logits_demo))]\nfig = go.Figure(data=[go.Bar(x=classes, y=probabilities[0], hovertemplate="%{x}<br>p=%{y:.4f}<extra></extra>")], frames=[go.Frame(name=str(i), data=[go.Bar(x=classes, y=p)]) for i, p in enumerate(probabilities)])\nfig.update_layout(title="Softmax temperature: same logits, different confidence", yaxis=dict(title="probability", range=[0, 1]), sliders=[dict(currentvalue=dict(prefix="temperature = "), steps=[dict(label=str(t), method="animate", args=[[str(i)], dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0))]) for i, t in enumerate(temperatures)])])\nif os.getenv("GITHUB_ACTIONS") == "true":\n    print("Interactive softmax-temperature explorer built. Run locally to move the confidence slider.")\nelse:\n    fig.show(renderer="plotly_mimetype")\n"""

OPTIMIZATION = """import os\nimport numpy as np\nimport plotly.graph_objects as go\n\nw_axis = np.linspace(-3, 4, 90); b_axis = np.linspace(-3, 2, 90); W, B = np.meshgrid(w_axis, b_axis); LOSS = (W - 1.5) ** 2 + 0.5 * (B + 0.5) ** 2\ndef path(lr, steps=18):\n    w, b, rows = -2.5, 1.8, []\n    for step in range(steps):\n        loss = (w - 1.5) ** 2 + 0.5 * (b + 0.5) ** 2; rows.append((w, b, loss, step)); w -= lr * 2 * (w - 1.5); b -= lr * (b + 0.5)\n    return rows\nfig = go.Figure(data=[go.Surface(x=W, y=B, z=LOSS, opacity=0.72, colorscale="Viridis", hovertemplate="w=%{x:.2f}<br>b=%{y:.2f}<br>loss=%{z:.3f}<extra></extra>")])\nfor lr in [0.05, 0.2, 0.8]:\n    rows = path(lr); fig.add_trace(go.Scatter3d(x=[r[0] for r in rows], y=[r[1] for r in rows], z=[r[2] for r in rows], mode="lines+markers", name=f"learning rate={lr}", text=[f"step={r[3]}" for r in rows], hovertemplate="%{text}<br>w=%{x:.3f}<br>b=%{y:.3f}<br>loss=%{z:.4f}<extra></extra>"))\nfig.update_layout(title="Interactive loss landscape + gradient-descent trajectories", scene=dict(xaxis_title="weight w", yaxis_title="bias b", zaxis_title="loss"))\nif os.getenv("GITHUB_ACTIONS") == "true":\n    print("Interactive 3D optimization landscape built. Run locally to rotate and compare learning rates.")\nelse:\n    fig.show(renderer="plotly_mimetype")\n"""

REPRESENTATION = """import os\nimport numpy as np\nimport plotly.graph_objects as go\n\nepochs = sorted(snaps); labels = np.asarray(y_train[:1500]); initial = np.asarray(snaps[epochs[0]])\ndef hover_for(coords, epoch):\n    return [f"sample={i}<br>true digit={int(labels[i])}<br>epoch={epoch}<br>h1={coords[i, 0]:.3f}<br>h2={coords[i, 1]:.3f}" for i in range(len(coords))]\nfig = go.Figure(data=[go.Scatter(x=initial[:, 0], y=initial[:, 1], mode="markers", marker=dict(color=labels, colorscale="Turbo", size=6, opacity=0.72, colorbar=dict(title="digit")), text=hover_for(initial, epochs[0]), hovertemplate="%{text}<extra></extra>")])\nfig.frames = tuple(go.Frame(name=str(epoch), data=[go.Scatter(x=np.asarray(snaps[epoch])[:, 0], y=np.asarray(snaps[epoch])[:, 1], mode="markers", marker=dict(color=labels, colorscale="Turbo", size=6, opacity=0.72), text=hover_for(np.asarray(snaps[epoch]), epoch), hovertemplate="%{text}<extra></extra>")]) for epoch in epochs)\nfig.update_layout(title="Representation learning: the same 1,500 MNIST samples move as weights learn", xaxis_title="hidden coordinate h1", yaxis_title="hidden coordinate h2", sliders=[dict(currentvalue=dict(prefix="epoch = "), steps=[dict(label=str(epoch), method="animate", args=[[str(epoch)], dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0))]) for epoch in epochs])], updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=700), fromcurrent=True)]), dict(label="Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])])\nif os.getenv("GITHUB_ACTIONS") == "true":\n    print("Interactive representation-learning animation built. Run locally to animate epochs and hover MNIST samples.")\nelse:\n    fig.show(renderer="plotly_mimetype")\n"""

CALIBRATION = """import os\nimport numpy as np\nimport plotly.graph_objects as go\nfrom plotly.subplots import make_subplots\n\npred = np.asarray(probs).argmax(axis=1); confidence = np.asarray(probs).max(axis=1); correct = (pred == np.asarray(y_test)).astype(float)\nedges = np.linspace(0, 1, 11); acc, conf, count = [], [], []\nfor left, right in zip(edges[:-1], edges[1:]):\n    mask = (confidence >= left) & (confidence < right if right < 1 else confidence <= right); count.append(int(mask.sum())); acc.append(float(correct[mask].mean()) if mask.any() else np.nan); conf.append(float(confidence[mask].mean()) if mask.any() else np.nan)\nfig = make_subplots(rows=1, cols=2, subplot_titles=("Reliability diagram", "Confidence distribution"))\nfig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration", line=dict(dash="dash")), row=1, col=1)\nfig.add_trace(go.Scatter(x=conf, y=acc, mode="lines+markers", name="model", text=[f"count={n}" for n in count], hovertemplate="mean confidence=%{x:.3f}<br>accuracy=%{y:.3f}<br>%{text}<extra></extra>"), row=1, col=1)\nfig.add_trace(go.Histogram(x=confidence, nbinsx=20, name="confidence"), row=1, col=2)\nfig.update_xaxes(title_text="predicted confidence", row=1, col=1); fig.update_yaxes(title_text="empirical accuracy", row=1, col=1); fig.update_xaxes(title_text="confidence", row=1, col=2); fig.update_yaxes(title_text="samples", row=1, col=2); fig.update_layout(title="Interactive calibration and confidence inspection")\nif os.getenv("GITHUB_ACTIONS") == "true":\n    print("Interactive calibration explorer built. Run locally to hover reliability bins and confidence counts.")\nelse:\n    fig.show(renderer="plotly_mimetype")\n"""

DRIFT = """import os\nimport numpy as np\nimport plotly.graph_objects as go\n\nbase_pixels = np.asarray(X_test[:300]).reshape(-1); severities = [0.0, 0.15, 0.30, 0.45, 0.60]; samples = [np.clip(base_pixels * (1 - s), 0, 1) for s in severities]\nfig = go.Figure(data=[go.Histogram(x=samples[0], nbinsx=40, histnorm="probability", hovertemplate="pixel≈%{x:.3f}<br>probability=%{y:.4f}<extra></extra>")], frames=[go.Frame(name=str(i), data=[go.Histogram(x=arr, nbinsx=40, histnorm="probability")]) for i, arr in enumerate(samples)])\nfig.update_layout(title="Interactive covariate-drift simulator: progressive image darkening", xaxis=dict(title="normalized pixel intensity", range=[0, 1]), yaxis_title="probability", sliders=[dict(currentvalue=dict(prefix="darkening severity = "), steps=[dict(label=f"{s:.2f}", method="animate", args=[[str(i)], dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0))]) for i, s in enumerate(severities)])])\nif os.getenv("GITHUB_ACTIONS") == "true":\n    print("Interactive drift simulator built. Run locally to move severity and inspect distribution shift.")\nelse:\n    fig.show(renderer="plotly_mimetype")\n"""


def main() -> None:
    ensure_dependencies()
    update_readme()
    append_lab(
        "03_neuron_linear_algebra_architecture.ipynb",
        "Interactive neuron geometry: weights and bias reshape the affine surface",
        "Rotate the 3D surface and move the parameter slider. The surface is the pre-activation $z=w_1x_1+w_2x_2+b$: weights change orientation while bias translates it. Hover exposes the exact numerical pre-activation at any input coordinate.",
        NEURON,
    )
    append_lab(
        "04_activations_initialization_gradient_flow.ipynb",
        "Interactive activation + derivative explorer",
        "Use the dropdown to compare activation values and local derivatives. Backpropagation multiplies by this local slope, so near-zero regions make saturation and vanishing gradients directly inspectable.",
        ACTIVATIONS,
    )
    append_lab(
        "05_forward_logits_softmax.ipynb",
        "Interactive softmax confidence and temperature",
        "The logits stay fixed while temperature changes. Class ranking can remain unchanged while probability sharpness changes substantially, which matters for confidence interpretation, calibration and decision thresholds.",
        SOFTMAX,
    )
    append_lab(
        "07_optimizers_end_to_end_training.ipynb",
        "Interactive 3D loss landscape and optimizer trajectories",
        "Rotate the surface, hover exact loss values and compare learning rates. This deliberately low-dimensional landscape exposes how step size changes convergence, overshoot and oscillation.",
        OPTIMIZATION,
    )
    append_lab(
        "09_representation_learning_visualized.ipynb",
        "Interactive representation learning across epochs",
        "This uses the notebook's real `snaps` dictionary: the **same 1,500 MNIST samples** at multiple epochs. Play the animation or move the epoch slider, then hover one sample to track how learned hidden coordinates change.",
        REPRESENTATION,
    )
    append_lab(
        "10_test_evaluation_error_calibration.ipynb",
        "Interactive calibration and confidence inspection",
        "This uses the notebook's actual test-set `probs` and `y_test`. Hover the reliability curve to compare mean predicted confidence with empirical accuracy and see how many samples support each bin.",
        CALIBRATION,
    )
    append_lab(
        "12_production_monitoring_drift_retraining_business.ipynb",
        "Interactive production drift simulator",
        "The slider progressively darkens the notebook's real normalized MNIST test pixels. Labels do not change, but the input distribution does, making covariate shift tangible before connecting it to monitoring, delayed-label performance and retraining governance.",
        DRIFT,
    )


if __name__ == "__main__":
    main()
