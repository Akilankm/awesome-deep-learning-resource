from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == 'tools' else Path.cwd()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip() + "\n",
    }


def write_nb(path: Path, title: str, learning: str, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    intro = md(f"""
    # {title}

    **Learning objective:** {learning}

    This notebook is part of the TensorFlow/Keras learning track. It is designed to be read top-to-bottom: intuition → shapes → mathematics → TensorFlow implementation → observed result → interpretation.

    > GitHub renders the committed executed output as a static learning artifact. Clone the repository and rerun it in Jupyter/VS Code for live experimentation.
    """)
    nb = {
        "cells": [intro] + cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


SETUP = r'''
import os, warnings, random
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

print("TensorFlow:", tf.__version__)
print("Keras:", tf.keras.__version__ if hasattr(tf.keras, "__version__") else "bundled with TensorFlow")
print("Execution device(s):", [d.device_type for d in tf.config.list_logical_devices()])
'''


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def build_shared_files() -> None:
    env_common = """
    channels:
      - conda-forge
    dependencies:
      - python=3.11
      - pip=24.3
      - numpy=2.0
      - pandas=2.2
      - scipy=1.14
      - scikit-learn=1.6
      - matplotlib=3.9
      - plotly=5.24
      - ipywidgets=8.1
      - jupyterlab=4.3
      - notebook=7.3
      - ipykernel=6.29
      - nbformat=5.10
      - nbclient=0.10
      - nbconvert=7.16
      - pyarrow=18
      - pyyaml=6.0
      - tqdm=4.67
      - pillow=11
      - pip:
          - tensorflow==2.18.1
    """
    write_text(ROOT / "RNN/environment.yml", "name: awesome-rnn\n\n" + textwrap.dedent(env_common))
    write_text(ROOT / "CNN/environment.yml", "name: awesome-cnn\n\n" + textwrap.dedent(env_common))

    write_text(ROOT / "RNN/README.md", """
    # Recurrent Neural Networks — TensorFlow/Keras

    A complete, executed learning path from sequence tensors and recurrence to SimpleRNN, LSTM, GRU, forecasting, inference and monitoring.

    ## Run
    ```bash
    conda env create -f RNN/environment.yml
    conda activate awesome-rnn
    jupyter lab
    ```

    Start with `notebooks/00_rnn_learning_map.ipynb`, progress numerically, then complete `projects/jena_climate_forecasting/00_end_to_end_jena_forecasting.ipynb`.

    Every committed notebook is executed by CI. Static outputs render on GitHub; rerunning locally enables experimentation.
    """)
    write_text(ROOT / "CNN/README.md", """
    # Convolutional Neural Networks — TensorFlow/Keras

    A complete, executed learning path from image tensors and manual convolution to Conv2D, pooling, regularization, transfer learning, Grad-CAM, evaluation, inference and monitoring.

    ## Run
    ```bash
    conda env create -f CNN/environment.yml
    conda activate awesome-cnn
    jupyter lab
    ```

    Start with `notebooks/00_cnn_learning_map.ipynb`, progress numerically, then complete `projects/fashion_mnist_classifier/00_end_to_end_fashion_mnist.ipynb`.

    Every committed notebook is executed by CI. Static outputs render on GitHub; rerunning locally enables experimentation.
    """)

    write_text(ROOT / "RNN/data/DATASET_CARD.md", """
    # Dataset card — Jena Climate

    The flagship RNN project uses the Jena Climate time-series dataset distributed through TensorFlow/Keras educational examples. The raw observations were recorded at the Max Planck Institute for Biogeochemistry weather station in Jena, Germany at 10-minute intervals.

    The project downloads the canonical compressed CSV with `tf.keras.utils.get_file`, downsamples it to hourly resolution for a CPU-friendly educational experiment, and predicts future temperature from recent multivariate weather history.

    **Important split contract:** all train/validation/test boundaries are chronological. Normalization statistics are learned on training data only.
    """)
    write_text(ROOT / "CNN/data/DATASET_CARD.md", """
    # Dataset card — Fashion-MNIST

    The flagship CNN project uses Fashion-MNIST: 28×28 grayscale images of 10 apparel categories. TensorFlow exposes the canonical dataset through `tf.keras.datasets.fashion_mnist.load_data()`.

    The original training partition is deterministically split into training and validation subsets. The canonical test partition remains untouched until final evaluation.

    Categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
    """)

    write_text(ROOT / "RNN/notebooks/README.md", """
    # RNN notebook sequence
    Run notebooks numerically. Each notebook teaches one conceptual layer and leaves rendered outputs in the committed `.ipynb` file.
    """)
    write_text(ROOT / "CNN/notebooks/README.md", """
    # CNN notebook sequence
    Run notebooks numerically. Each notebook teaches one conceptual layer and leaves rendered outputs in the committed `.ipynb` file.
    """)

    write_text(ROOT / "RNN/projects/jena_climate_forecasting/README.md", """
    # End-to-end Jena climate forecasting
    Raw multivariate time-series data → quality checks → chronological split → train-only normalization → windowing → naive baseline → SimpleRNN/LSTM/GRU → validation selection → untouched test evaluation → saved Keras model → reload/inference → drift checks.
    """)
    write_text(ROOT / "CNN/projects/fashion_mnist_classifier/README.md", """
    # End-to-end Fashion-MNIST classification
    Raw image data → EDA → deterministic train/validation/test contract → normalization → dense baseline → CNN → validation selection → untouched test evaluation → confusion/error analysis → calibration → saved Keras model → reload/inference → drift simulation.
    """)

    write_text(ROOT / "RNN/projects/jena_climate_forecasting/configs/default.json", json.dumps({
        "seed": 42, "hourly_rows": 8000, "lookback_hours": 24, "batch_size": 64,
        "epochs": 2, "features": ["T (degC)", "p (mbar)", "rho (g/m**3)"]
    }, indent=2))
    write_text(ROOT / "CNN/projects/fashion_mnist_classifier/configs/default.json", json.dumps({
        "seed": 42, "train_samples": 12000, "validation_samples": 2000,
        "test_samples": 3000, "batch_size": 64, "epochs": 2
    }, indent=2))

    write_text(ROOT / "RNN/projects/jena_climate_forecasting/src/models.py", """
    import tensorflow as tf

    def build_model(kind: str, input_shape):
        layer_map = {
            "SimpleRNN": tf.keras.layers.SimpleRNN,
            "LSTM": tf.keras.layers.LSTM,
            "GRU": tf.keras.layers.GRU,
        }
        if kind not in layer_map:
            raise ValueError(f"Unknown model kind: {kind}")
        inputs = tf.keras.Input(shape=input_shape)
        x = layer_map[kind](16)(inputs)
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs, outputs, name=kind.lower())
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model
    """)
    write_text(ROOT / "RNN/projects/jena_climate_forecasting/src/__init__.py", "")
    write_text(ROOT / "CNN/projects/fashion_mnist_classifier/src/models.py", """
    import tensorflow as tf

    def build_mlp():
        model = tf.keras.Sequential([
            tf.keras.layers.Input((28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ], name="mlp_baseline")
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def build_cnn():
        model = tf.keras.Sequential([
            tf.keras.layers.Input((28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ], name="cnn")
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
    """)
    write_text(ROOT / "CNN/projects/fashion_mnist_classifier/src/__init__.py", "")

    write_text(ROOT / "scripts/execute_notebooks.py", r'''
    from __future__ import annotations
    import argparse, json
    from pathlib import Path
    import nbformat
    from nbclient import NotebookClient

    def execute(path: Path, repo_root: Path) -> dict:
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=1200, kernel_name="python3", allow_errors=False,
                                resources={"metadata": {"path": str(repo_root)}})
        client.execute()
        nbformat.write(nb, path)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        return {
            "notebook": str(path.relative_to(repo_root)),
            "code_cells": len(code_cells),
            "executed": sum(c.execution_count is not None for c in code_cells),
            "output_cells": sum(bool(c.get("outputs")) for c in code_cells),
        }

    def main():
        p = argparse.ArgumentParser()
        p.add_argument("paths", nargs="+")
        p.add_argument("--report", required=True)
        args = p.parse_args()
        root = Path.cwd().resolve()
        files = []
        for raw in args.paths:
            path = (root / raw).resolve()
            if path.is_dir():
                files.extend(sorted(path.rglob("*.ipynb")))
            elif path.suffix == ".ipynb":
                files.append(path)
        report = []
        for f in files:
            print(f"EXECUTING {f.relative_to(root)}", flush=True)
            report.append(execute(f, root))
        report_path = root / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))

    if __name__ == "__main__":
        main()
    ''')

    write_text(ROOT / "scripts/verify_notebooks.py", r'''
    from __future__ import annotations
    import json, sys
    from pathlib import Path
    import nbformat

    roots = [Path("RNN/notebooks"), Path("RNN/projects"), Path("CNN/notebooks"), Path("CNN/projects")]
    failures = []
    summary = []
    for root in roots:
        for path in sorted(root.rglob("*.ipynb")):
            nb = nbformat.read(path, as_version=4)
            code_cells = [c for c in nb.cells if c.cell_type == "code"]
            unexecuted = [i for i, c in enumerate(code_cells) if c.execution_count is None]
            errors = [o for c in code_cells for o in c.get("outputs", []) if o.output_type == "error"]
            output_count = sum(bool(c.get("outputs")) for c in code_cells)
            ok = bool(code_cells) and not unexecuted and not errors and output_count > 0
            summary.append({"notebook": str(path), "code_cells": len(code_cells), "output_cells": output_count, "ok": ok})
            if not ok:
                failures.append({"path": str(path), "unexecuted": unexecuted, "errors": len(errors), "outputs": output_count})
    Path("notebook_execution_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        print("FAILURES", json.dumps(failures, indent=2), file=sys.stderr)
        raise SystemExit(1)
    ''')


def rnn_notebooks() -> None:
    N = ROOT / "RNN/notebooks"

    write_nb(N / "00_rnn_learning_map.ipynb", "RNN learning map", "Understand where recurrent models fit, the learning sequence, and the tensor conventions used throughout the track.", [
        md("""## Why recurrent networks exist
        Feed-forward networks treat examples as independent. Sequence problems are different: order carries information. RNNs introduce a **state** that is updated one timestep at a time, creating a learnable summary of the past.

        We will move from `SimpleRNN` to LSTM and GRU, then connect them to forecasting, evaluation, inference and monitoring."""),
        code(SETUP),
        code(r'''
        tasks = pd.DataFrame({
            "problem": ["sensor forecasting", "sentiment", "sequence labeling", "sequence generation"],
            "input": ["many timesteps", "many tokens", "many tokens", "prefix"],
            "output": ["future value", "one label", "label per token", "next token(s)"],
            "shape pattern": ["many→one", "many→one", "many→many", "many→many"],
        })
        display(tasks)
        '''),
        code(r'''
        t = np.arange(40)
        x = np.sin(t / 5) + 0.03 * t
        plt.figure(figsize=(9,3)); plt.plot(t, x, marker="o", ms=3); plt.title("A sequence: each value has context"); plt.xlabel("time"); plt.ylabel("signal"); plt.grid(alpha=.25); plt.show()
        '''),
        md("""## Track contract
        Every later notebook uses `(batch, time, features)` for sequence tensors. The flagship project uses chronological splitting and training-only normalization to prevent leakage.""")
    ])

    write_nb(N / "01_sequence_data_and_shapes.ipynb", "Sequence data and tensor shapes", "Read, construct and reason about `(batch, time, features)` tensors before building recurrent layers.", [
        code(SETUP),
        md("""## The three axes
        A dense tabular batch is commonly `(batch, features)`. A sequence adds time: `(batch, time, features)`. A batch of 4 sensor histories, each 6 hours long with 3 measurements per hour, therefore has shape `(4, 6, 3)`."""),
        code(r'''
        x = np.arange(4*6*3, dtype=np.float32).reshape(4,6,3)
        print("shape:", x.shape)
        print("first example:\n", x[0])
        print("timestep 2 across batch:\n", x[:,2,:])
        '''),
        code(r'''
        fig, axes = plt.subplots(1,3, figsize=(11,3))
        for feature, ax in enumerate(axes):
            ax.plot(x[0,:,feature], marker="o"); ax.set_title(f"feature {feature}"); ax.set_xlabel("time")
        plt.tight_layout(); plt.show()
        '''),
        md("""**Interpretation:** time is not interchangeable with the feature axis. Recurrent layers move along the time axis while observing the feature vector at each step.""")
    ])

    write_nb(N / "02_time_series_windowing.ipynb", "Windowing, chronological splits and leakage", "Convert one long time series into supervised examples and understand why time-series splits must preserve temporal order.", [
        code(SETUP),
        code(r'''
        n = 120
        t = np.arange(n)
        series = np.sin(t/8) + 0.005*t
        lookback = 12
        X, y = [], []
        for i in range(len(series)-lookback):
            X.append(series[i:i+lookback]); y.append(series[i+lookback])
        X = np.asarray(X)[...,None]; y = np.asarray(y)
        print("X:", X.shape, "y:", y.shape)
        print("first window:", np.round(X[0,:,0],3)); print("first target:", round(float(y[0]),3))
        '''),
        code(r'''
        train_end = int(len(X)*0.7); val_end = int(len(X)*0.85)
        splits = pd.DataFrame({"split":["train","validation","test"], "start":[0,train_end,val_end], "end":[train_end,val_end,len(X)]})
        display(splits)
        plt.figure(figsize=(10,3)); plt.plot(series); plt.axvline(train_end+lookback, ls="--"); plt.axvline(val_end+lookback, ls="--"); plt.title("Chronological boundaries"); plt.show()
        '''),
        md("""Random shuffling before splitting can leak future regimes into training. We may shuffle **training windows during optimization**, but the dataset partitions themselves remain chronological.""")
    ])

    write_nb(N / "03_understanding_recurrence.ipynb", "Recurrence from first principles", "Compute the hidden-state recurrence manually and connect the equation to a TensorFlow `SimpleRNNCell`.", [
        code(SETUP),
        md(r"""For a vanilla RNN, one common update is
        \[h_t = \tanh(x_tW_x + h_{t-1}W_h + b).\]
        The same weights are reused at every timestep. This **weight sharing through time** is the defining recurrence."""),
        code(r'''
        xs = np.array([[1.0],[0.5],[-0.25]], dtype=np.float32)
        Wx = np.array([[0.8]], dtype=np.float32); Wh = np.array([[0.4]], dtype=np.float32); b = np.array([0.1], dtype=np.float32)
        h = np.array([0.0], dtype=np.float32); states=[]
        for step, xt in enumerate(xs):
            h = np.tanh(xt @ Wx + h @ Wh + b)
            states.append(float(h[0])); print(f"t={step}: x={xt[0]: .2f} -> h={h[0]: .4f}")
        '''),
        code(r'''
        cell = tf.keras.layers.SimpleRNNCell(1, activation="tanh")
        _ = cell(tf.constant([[0.0]]), [tf.constant([[0.0]])])
        cell.set_weights([Wx, Wh, b])
        h_tf = tf.zeros((1,1))
        tf_states=[]
        for xt in xs:
            out, [h_tf] = cell(tf.constant(xt.reshape(1,1)), [h_tf]); tf_states.append(float(h_tf.numpy()[0,0]))
        print("NumPy states:", np.round(states,6)); print("TensorFlow states:", np.round(tf_states,6)); print("match:", np.allclose(states, tf_states))
        '''),
        md("""The match demonstrates that Keras is applying the same recurrence we computed explicitly; the framework automates tensor operations and differentiation, not the underlying idea.""")
    ])

    write_nb(N / "04_simple_rnn_tensorflow.ipynb", "Your first TensorFlow SimpleRNN", "Build, train and inspect a minimal Keras recurrent forecasting model.", [
        code(SETUP),
        code(r'''
        t = np.arange(900); s = np.sin(t/18) + 0.2*np.sin(t/5)
        lookback=24; X=[]; y=[]
        for i in range(len(s)-lookback): X.append(s[i:i+lookback]); y.append(s[i+lookback])
        X=np.asarray(X,dtype="float32")[...,None]; y=np.asarray(y,dtype="float32")
        cut=700
        model=tf.keras.Sequential([tf.keras.layers.Input((lookback,1)), tf.keras.layers.SimpleRNN(12), tf.keras.layers.Dense(1)])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.summary()
        hist=model.fit(X[:cut],y[:cut],validation_data=(X[cut:],y[cut:]),epochs=3,batch_size=32,verbose=0)
        display(pd.DataFrame(hist.history).round(4))
        '''),
        code(r'''
        pred=model.predict(X[cut:cut+120],verbose=0).ravel()
        plt.figure(figsize=(10,3)); plt.plot(y[cut:cut+120],label="actual"); plt.plot(pred,label="prediction"); plt.legend(); plt.title("SimpleRNN forecast"); plt.show()
        '''),
        md("""A recurrent layer converts the entire history window into a learned representation. The Dense layer maps that representation to the numeric forecast.""")
    ])

    write_nb(N / "05_hidden_state_and_unrolling.ipynb", "Hidden state, unrolling and `return_sequences`", "Inspect per-timestep outputs and final state to understand what Keras returns from recurrent layers.", [
        code(SETUP),
        code(r'''
        x=tf.constant(np.arange(2*5*3).reshape(2,5,3)/10, dtype=tf.float32)
        layer=tf.keras.layers.SimpleRNN(4, return_sequences=True, return_state=True)
        sequence_output, final_state=layer(x)
        print("input:",x.shape); print("all hidden states:",sequence_output.shape); print("final state:",final_state.shape)
        print("last sequence state equals final state:", np.allclose(sequence_output[:,-1,:].numpy(), final_state.numpy()))
        '''),
        code(r'''
        norms=tf.norm(sequence_output[0],axis=1).numpy()
        plt.figure(figsize=(7,3)); plt.plot(norms,marker="o"); plt.xlabel("timestep"); plt.ylabel("hidden-state L2 norm"); plt.title("State evolves through time"); plt.show()
        '''),
        md("""Use `return_sequences=True` when downstream layers need every timestep (stacked RNNs, sequence labeling, attention-like processing). Use the final representation for many-to-one tasks.""")
    ])

    write_nb(N / "06_backpropagation_through_time.ipynb", "Backpropagation through time (BPTT)", "Use TensorFlow GradientTape to see how a final loss sends gradients through the unrolled recurrent computation.", [
        code(SETUP),
        md("""Training unfolds recurrence conceptually into a deep graph whose depth is the number of timesteps. The same recurrent matrix participates repeatedly, so its gradient accumulates contributions from many temporal paths."""),
        code(r'''
        rnn=tf.keras.layers.SimpleRNN(3); dense=tf.keras.layers.Dense(1)
        x=tf.random.normal((8,12,2),seed=SEED); y=tf.random.normal((8,1),seed=SEED+1)
        with tf.GradientTape() as tape:
            h=rnn(x); pred=dense(h); loss=tf.reduce_mean(tf.square(pred-y))
        vars_=rnn.trainable_variables+dense.trainable_variables
        grads=tape.gradient(loss,vars_)
        table=pd.DataFrame({"variable":[v.name for v in vars_],"shape":[str(tuple(v.shape)) for v in vars_],"gradient_norm":[float(tf.norm(g)) for g in grads]})
        print("loss:",float(loss)); display(table)
        '''),
        code(r'''
        plt.figure(figsize=(8,3)); plt.bar(np.arange(len(grads)), [float(tf.norm(g)) for g in grads]); plt.xticks(np.arange(len(grads)), range(len(grads))); plt.ylabel("gradient norm"); plt.title("Gradient signal reaching trainable variables"); plt.show()
        '''),
        md("""BPTT is ordinary reverse-mode automatic differentiation applied to the unrolled recurrent graph. TensorFlow builds and differentiates that graph for us.""")
    ])

    write_nb(N / "07_vanishing_and_exploding_gradients.ipynb", "Vanishing and exploding gradients", "See why repeatedly multiplying temporal Jacobians can shrink or amplify gradients exponentially.", [
        code(SETUP),
        code(r'''
        steps=np.arange(1,41)
        curves={"0.7 (vanish)":0.7**steps,"1.0 (stable)":1.0**steps,"1.3 (explode)":1.3**steps}
        df=pd.DataFrame(curves,index=steps); display(df.iloc[[0,4,9,19,39]])
        plt.figure(figsize=(9,4))
        for name,v in curves.items(): plt.semilogy(steps,np.abs(v),label=name)
        plt.xlabel("repeated temporal multiplications"); plt.ylabel("magnitude (log scale)"); plt.legend(); plt.title("Why long gradient paths are difficult"); plt.show()
        '''),
        md("""LSTM and GRU architectures add gated paths that make retaining or forgetting information easier to learn. Gradient clipping is another practical safeguard against explosions, but it does not solve memory design by itself."""),
        code(r'''
        opt=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
        print("Example optimizer with global per-variable norm clipping:", opt.get_config()["clipnorm"])
        ''')
    ])

    write_nb(N / "08_lstm_intuition_and_tensorflow.ipynb", "LSTM: controlled memory", "Understand the cell state and gates, then inspect TensorFlow's LSTM shapes and parameterization.", [
        code(SETUP),
        md(r"""An LSTM learns four transformations: forget gate, input gate, candidate content and output gate. The cell state provides a controlled memory path:
        \[c_t=f_t\odot c_{t-1}+i_t\odot\tilde{c}_t,\quad h_t=o_t\odot\tanh(c_t).\]"""),
        code(r'''
        sigmoid=lambda z:1/(1+np.exp(-z))
        c_prev=np.array([0.8,-0.4]); f=sigmoid(np.array([2.0,-1.0])); i=sigmoid(np.array([-0.5,1.5])); candidate=np.tanh(np.array([0.3,0.9])); o=sigmoid(np.array([1.0,0.2]))
        c=f*c_prev+i*candidate; h=o*np.tanh(c)
        display(pd.DataFrame({"c_prev":c_prev,"forget":f,"input":i,"candidate":candidate,"c_new":c,"output":o,"h_new":h}).round(4))
        '''),
        code(r'''
        x=tf.random.normal((4,10,3),seed=SEED)
        lstm=tf.keras.layers.LSTM(6,return_sequences=True,return_state=True)
        seq,h,c=lstm(x)
        print("sequence output:",seq.shape,"hidden state:",h.shape,"cell state:",c.shape)
        print("parameters:",lstm.count_params())
        '''),
        md("""Compared with a vanilla RNN, an LSTM carries both `h` (exposed hidden representation) and `c` (cell memory).""")
    ])

    write_nb(N / "09_gru_intuition_and_tensorflow.ipynb", "GRU: a compact gated recurrent unit", "Understand the GRU's update/reset gates and compare its parameter count with an LSTM.", [
        code(SETUP),
        code(r'''
        input_dim=5; units=8
        rnn=tf.keras.layers.SimpleRNN(units); lstm=tf.keras.layers.LSTM(units); gru=tf.keras.layers.GRU(units)
        dummy=tf.zeros((2,12,input_dim)); _=rnn(dummy); _=lstm(dummy); _=gru(dummy)
        display(pd.DataFrame({"layer":["SimpleRNN","LSTM","GRU"],"parameters":[rnn.count_params(),lstm.count_params(),gru.count_params()]}))
        '''),
        code(r'''
        labels=["SimpleRNN","LSTM","GRU"]; vals=[rnn.count_params(),lstm.count_params(),gru.count_params()]
        plt.figure(figsize=(6,3)); plt.bar(labels,vals); plt.ylabel("trainable parameters"); plt.title("Same input size and hidden width"); plt.show()
        '''),
        md("""GRU merges some of the LSTM's gating machinery and has no separate cell state. Neither architecture is universally superior: validation performance, latency and memory determine the choice.""")
    ])

    write_nb(N / "10_sequence_architectures.ipynb", "Sequence architecture patterns", "Map many-to-one and many-to-many tasks to TensorFlow model shapes.", [
        code(SETUP),
        code(r'''
        inp=tf.keras.Input((7,4))
        many_to_one=tf.keras.Model(inp, tf.keras.layers.Dense(2)(tf.keras.layers.GRU(6)(inp)), name="many_to_one")
        inp2=tf.keras.Input((7,4))
        many_to_many=tf.keras.Model(inp2, tf.keras.layers.Dense(2)(tf.keras.layers.GRU(6,return_sequences=True)(inp2)), name="many_to_many")
        x=tf.zeros((3,7,4))
        print("many→one:",many_to_one(x).shape); print("many→many:",many_to_many(x).shape)
        '''),
        code(r'''
        patterns=pd.DataFrame({"pattern":["many→one","many→many (aligned)","encoder→decoder"],"example":["sentiment / forecast","tag every timestep","translation / generation"],"typical RNN output":["final state","return_sequences=True","state passed to decoder"]})
        display(patterns)
        '''),
        md("""Architecture follows the output contract. Always write the desired input and output tensor shapes before selecting recurrent-layer options.""")
    ])

    write_nb(N / "11_stacked_bidirectional_rnn.ipynb", "Stacked and bidirectional recurrence", "Build deeper and bidirectional recurrent networks and understand when they are valid.", [
        code(SETUP),
        code(r'''
        stacked=tf.keras.Sequential([
            tf.keras.layers.Input((20,3)),
            tf.keras.layers.GRU(12,return_sequences=True),
            tf.keras.layers.GRU(8),
            tf.keras.layers.Dense(1)
        ],name="stacked_gru")
        bidir=tf.keras.Sequential([
            tf.keras.layers.Input((20,3)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
            tf.keras.layers.Dense(1)
        ],name="bidirectional_lstm")
        print("stacked output:",stacked(tf.zeros((4,20,3))).shape,"parameters:",stacked.count_params())
        print("bidirectional output:",bidir(tf.zeros((4,20,3))).shape,"parameters:",bidir.count_params())
        '''),
        md("""**Critical causality rule:** bidirectional recurrence uses future context. It is excellent when the entire sequence is available at inference (e.g., document classification) but invalid for causal forecasting where future timesteps do not yet exist."""),
        code(r'''
        display(pd.DataFrame({"use case":["offline text classification","causal sensor forecast"],"bidirectional valid?":[True,False],"reason":["whole sequence known","future context would leak"]}))
        ''')
    ])

    write_nb(N / "12_regularization_and_training.ipynb", "Regularization and reliable RNN training", "Use dropout, early stopping, checkpoints and gradient clipping as part of a controlled training process.", [
        code(SETUP),
        code(r'''
        model=tf.keras.Sequential([tf.keras.layers.Input((24,2)),tf.keras.layers.GRU(16,dropout=0.2),tf.keras.layers.Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),loss="mse",metrics=["mae"])
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True)]
        model.summary(); print("callbacks:",[type(c).__name__ for c in callbacks]); print("clipnorm:",model.optimizer.clipnorm)
        '''),
        code(r'''
        guidance=pd.DataFrame({"tool":["dropout","early stopping","gradient clipping","checkpoint"],"controls":["co-adaptation","over-training","exploding gradients","model reproducibility"]})
        display(guidance)
        '''),
        md("""Regularization is not a substitute for correct validation. Hyperparameters and stopping decisions belong to validation data; the final test set is not a tuning surface.""")
    ])

    write_nb(N / "13_forecasting_evaluation.ipynb", "Forecasting baselines and evaluation", "Compute naive baselines, MAE, RMSE and residual diagnostics before trusting a recurrent model.", [
        code(SETUP),
        code(r'''
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        t=np.arange(240); y=np.sin(t/15)+0.002*t
        actual=y[1:]; naive=y[:-1]; biased=naive+0.1
        rows=[]
        for name,p in [("last-value baseline",naive),("biased forecast",biased)]:
            rows.append({"model":name,"MAE":mean_absolute_error(actual,p),"RMSE":mean_squared_error(actual,p)**0.5})
        display(pd.DataFrame(rows).round(4))
        '''),
        code(r'''
        resid=actual-naive
        fig,axes=plt.subplots(1,2,figsize=(10,3)); axes[0].plot(resid); axes[0].axhline(0,c="k",lw=1); axes[0].set_title("residual over time"); axes[1].hist(resid,bins=20); axes[1].set_title("residual distribution"); plt.tight_layout(); plt.show()
        '''),
        md("""A neural forecast should beat a meaningful baseline. Aggregate metrics should be complemented by error slices over time/regime so systematic failure is visible.""")
    ])

    write_nb(N / "14_rnn_inference_and_monitoring.ipynb", "RNN serialization, inference and monitoring", "Save/reload a recurrent model and quantify a simple feature-drift signal.", [
        code(SETUP),
        code(r'''
        model=tf.keras.Sequential([tf.keras.layers.Input((6,1)),tf.keras.layers.GRU(4),tf.keras.layers.Dense(1)])
        model.compile(optimizer="adam",loss="mse")
        sample=tf.random.normal((1,6,1),seed=SEED)
        _=model(sample)
        path=Path("RNN")/"artifacts_demo_rnn.keras"; model.save(path)
        restored=tf.keras.models.load_model(path)
        a=model.predict(sample,verbose=0); b=restored.predict(sample,verbose=0)
        print("saved:",path,"reload prediction identical:",np.allclose(a,b))
        '''),
        code(r'''
        rng=np.random.default_rng(SEED); reference=rng.normal(0,1,1000); production=rng.normal(0.7,1.2,1000)
        drift=(production.mean()-reference.mean())/reference.std()
        print("reference mean/std:",reference.mean().round(3),reference.std().round(3)); print("production mean/std:",production.mean().round(3),production.std().round(3)); print("standardized mean shift:",round(float(drift),3))
        '''),
        md("""Production monitoring needs both data signals (missingness, distribution shift, sequence length) and outcome signals (error when labels arrive). A saved model without an input contract and monitoring plan is not an operational system.""")
    ])


def rnn_project() -> None:
    P = ROOT / "RNN/projects/jena_climate_forecasting"
    write_nb(P / "00_end_to_end_jena_forecasting.ipynb", "Project: end-to-end Jena climate forecasting", "Take real multivariate weather observations from acquisition through validation, recurrent-model comparison, final test evaluation, saved-model inference and drift checks.", [
        code(SETUP + r'''
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json, zipfile, time
PROJECT=Path("RNN/projects/jena_climate_forecasting")
(PROJECT/"artifacts").mkdir(parents=True,exist_ok=True)
(PROJECT/"reports").mkdir(parents=True,exist_ok=True)
        '''),
        md("""## 1. Problem and data contract
        **Goal:** predict temperature one hour ahead using the previous 24 hourly observations of temperature, pressure and air density.

        We use real Jena weather-station observations. The original 10-minute stream is downsampled to hourly resolution. Splits are chronological and normalization statistics come from training data only."""),
        code(r'''
        url="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
        zip_path=tf.keras.utils.get_file("jena_climate_2009_2016.csv.zip",origin=url,extract=False)
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open("jena_climate_2009_2016.csv") as f: raw=pd.read_csv(f)
        print("raw shape:",raw.shape); display(raw.head(3)); print("missing cells:",int(raw.isna().sum().sum()))
        '''),
        md("""## 2. EDA and hourly sampling
        The timestamp column is parsed, observations are sampled every sixth row (10-minute → hourly), and a compact 8,000-hour teaching slice keeps execution CPU-friendly while preserving real temporal structure."""),
        code(r'''
        df=raw.iloc[::6].copy().iloc[:8000]
        df["Date Time"]=pd.to_datetime(df["Date Time"],format="%d.%m.%Y %H:%M:%S")
        features=["T (degC)","p (mbar)","rho (g/m**3)"]
        display(df[features].describe().T.round(3))
        fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True)
        for col,ax in zip(features,axes): ax.plot(df["Date Time"],df[col],lw=.8); ax.set_ylabel(col)
        plt.tight_layout(); plt.show()
        '''),
        md("""## 3. Chronological split and leakage-safe normalization
        70% train, 15% validation, 15% test. Mean/std are fitted on the training partition and then reused unchanged."""),
        code(r'''
        values=df[features].to_numpy(dtype="float32")
        n=len(values); train_end=int(n*.70); val_end=int(n*.85)
        mean=values[:train_end].mean(axis=0); std=values[:train_end].std(axis=0)
        norm=(values-mean)/std
        print("boundaries:",train_end,val_end,n); print("train mean after scaling:",np.round(norm[:train_end].mean(axis=0),4))
        '''),
        md("""## 4. Windowing
        A sample is 24 consecutive hourly vectors; the label is next-hour normalized temperature. Windows are assigned to a split according to the target timestamp, so no test target appears in training."""),
        code(r'''
        lookback=24; target_col=0
        X=[]; y=[]; target_idx=[]
        for i in range(lookback,len(norm)):
            X.append(norm[i-lookback:i]); y.append(norm[i,target_col]); target_idx.append(i)
        X=np.asarray(X,dtype="float32"); y=np.asarray(y,dtype="float32"); target_idx=np.asarray(target_idx)
        tr=target_idx<train_end; va=(target_idx>=train_end)&(target_idx<val_end); te=target_idx>=val_end
        Xtr,ytr=X[tr],y[tr]; Xva,yva=X[va],y[va]; Xte,yte=X[te],y[te]
        print("train/val/test:",Xtr.shape,Xva.shape,Xte.shape)
        '''),
        md("""## 5. Baseline first
        The last observed temperature is a strong one-hour baseline. Neural models need to improve on it to justify their complexity."""),
        code(r'''
        def inv_temp(z): return z*std[0]+mean[0]
        yte_real=inv_temp(yte); naive_real=inv_temp(Xte[:,-1,0])
        baseline={"MAE":mean_absolute_error(yte_real,naive_real),"RMSE":mean_squared_error(yte_real,naive_real)**0.5}
        display(pd.DataFrame([{"model":"last value",**baseline}]).round(4))
        '''),
        md("""## 6. Train SimpleRNN, LSTM and GRU under the same contract
        Architecture width, optimizer, epochs and input windows are intentionally aligned so the comparison isolates the recurrent cell choice."""),
        code(r'''
        def build(kind):
            cls={"SimpleRNN":tf.keras.layers.SimpleRNN,"LSTM":tf.keras.layers.LSTM,"GRU":tf.keras.layers.GRU}[kind]
            m=tf.keras.Sequential([tf.keras.layers.Input((lookback,len(features))),cls(16),tf.keras.layers.Dense(1)],name=kind.lower())
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-3,clipnorm=1.0),loss="mse",metrics=["mae"]); return m
        histories={}; models={}; rows=[]
        for kind in ["SimpleRNN","LSTM","GRU"]:
            tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
            m=build(kind)
            h=m.fit(Xtr,ytr,validation_data=(Xva,yva),epochs=2,batch_size=64,verbose=0)
            histories[kind]=h.history; models[kind]=m
            rows.append({"model":kind,"best_val_mae_norm":min(h.history["val_mae"]),"parameters":m.count_params()})
        comparison=pd.DataFrame(rows).sort_values("best_val_mae_norm"); display(comparison.round(4))
        '''),
        code(r'''
        plt.figure(figsize=(8,4))
        for kind,h in histories.items(): plt.plot(h["val_mae"],marker="o",label=kind)
        plt.xlabel("epoch"); plt.ylabel("validation MAE (normalized)"); plt.xticks(range(2),[1,2]); plt.legend(); plt.title("Model selection uses validation only"); plt.show()
        '''),
        md("""## 7. Final untouched test evaluation
        We choose the model with the lowest validation MAE, then evaluate that model on test exactly once for the final headline result."""),
        code(r'''
        best=comparison.iloc[0]["model"]; best_model=models[best]
        pred_norm=best_model.predict(Xte,batch_size=128,verbose=0).ravel(); pred=inv_temp(pred_norm)
        test_mae=mean_absolute_error(yte_real,pred); test_rmse=mean_squared_error(yte_real,pred)**0.5
        metrics={"selected_model":best,"baseline_mae":float(baseline["MAE"]),"baseline_rmse":float(baseline["RMSE"]),"test_mae":float(test_mae),"test_rmse":float(test_rmse)}
        display(pd.DataFrame([metrics]).round(4))
        '''),
        md("""## 8. Error analysis
        Aggregate errors can hide temporal regimes, so we inspect a forecast segment and residual distribution."""),
        code(r'''
        resid=yte_real-pred
        fig,axes=plt.subplots(1,2,figsize=(12,4)); axes[0].plot(yte_real[:250],label="actual"); axes[0].plot(pred[:250],label="forecast"); axes[0].legend(); axes[0].set_title("First 250 test hours"); axes[1].hist(resid,bins=30); axes[1].set_title("Residual distribution"); plt.tight_layout(); plt.show()
        print("largest absolute errors (°C):",np.round(np.sort(np.abs(resid))[-5:],2))
        '''),
        md("""## 9. Persist model, preprocessing contract and predictions"""),
        code(r'''
        model_path=PROJECT/"artifacts"/"best_model.keras"; best_model.save(model_path)
        (PROJECT/"artifacts"/"preprocessing.json").write_text(json.dumps({"features":features,"mean":mean.tolist(),"std":std.tolist(),"lookback":lookback},indent=2))
        (PROJECT/"reports"/"metrics.json").write_text(json.dumps(metrics,indent=2))
        pd.DataFrame({"actual_temp_c":yte_real[:500],"predicted_temp_c":pred[:500],"residual_c":resid[:500]}).to_csv(PROJECT/"reports"/"predictions.csv",index=False)
        pd.DataFrame({k:v for k,v in histories[best].items()}).to_csv(PROJECT/"reports"/"training_history.csv",index=False)
        print("saved:",model_path); print("artifact files:",[p.name for p in sorted((PROJECT/"artifacts").iterdir())])
        '''),
        md("""## 10. Reloaded inference and monitoring signal
        Reloading is part of the test: an artifact is useful only if it can independently reproduce inference. We also compare train/test feature means in standardized units as a simple drift diagnostic."""),
        code(r'''
        reloaded=tf.keras.models.load_model(model_path)
        one=float(inv_temp(reloaded.predict(Xte[:1],verbose=0).ravel())[0])
        print("one reloaded prediction °C:",round(one,3),"actual:",round(float(yte_real[0]),3))
        drift=pd.DataFrame({"feature":features,"train_mean":values[:train_end].mean(0),"test_mean":values[val_end:].mean(0),"shift_in_train_std":(values[val_end:].mean(0)-values[:train_end].mean(0))/std})
        display(drift.round(3))
        '''),
        md("""## Production interpretation
        The notebook demonstrates the complete experiment lifecycle. A production system would additionally version raw data snapshots, schedule label-aware performance monitoring, alert on data-quality/drift thresholds, and retrain only through a validated promotion process.""")
    ])


def cnn_notebooks() -> None:
    N = ROOT / "CNN/notebooks"

    write_nb(N / "00_cnn_learning_map.ipynb", "CNN learning map", "Understand why convolution is useful for spatial data and how the track progresses from pixels to production evaluation.", [
        code(SETUP),
        md("""CNNs preserve spatial locality. Instead of connecting every pixel to every hidden unit, a convolution learns small reusable filters that scan the image. Deeper layers combine local patterns into increasingly abstract representations."""),
        code(r'''
        stages=pd.DataFrame({"stage":["pixels","convolution","activation","pooling","deeper features","classifier"],"question":["what values exist?","what local pattern is present?","which responses matter?","what can be compressed?","what larger pattern emerges?","which class fits?"]})
        display(stages)
        '''),
        code(r'''
        img=np.zeros((16,16)); img[4:12,7:9]=1; img[7:9,4:12]=1
        plt.figure(figsize=(3,3)); plt.imshow(img,cmap="gray"); plt.title("Spatial structure matters"); plt.axis("off"); plt.show()
        ''')
    ])

    write_nb(N / "01_image_tensors.ipynb", "Images as tensors", "Read grayscale image batches and understand height, width and channel axes.", [
        code(SETUP),
        code(r'''
        (x_train,y_train),_=tf.keras.datasets.fashion_mnist.load_data()
        names=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
        print("raw image batch:",x_train.shape,"labels:",y_train.shape,"dtype:",x_train.dtype,"range:",(x_train.min(),x_train.max()))
        x=x_train[:8,...,None]
        print("CNN-ready batch:",x.shape,"= (batch, height, width, channels)")
        '''),
        code(r'''
        fig,axes=plt.subplots(2,4,figsize=(8,4))
        for ax,img,label in zip(axes.flat,x_train[:8],y_train[:8]): ax.imshow(img,cmap="gray"); ax.set_title(names[label]); ax.axis("off")
        plt.tight_layout(); plt.show()
        '''),
        md("""For RGB images the last dimension is usually 3. Keras `Conv2D` expects `(batch, height, width, channels)` by default.""")
    ])

    write_nb(N / "02_convolution_intuition.ipynb", "Convolution from first principles", "Compute one local filter response manually and match it with TensorFlow's `tf.nn.conv2d`.", [
        code(SETUP),
        code(r'''
        image=np.array([[0,0,0,0,0],[0,1,1,0,0],[0,1,1,0,0],[0,0,0,1,1],[0,0,0,1,1]],dtype="float32")
        kernel=np.array([[1,0,-1],[1,0,-1],[1,0,-1]],dtype="float32")
        patch=image[:3,:3]; manual=float(np.sum(patch*kernel))
        print("patch:\n",patch); print("kernel:\n",kernel); print("manual dot-product response:",manual)
        '''),
        code(r'''
        tf_out=tf.nn.conv2d(image[None,:,:,None],kernel[:,:,None,None],strides=1,padding="VALID")
        print("TensorFlow first response:",float(tf_out.numpy()[0,0,0,0])); print("full feature map:\n",tf_out.numpy()[0,:,:,0])
        plt.figure(figsize=(7,3)); plt.subplot(1,2,1); plt.imshow(image,cmap="gray"); plt.title("image"); plt.subplot(1,2,2); plt.imshow(tf_out.numpy()[0,:,:,0],cmap="coolwarm"); plt.title("feature map"); plt.tight_layout(); plt.show()
        '''),
        md("""Deep-learning libraries typically implement cross-correlation (no kernel flip) while using the conventional term convolution. The learning behavior is unaffected because filter values are learned.""")
    ])

    write_nb(N / "03_kernels_and_feature_maps.ipynb", "Kernels, filters and feature maps", "Apply interpretable edge filters and observe how different kernels produce different spatial responses.", [
        code(SETUP),
        code(r'''
        (_, _),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data(); img=x_test[0].astype("float32")/255
        kx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype="float32"); ky=kx.T
        def apply(k): return tf.nn.conv2d(img[None,:,:,None],k[:,:,None,None],1,"SAME").numpy()[0,:,:,0]
        fx,fy=apply(kx),apply(ky)
        fig,axes=plt.subplots(1,3,figsize=(9,3));
        for ax,a,t in zip(axes,[img,fx,fy],["image","vertical-edge response","horizontal-edge response"]): ax.imshow(a,cmap="gray"); ax.set_title(t); ax.axis("off")
        plt.tight_layout(); plt.show()
        '''),
        md("""A learned CNN begins with random filters and discovers useful patterns from the task loss. Every output channel is a separate learned feature map."""),
        code(r'''
        layer=tf.keras.layers.Conv2D(16,3,padding="same")
        out=layer(tf.zeros((4,28,28,1)))
        print("16 filters -> output shape:",out.shape,"kernel tensor:",layer.kernel.shape)
        ''')
    ])

    write_nb(N / "04_stride_padding_and_output_shapes.ipynb", "Stride, padding and output shapes", "Predict convolution output dimensions and verify them with TensorFlow.", [
        code(SETUP),
        code(r'''
        configs=[]
        for padding in ["valid","same"]:
            for stride in [1,2]:
                layer=tf.keras.layers.Conv2D(8,3,strides=stride,padding=padding)
                y=layer(tf.zeros((1,28,28,1)))
                configs.append({"padding":padding,"stride":stride,"output":str(tuple(y.shape)),"params":layer.count_params()})
        display(pd.DataFrame(configs))
        '''),
        md(r"""For `VALID`, spatial size roughly follows $\lfloor (n-k)/s\rfloor+1$. `SAME` chooses padding so stride 1 preserves size. Stride changes spatial sampling; filter count changes channel depth."""),
        code(r'''
        n=28;k=3
        print("VALID stride 1 formula:",(n-k)//1+1); print("VALID stride 2 formula:",(n-k)//2+1)
        ''')
    ])

    write_nb(N / "05_pooling_and_receptive_fields.ipynb", "Pooling and receptive fields", "See how pooling reduces spatial resolution and calculate how receptive fields expand through a stack.", [
        code(SETUP),
        code(r'''
        x=tf.reshape(tf.range(1,17,dtype=tf.float32),(1,4,4,1))
        maxp=tf.keras.layers.MaxPooling2D(2)(x); avgp=tf.keras.layers.AveragePooling2D(2)(x)
        print("input:\n",x.numpy()[0,:,:,0]); print("max pool:\n",maxp.numpy()[0,:,:,0]); print("average pool:\n",avgp.numpy()[0,:,:,0])
        '''),
        code(r'''
        layers=[("conv3",3,1),("pool2",2,2),("conv3",3,1),("pool2",2,2)]
        rf=1; jump=1; rows=[]
        for name,k,s in layers:
            rf=rf+(k-1)*jump; jump*=s; rows.append({"layer":name,"receptive_field":rf,"effective_jump":jump})
        display(pd.DataFrame(rows))
        '''),
        md("""Deeper units can represent larger spatial context because each layer's local neighborhood is built from neighborhoods in the preceding layer.""")
    ])

    write_nb(N / "06_conv2d_tensorflow.ipynb", "Conv2D in TensorFlow/Keras", "Connect filter count, kernel depth, bias and parameter count to a real Keras layer.", [
        code(SETUP),
        code(r'''
        layer=tf.keras.layers.Conv2D(32,3,padding="same",activation="relu")
        x=tf.zeros((5,28,28,3)); y=layer(x)
        expected=3*3*3*32+32
        print("input:",x.shape,"output:",y.shape); print("kernel:",layer.kernel.shape,"bias:",layer.bias.shape); print("parameters expected/actual:",expected,layer.count_params())
        '''),
        code(r'''
        model=tf.keras.Sequential([tf.keras.layers.Input((28,28,1)),tf.keras.layers.Conv2D(32,3,activation="relu"),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Conv2D(64,3,activation="relu")])
        model.summary()
        '''),
        md("""Parameter count depends on kernel area × input channels × output filters, not directly on image width/height. Weight sharing is why convolution is parameter-efficient.""")
    ])

    write_nb(N / "07_building_first_cnn.ipynb", "Build and train your first CNN", "Train a compact Fashion-MNIST CNN and read its learning curves and test accuracy.", [
        code(SETUP),
        code(r'''
        (x,y),(xt,yt)=tf.keras.datasets.fashion_mnist.load_data(); x=x[:5000].astype("float32")/255; y=y[:5000]; xt=xt[:1000].astype("float32")/255; yt=yt[:1000]
        x=x[...,None]; xt=xt[...,None]
        model=tf.keras.Sequential([tf.keras.layers.Input((28,28,1)),tf.keras.layers.Conv2D(16,3,activation="relu"),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Conv2D(32,3,activation="relu"),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Flatten(),tf.keras.layers.Dense(32,activation="relu"),tf.keras.layers.Dense(10,activation="softmax")])
        model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
        hist=model.fit(x,y,validation_split=.2,epochs=2,batch_size=64,verbose=0)
        loss,acc=model.evaluate(xt,yt,verbose=0); display(pd.DataFrame(hist.history).round(4)); print("held-out test accuracy:",round(float(acc),4))
        '''),
        code(r'''
        h=pd.DataFrame(hist.history); plt.figure(figsize=(7,3)); plt.plot(h["accuracy"],marker="o",label="train"); plt.plot(h["val_accuracy"],marker="o",label="validation"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Learning curve"); plt.show()
        ''')
    ])

    write_nb(N / "08_cnn_training_pipeline.ipynb", "A clean TensorFlow CNN training pipeline", "Use `tf.data`, batching, prefetching and explicit validation/test contracts.", [
        code(SETUP),
        code(r'''
        (x,y),(xt,yt)=tf.keras.datasets.fashion_mnist.load_data(); x=x[:6000,...,None].astype("float32")/255; y=y[:6000]
        xtr,xva=x[:5000],x[5000:]; ytr,yva=y[:5000],y[5000:]
        train_ds=tf.data.Dataset.from_tensor_slices((xtr,ytr)).shuffle(5000,seed=SEED).batch(64).prefetch(tf.data.AUTOTUNE)
        val_ds=tf.data.Dataset.from_tensor_slices((xva,yva)).batch(128).prefetch(tf.data.AUTOTUNE)
        batch=next(iter(train_ds)); print("one batch images:",batch[0].shape,"labels:",batch[1].shape); print("train batches:",tf.data.experimental.cardinality(train_ds).numpy())
        '''),
        code(r'''
        model=tf.keras.Sequential([tf.keras.layers.Input((28,28,1)),tf.keras.layers.Conv2D(16,3,padding="same",activation="relu"),tf.keras.layers.MaxPooling2D(),tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(10,activation="softmax")])
        model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
        h=model.fit(train_ds,validation_data=val_ds,epochs=1,verbose=0)
        display(pd.DataFrame(h.history).round(4))
        '''),
        md("""`tf.data` separates data movement from model definition. In larger systems it becomes the place for parsing, caching, parallel mapping, shuffling and prefetching.""")
    ])

    write_nb(N / "09_batchnorm_dropout.ipynb", "Batch normalization and dropout", "Observe train/inference behavior and understand where regularization layers fit in a CNN.", [
        code(SETUP),
        code(r'''
        x=tf.ones((4,8,8,3))
        drop=tf.keras.layers.Dropout(.5); bn=tf.keras.layers.BatchNormalization()
        train_drop=drop(x,training=True); infer_drop=drop(x,training=False)
        _=bn(x,training=True); infer_bn=bn(x,training=False)
        print("dropout zeros during training:",float(tf.reduce_mean(tf.cast(train_drop==0,tf.float32)))); print("dropout inference mean:",float(tf.reduce_mean(infer_drop))); print("batchnorm inference output mean:",float(tf.reduce_mean(infer_bn)))
        '''),
        code(r'''
        block=tf.keras.Sequential([tf.keras.layers.Input((28,28,1)),tf.keras.layers.Conv2D(32,3,use_bias=False),tf.keras.layers.BatchNormalization(),tf.keras.layers.ReLU(),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Dropout(.25)])
        block.summary()
        '''),
        md("""Batch normalization changes activation statistics; dropout randomly masks activations during training. Both have different behavior when `training=False`, which is why inference mode matters.""")
    ])

    write_nb(N / "10_data_augmentation.ipynb", "Image augmentation", "Create label-preserving image variations inside the TensorFlow model and inspect the transformed examples.", [
        code(SETUP),
        code(r'''
        (x,y),_=tf.keras.datasets.fashion_mnist.load_data(); img=x[0].astype("float32")/255
        aug=tf.keras.Sequential([tf.keras.layers.RandomRotation(.06,seed=SEED),tf.keras.layers.RandomTranslation(.08,.08,seed=SEED+1)])
        batch=tf.repeat(img[None,:,:,None],6,axis=0); out=aug(batch,training=True).numpy()
        fig,axes=plt.subplots(2,3,figsize=(6,6))
        for ax,a in zip(axes.flat,out): ax.imshow(a[:,:,0],cmap="gray"); ax.axis("off")
        plt.suptitle("Different stochastic views of one image"); plt.tight_layout(); plt.show()
        '''),
        md("""Augmentation should encode plausible invariances. Transformations that change the semantic label create corrupted supervision rather than regularization."""),
        code(r'''
        print("augmentation output shape:",out.shape,"value range:",(float(out.min()),float(out.max())))
        ''')
    ])

    write_nb(N / "11_deeper_cnn_architectures.ipynb", "Designing deeper CNN architectures", "Build reusable convolution blocks and observe how spatial resolution and channel depth evolve.", [
        code(SETUP),
        code(r'''
        inputs=tf.keras.Input((64,64,3)); x=inputs; rows=[]
        for i,filters in enumerate([32,64,128],start=1):
            x=tf.keras.layers.Conv2D(filters,3,padding="same",activation="relu",name=f"block{i}_conv1")(x)
            x=tf.keras.layers.Conv2D(filters,3,padding="same",activation="relu",name=f"block{i}_conv2")(x)
            x=tf.keras.layers.MaxPooling2D(name=f"block{i}_pool")(x)
            rows.append({"block":i,"shape":str(tuple(x.shape)),"filters":filters})
        x=tf.keras.layers.GlobalAveragePooling2D()(x); outputs=tf.keras.layers.Dense(10,activation="softmax")(x); model=tf.keras.Model(inputs,outputs)
        display(pd.DataFrame(rows)); print("parameters:",model.count_params()); print("output:",model(tf.zeros((2,64,64,3))).shape)
        '''),
        md("""A common design pattern is decreasing spatial resolution while increasing channel capacity. Global average pooling can replace a large Flatten→Dense head and sharply reduce parameters."""),
        code(r'''
        model.summary()
        ''')
    ])

    write_nb(N / "12_transfer_learning.ipynb", "Transfer learning with MobileNetV2", "Use an ImageNet-pretrained TensorFlow backbone as a frozen feature extractor and inspect the resulting representation.", [
        code(SETUP),
        md("""Transfer learning reuses representations learned from a large source task. We freeze a pretrained MobileNetV2 backbone and transform a Fashion-MNIST image into a 1,280-dimensional feature vector."""),
        code(r'''
        base=tf.keras.applications.MobileNetV2(weights="imagenet",include_top=False,input_shape=(96,96,3),pooling="avg")
        base.trainable=False
        (x,y),_=tf.keras.datasets.fashion_mnist.load_data(); img=tf.cast(x[0][...,None],tf.float32)
        img=tf.image.resize(img,(96,96)); img=tf.repeat(img,3,axis=-1); batch=tf.keras.applications.mobilenet_v2.preprocess_input(img[None,...])
        feat=base(batch,training=False)
        print("backbone trainable:",base.trainable,"parameters:",base.count_params()); print("feature vector:",feat.shape,"L2 norm:",round(float(tf.norm(feat)),3))
        '''),
        code(r'''
        head=tf.keras.Sequential([tf.keras.layers.Input((1280,)),tf.keras.layers.Dense(10,activation="softmax")])
        print("small task-specific head parameters:",head.count_params())
        '''),
        md("""Typical workflow: train the new head with the backbone frozen, then optionally unfreeze a small upper portion with a much lower learning rate. Validation data decides whether fine-tuning helps.""")
    ])

    write_nb(N / "13_feature_maps_and_gradcam.ipynb", "Feature maps and Grad-CAM", "Inspect internal activations and compute a class-specific Grad-CAM heatmap for a trained CNN.", [
        code(SETUP),
        code(r'''
        (x,y),(xt,yt)=tf.keras.datasets.fashion_mnist.load_data(); x=x[:4000,...,None].astype("float32")/255; y=y[:4000]; xt=xt[:200,...,None].astype("float32")/255
        inputs=tf.keras.Input((28,28,1)); z=tf.keras.layers.Conv2D(16,3,padding="same",activation="relu",name="conv1")(inputs); z=tf.keras.layers.MaxPooling2D()(z); z=tf.keras.layers.Conv2D(32,3,padding="same",activation="relu",name="conv2")(z); z=tf.keras.layers.GlobalAveragePooling2D()(z); outputs=tf.keras.layers.Dense(10,activation="softmax")(z)
        model=tf.keras.Model(inputs,outputs); model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]); model.fit(x,y,epochs=1,batch_size=64,verbose=0)
        grad_model=tf.keras.Model(model.inputs,[model.get_layer("conv2").output,model.output])
        sample=tf.convert_to_tensor(xt[:1]);
        with tf.GradientTape() as tape:
            conv,preds=grad_model(sample); cls=tf.argmax(preds[0]); score=preds[:,cls]
        grads=tape.gradient(score,conv); weights=tf.reduce_mean(grads,axis=(0,1,2)); heat=tf.reduce_sum(conv[0]*weights,axis=-1); heat=tf.maximum(heat,0); heat=heat/(tf.reduce_max(heat)+1e-8)
        print("predicted class:",int(cls),"confidence:",round(float(tf.reduce_max(preds)),4),"heatmap shape:",heat.shape)
        '''),
        code(r'''
        fig,axes=plt.subplots(1,2,figsize=(6,3)); axes[0].imshow(xt[0,:,:,0],cmap="gray"); axes[0].set_title("input"); axes[1].imshow(tf.image.resize(heat[...,None],(28,28)).numpy()[:,:,0],cmap="inferno"); axes[1].set_title("Grad-CAM"); [a.axis("off") for a in axes]; plt.tight_layout(); plt.show()
        '''),
        md("""Grad-CAM is an explanatory diagnostic, not proof of causal reasoning. Use it to inspect where class-sensitive gradients concentrate and combine it with quantitative error analysis.""")
    ])

    write_nb(N / "14_error_analysis_and_calibration.ipynb", "Error analysis and calibration", "Go beyond accuracy with a confusion matrix, per-class error rates and expected calibration error.", [
        code(SETUP),
        code(r'''
        from sklearn.metrics import confusion_matrix, accuracy_score
        (x,y),(xt,yt)=tf.keras.datasets.fashion_mnist.load_data(); x=x[:5000,...,None].astype("float32")/255; y=y[:5000]; xt=xt[:1500,...,None].astype("float32")/255; yt=yt[:1500]
        model=tf.keras.Sequential([tf.keras.layers.Input((28,28,1)),tf.keras.layers.Conv2D(16,3,activation="relu"),tf.keras.layers.MaxPooling2D(),tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(10,activation="softmax")]); model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]); model.fit(x,y,epochs=2,batch_size=64,verbose=0)
        prob=model.predict(xt,verbose=0); pred=prob.argmax(1); cm=confusion_matrix(yt,pred); print("accuracy:",round(accuracy_score(yt,pred),4))
        '''),
        code(r'''
        plt.figure(figsize=(6,5)); plt.imshow(cm,cmap="Blues"); plt.colorbar(); plt.xlabel("predicted"); plt.ylabel("actual"); plt.title("Confusion matrix"); plt.show()
        conf=prob.max(1); correct=(pred==yt).astype(float); bins=np.linspace(0,1,11); ece=0
        for lo,hi in zip(bins[:-1],bins[1:]):
            m=(conf>=lo)&(conf<hi)
            if m.any(): ece += m.mean()*abs(correct[m].mean()-conf[m].mean())
        print("10-bin expected calibration error:",round(float(ece),4)); print("mean confidence:",round(float(conf.mean()),4),"accuracy:",round(float(correct.mean()),4))
        '''),
        md("""Calibration asks whether a prediction reported at 80% confidence is correct about 80% of the time. A model can be accurate yet overconfident.""")
    ])

    write_nb(N / "15_cnn_inference_and_monitoring.ipynb", "CNN serialization, inference and monitoring", "Save and reload a CNN, measure inference latency and detect a simple brightness distribution shift.", [
        code(SETUP),
        code(r'''
        model=tf.keras.Sequential([tf.keras.layers.Input((28,28,1)),tf.keras.layers.Conv2D(8,3,activation="relu"),tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(10,activation="softmax")]); _=model(tf.zeros((1,28,28,1)))
        path=Path("CNN")/"artifacts_demo_cnn.keras"; model.save(path); restored=tf.keras.models.load_model(path)
        sample=tf.random.uniform((32,28,28,1),seed=SEED); import time; start=time.perf_counter(); out=restored.predict(sample,verbose=0); ms=(time.perf_counter()-start)*1000/len(sample)
        print("saved/reloaded:",path,"output:",out.shape,"approx ms/image:",round(ms,3))
        '''),
        code(r'''
        (x,_),_=tf.keras.datasets.fashion_mnist.load_data(); ref=x[:2000].astype("float32")/255; shifted=np.clip(ref*.65,0,1)
        stats=pd.DataFrame({"population":["reference","darkened production simulation"],"pixel_mean":[ref.mean(),shifted.mean()],"pixel_std":[ref.std(),shifted.std()]}); display(stats.round(4))
        '''),
        md("""Production image monitoring can track intensity/channel statistics, resolution, corruption rates, class mix, confidence, latency and labeled performance. Drift alarms should trigger investigation, not automatic retraining without validation.""")
    ])


def cnn_project() -> None:
    P = ROOT / "CNN/projects/fashion_mnist_classifier"
    write_nb(P / "00_end_to_end_fashion_mnist.ipynb", "Project: end-to-end Fashion-MNIST classifier", "Take real image data from acquisition and EDA through baseline/CNN comparison, untouched-test evaluation, error analysis, calibration, serialization, inference and drift simulation.", [
        code(SETUP + r'''
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json, time
PROJECT=Path("CNN/projects/fashion_mnist_classifier")
(PROJECT/"artifacts").mkdir(parents=True,exist_ok=True)
(PROJECT/"reports").mkdir(parents=True,exist_ok=True)
names=np.array(["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])
        '''),
        md("""## 1. Problem and split contract
        **Goal:** classify a 28×28 grayscale apparel image into one of ten classes. We use TensorFlow's canonical Fashion-MNIST loader. Validation is carved only from the original training partition; the canonical test partition remains untouched until final evaluation."""),
        code(r'''
        (x_all,y_all),(x_test_all,y_test_all)=tf.keras.datasets.fashion_mnist.load_data()
        print("canonical training:",x_all.shape,y_all.shape,"canonical test:",x_test_all.shape,y_test_all.shape); print("pixel range:",(x_all.min(),x_all.max()))
        display(pd.Series(y_all).value_counts().sort_index().rename(index=dict(enumerate(names))).to_frame("count"))
        '''),
        md("""## 2. EDA
        Visual inspection checks whether labels and image semantics align and exposes the inherent ambiguity between visually similar classes such as shirts, T-shirts, coats and pullovers."""),
        code(r'''
        fig,axes=plt.subplots(3,6,figsize=(11,6))
        for ax,img,label in zip(axes.flat,x_all[:18],y_all[:18]): ax.imshow(img,cmap="gray"); ax.set_title(names[label],fontsize=8); ax.axis("off")
        plt.tight_layout(); plt.show()
        '''),
        md("""## 3. Deterministic train/validation/test subsets and normalization
        A compact subset keeps the notebook practical on CPU while preserving all 10 classes. Pixel scaling to `[0,1]` is deterministic and does not learn statistics from test data."""),
        code(r'''
        rng=np.random.default_rng(SEED); idx=rng.permutation(len(x_all)); train_idx=idx[:12000]; val_idx=idx[12000:14000]
        Xtr=x_all[train_idx,...,None].astype("float32")/255; ytr=y_all[train_idx]
        Xva=x_all[val_idx,...,None].astype("float32")/255; yva=y_all[val_idx]
        Xte=x_test_all[:3000,...,None].astype("float32")/255; yte=y_test_all[:3000]
        print("train/val/test:",Xtr.shape,Xva.shape,Xte.shape)
        '''),
        md("""## 4. Dense baseline
        Flattening discards explicit spatial structure. That makes an MLP a useful baseline: if convolution helps, it should beat or match this simpler hypothesis under the same data contract."""),
        code(r'''
        def build_mlp():
            m=tf.keras.Sequential([tf.keras.layers.Input((28,28,1)),tf.keras.layers.Flatten(),tf.keras.layers.Dense(128,activation="relu"),tf.keras.layers.Dense(10,activation="softmax")],name="mlp_baseline")
            m.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]); return m
        tf.keras.utils.set_random_seed(SEED); mlp=build_mlp(); h_mlp=mlp.fit(Xtr,ytr,validation_data=(Xva,yva),epochs=2,batch_size=64,verbose=0); print("MLP parameters:",mlp.count_params()); display(pd.DataFrame(h_mlp.history).round(4))
        '''),
        md("""## 5. CNN model
        The CNN preserves local neighborhoods through two convolution/pooling stages, then uses a compact dense head. Dropout regularizes the representation."""),
        code(r'''
        def build_cnn():
            m=tf.keras.Sequential([tf.keras.layers.Input((28,28,1)),tf.keras.layers.Conv2D(32,3,padding="same",activation="relu",name="conv1"),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Conv2D(64,3,padding="same",activation="relu",name="conv2"),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Flatten(),tf.keras.layers.Dense(64,activation="relu"),tf.keras.layers.Dropout(.2),tf.keras.layers.Dense(10,activation="softmax")],name="cnn")
            m.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]); return m
        tf.keras.utils.set_random_seed(SEED); cnn=build_cnn(); cnn.summary(); h_cnn=cnn.fit(Xtr,ytr,validation_data=(Xva,yva),epochs=2,batch_size=64,verbose=0); display(pd.DataFrame(h_cnn.history).round(4))
        '''),
        md("""## 6. Validation-based model selection
        Test data still has not been used. We choose using validation accuracy only."""),
        code(r'''
        comparison=pd.DataFrame([{"model":"MLP","best_val_accuracy":max(h_mlp.history["val_accuracy"]),"parameters":mlp.count_params()},{"model":"CNN","best_val_accuracy":max(h_cnn.history["val_accuracy"]),"parameters":cnn.count_params()}]).sort_values("best_val_accuracy",ascending=False)
        display(comparison.round(4)); selected=comparison.iloc[0]["model"]; best=cnn if selected=="CNN" else mlp; print("selected:",selected)
        '''),
        md("""## 7. Final untouched test evaluation"""),
        code(r'''
        prob=best.predict(Xte,batch_size=128,verbose=0); pred=prob.argmax(1); test_acc=accuracy_score(yte,pred)
        report=classification_report(yte,pred,target_names=names,output_dict=True,zero_division=0)
        print("test accuracy:",round(float(test_acc),4)); display(pd.DataFrame(report).T.loc[names,["precision","recall","f1-score","support"]].round(3))
        '''),
        md("""## 8. Confusion and misclassification analysis"""),
        code(r'''
        cm=confusion_matrix(yte,pred); plt.figure(figsize=(7,6)); plt.imshow(cm,cmap="Blues"); plt.colorbar(); plt.xticks(range(10),names,rotation=60,ha="right",fontsize=8); plt.yticks(range(10),names,fontsize=8); plt.xlabel("predicted"); plt.ylabel("actual"); plt.title("Test confusion matrix"); plt.tight_layout(); plt.show()
        wrong=np.where(pred!=yte)[0]; hardest=wrong[np.argsort(prob[wrong,pred[wrong]])[-8:]] if len(wrong)>=8 else wrong
        fig,axes=plt.subplots(2,4,figsize=(9,5))
        for ax,i in zip(axes.flat,hardest): ax.imshow(Xte[i,:,:,0],cmap="gray"); ax.set_title(f"true {names[yte[i]]}\npred {names[pred[i]]}\nconf {prob[i,pred[i]]:.2f}",fontsize=8); ax.axis("off")
        plt.tight_layout(); plt.show()
        '''),
        md("""## 9. Calibration diagnostic
        Accuracy and confidence are different properties. Expected calibration error (ECE) compares confidence with empirical correctness across bins."""),
        code(r'''
        conf=prob.max(1); correct=(pred==yte).astype(float); bins=np.linspace(0,1,11); ece=0.; rows=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            m=(conf>=lo)&(conf<hi)
            if m.any():
                acc=correct[m].mean(); c=conf[m].mean(); ece += m.mean()*abs(acc-c); rows.append({"bin":f"{lo:.1f}-{hi:.1f}","n":int(m.sum()),"accuracy":acc,"confidence":c})
        display(pd.DataFrame(rows).round(3)); print("ECE:",round(float(ece),4))
        '''),
        md("""## 10. Persist model, metrics, history and predictions"""),
        code(r'''
        model_path=PROJECT/"artifacts"/"best_model.keras"; best.save(model_path)
        metrics={"selected_model":selected,"test_accuracy":float(test_acc),"ece":float(ece),"validation_accuracy":float(comparison.iloc[0]["best_val_accuracy"])}
        (PROJECT/"reports"/"metrics.json").write_text(json.dumps(metrics,indent=2))
        hist=h_cnn.history if selected=="CNN" else h_mlp.history; pd.DataFrame(hist).to_csv(PROJECT/"reports"/"training_history.csv",index=False)
        pd.DataFrame({"actual":yte[:1000],"predicted":pred[:1000],"confidence":conf[:1000]}).to_csv(PROJECT/"reports"/"predictions.csv",index=False)
        print("saved:",model_path); print(metrics)
        '''),
        md("""## 11. Reloaded inference"""),
        code(r'''
        restored=tf.keras.models.load_model(model_path); one=restored.predict(Xte[:1],verbose=0)[0]; print("actual:",names[yte[0]],"predicted:",names[one.argmax()],"confidence:",round(float(one.max()),4))
        '''),
        md("""## 12. Monitoring simulation
        We darken images to mimic an input-distribution change, then compare confidence and accuracy. This is not a complete drift detector; it demonstrates the operational question a monitoring layer must answer."""),
        code(r'''
        shifted=Xte*.65; p_shift=restored.predict(shifted,batch_size=128,verbose=0); pred_shift=p_shift.argmax(1)
        monitor=pd.DataFrame({"population":["reference test","darkened simulation"],"pixel_mean":[Xte.mean(),shifted.mean()],"accuracy":[test_acc,accuracy_score(yte,pred_shift)],"mean_confidence":[prob.max(1).mean(),p_shift.max(1).mean()]})
        display(monitor.round(4))
        '''),
        md("""## Project conclusion
        This project demonstrates the full experimental path from raw dataset loading through model choice and final evaluation to deployable artifacts and post-deployment diagnostics. Production extensions include data/version lineage, service-level latency/load tests, class-specific monitoring and governed retraining.""")
    ])


def update_root_readme() -> None:
    path = ROOT / "README.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    marker = "<!-- RNN_CNN_TRACKS -->"
    block = """
<!-- RNN_CNN_TRACKS -->
## Recurrent Neural Networks (TensorFlow/Keras)
The [`RNN/`](RNN/) track contains executed concept notebooks and a complete Jena-climate forecasting project. It covers sequence tensors, windowing, recurrence, BPTT, gradient pathologies, LSTM, GRU, architecture patterns, evaluation, serialization and monitoring.

- [RNN run guide](RNN/README.md)
- [RNN Conda environment](RNN/environment.yml)
- [RNN notebooks](RNN/notebooks/)
- [End-to-end Jena forecasting project](RNN/projects/jena_climate_forecasting/)

## Convolutional Neural Networks (TensorFlow/Keras)
The [`CNN/`](CNN/) track contains executed concept notebooks and a complete Fashion-MNIST project. It covers image tensors, manual convolution, kernels, padding/stride, pooling, Conv2D, regularization, augmentation, transfer learning, Grad-CAM, calibration, serialization and monitoring.

- [CNN run guide](CNN/README.md)
- [CNN Conda environment](CNN/environment.yml)
- [CNN notebooks](CNN/notebooks/)
- [End-to-end Fashion-MNIST project](CNN/projects/fashion_mnist_classifier/)
<!-- /RNN_CNN_TRACKS -->
"""
    if marker in text:
        start = text.index(marker)
        end_marker = "<!-- /RNN_CNN_TRACKS -->"
        end = text.index(end_marker, start) + len(end_marker)
        text = text[:start] + block.strip() + text[end:]
    else:
        text = text.rstrip() + "\n\n" + block.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    build_shared_files()
    rnn_notebooks()
    rnn_project()
    cnn_notebooks()
    cnn_project()
    update_root_readme()
    print("Generated TensorFlow RNN/CNN curriculum under", ROOT)


if __name__ == "__main__":
    main()
