from pathlib import Path
import nbformat

TARGET = Path('04_training_and_representation_learning.ipynb')

if not TARGET.exists():
    raise SystemExit(f'Missing generated notebook: {TARGET}')

nb = nbformat.read(TARGET, as_version=4)

explanation = r'''## Why this notebook matters: training is geometry changing over time

A neural network does not merely memorize a better answer after every optimizer step. Its **internal coordinate system changes**. The same MNIST image is repeatedly mapped through the network, but the hidden coordinates produced by the learned weights and biases move as training proceeds.

For a hidden layer,

$$
\mathbf{h}=\phi(W\mathbf{x}+\mathbf{b})
$$

so changing $W$ or $\mathbf{b}$ changes the coordinates $\mathbf{h}$ assigned to the same input image $\mathbf{x}$. This is the concrete meaning of **representation learning**.

### The complete causal loop

1. A 28×28 digit becomes a 784-dimensional input vector $\mathbf{x}$.
2. Forward propagation computes hidden coordinates and class logits.
3. The loss measures how incompatible those logits are with the true digit label.
4. Backpropagation computes how each parameter contributed to that error.
5. The optimizer updates parameters in the direction that should reduce future loss.
6. On the next forward pass, the same digit lands at a slightly different hidden representation.
7. Repeating this process can make digits of the same class form more coherent regions while separating confusing classes.

The update is conceptually

$$
\theta_{t+1}=\theta_t-\eta\nabla_{\theta}\mathcal{L}(\theta_t),
$$

where $\theta$ represents all weights and biases and $\eta$ is the learning rate. Because the representation itself depends on $\theta$, every update can reshape the geometry seen by later layers.

### What controls the learned geometry?

| Factor | What changes | What you may observe |
| --- | --- | --- |
| Initialization | Starting orientation and scale of the feature space | Different early trajectories and convergence speed |
| Learning rate | Size of each parameter update | Smooth progress, oscillation, or divergence |
| Activation | Which affine responses survive or saturate | Sparse, clipped, curved, or saturated representations |
| Hidden width | Capacity of the intermediate representation | Underfitting vs richer separability |
| Loss function | What errors are penalized | Different class margins and confidence behaviour |
| Optimizer | How gradient history is converted into updates | Different paths through parameter space |
| Regularization | Preference for simpler parameter configurations | Better generalization or reduced overfitting |

### How to read the visualizations in this notebook

Do not look only at the final accuracy. Follow one sample conceptually through time. At epoch 0 its hidden coordinate may be close to several competing classes. After many gradient updates, that same sample may move into a region where the output layer can separate it with a much simpler boundary. The model has therefore made the downstream classification problem easier by learning a better representation.

A useful mental model is:

$$
\text{pixels} \rightarrow \text{learned coordinates} \rightarrow \text{class evidence} \rightarrow \text{loss}
$$

and during training the information flows backward:

$$
\text{loss} \rightarrow \text{gradients} \rightarrow \text{parameter updates} \rightarrow \text{new coordinates}.
$$

### Engineering interpretation

Training curves alone are insufficient for debugging. Two runs can have similar loss while learning very different internal geometries. In production work, inspect several signals together: training/test loss, class-wise accuracy, confusion matrices, confidence distributions, gradient magnitude, and representative hidden embeddings. If the loss falls but test geometry remains poorly separated, the model may be fitting shortcuts rather than learning transferable structure.

### Business interpretation

For MNIST the customer is simply the downstream digit classifier, but the same mechanism appears in enterprise systems. A representation layer can turn raw events, documents, images, transactions, or customer attributes into coordinates where fraud, intent, defect type, risk category, or product class becomes easier to separate. The commercial value of a deep model often comes from this learned representation rather than from the final linear decision layer alone.

### Final checkpoint

After this notebook you should be able to explain the full causal chain without framework terminology:

**input → weighted transformations → hidden representation → prediction → loss → gradient → parameter update → changed representation.**

That chain is the core mechanism behind learning in every ANN implementation in this repository, whether the arithmetic is written manually, delegated to PyTorch autograd, or delegated to TensorFlow GradientTape.
'''

# Insert immediately after the title/intro so GitHub readers see the conceptual map before code.
nb.cells.insert(1, nbformat.v4.new_markdown_cell(explanation))
nbformat.write(nb, TARGET)
print(f'Enriched {TARGET} with representation-learning explanation.')
