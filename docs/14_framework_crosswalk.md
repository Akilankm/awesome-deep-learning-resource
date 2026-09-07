# Manual vs PyTorch vs TensorFlow Crosswalk

The three implementation branches solve the same MNIST learning problem. Manual NumPy exposes matrix operations and derivatives; PyTorch exposes the same math through tensors, `nn.Module`, autograd and optimizers; TensorFlow exposes it through tensors, Keras layers, `GradientTape` and optimizers.

Use the manual branch to understand mechanism, then compare identical stages in PyTorch and TensorFlow. Framework abstractions should reduce boilerplate without reducing conceptual understanding.

## Learning questions
- What assumption does this stage make?
- What can fail silently?
- What evidence would you monitor?
- How does this affect business or production behavior?
