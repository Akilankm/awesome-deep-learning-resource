# Activations and Initialization

Without nonlinear activations, stacked dense layers collapse into one affine map. Sigmoid and tanh can saturate; ReLU is piecewise linear and computationally simple; Leaky ReLU preserves a small negative-side derivative.

Initialization controls signal and gradient scale before learning. Xavier/Glorot is appropriate for approximately symmetric activations; He/Kaiming targets ReLU-family networks. Poor initialization can yield exploding, vanishing or dead activations.

## Learning questions
- What assumption does this stage make?
- What can fail silently?
- What evidence would you monitor?
- How does this affect business or production behavior?
