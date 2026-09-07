# Neuron and Linear Algebra

A neuron computes $z=\mathbf{w}^T\mathbf{x}+b$. A dense layer vectorizes many neurons: $Z=XW+b$. The weight matrix determines directional sensitivity; the bias translates the affine response.

Shape reasoning is a debugging superpower: if $X\in\mathbb{R}^{B\times D}$ and $W\in\mathbb{R}^{D\times H}$, then $Z\in\mathbb{R}^{B\times H}$. Parameter count and FLOPs connect architecture to deployment cost.

## Learning questions
- What assumption does this stage make?
- What can fail silently?
- What evidence would you monitor?
- How does this affect business or production behavior?
