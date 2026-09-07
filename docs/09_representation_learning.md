# Representation Learning

Hidden layers learn coordinates useful for the objective. Because $h=\phi(Wx+b)$, changing $W$ and $b$ changes where the same input lives in hidden space. A 2D bottleneck makes this movement directly observable across epochs.

Representations become embeddings in many production systems: documents, products, customers, transactions and images. Downstream simplicity often depends on representation quality.

## Learning questions
- What assumption does this stage make?
- What can fail silently?
- What evidence would you monitor?
- How does this affect business or production behavior?
