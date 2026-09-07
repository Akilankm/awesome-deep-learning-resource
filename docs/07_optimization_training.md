# Optimization and Training Loops

SGD applies the gradient directly; Momentum accumulates velocity; Adam normalizes updates using moving first/second moments. The learning rate is often the dominant optimization hyperparameter.

A complete epoch contains shuffled mini-batch training plus forward-only validation. Record training/validation loss, accuracy, gradient/weight norms, hyperparameters, seed, code version, data version and artifact metadata.

## Learning questions
- What assumption does this stage make?
- What can fail silently?
- What evidence would you monitor?
- How does this affect business or production behavior?
