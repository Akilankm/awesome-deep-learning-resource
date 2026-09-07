# Validation, Hyperparameters, and Failure Modes

Use validation—not test—to choose learning rate, architecture, regularization, optimizer and epoch budget. Underfitting produces poor train and validation performance; overfitting produces a widening generalization gap.

Failure modes include exploding/vanishing gradients, dead ReLUs, saturation, data leakage, unstable loss, bad preprocessing, excessive capacity and distribution mismatch. Curves are diagnostic evidence, not decoration.

## Learning questions
- What assumption does this stage make?
- What can fail silently?
- What evidence would you monitor?
- How does this affect business or production behavior?
