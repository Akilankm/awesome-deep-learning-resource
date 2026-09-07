# Loss, Chain Rule, and Backpropagation

A loss defines what optimization means. Cross-entropy for a true class with probability $p_y$ is $-\log p_y$. Backpropagation applies the chain rule from the scalar loss through every operation to compute parameter sensitivities.

For debugging custom math, compare analytical gradients with finite differences: $\frac{L(w+\epsilon)-L(w-\epsilon)}{2\epsilon}$. Autodiff automates bookkeeping, not the underlying calculus.

## Learning questions
- What assumption does this stage make?
- What can fail silently?
- What evidence would you monitor?
- How does this affect business or production behavior?
