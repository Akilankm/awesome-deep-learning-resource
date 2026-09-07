# Training Debug Playbook

1. Overfit a tiny batch to near-zero loss; failure implies model/loss/gradient bugs.
2. Verify label range, dtype, input scale and split integrity.
3. Inspect initial logits and loss; 10-class random cross-entropy should be near $\log(10)$.
4. Inspect gradient norms and weight norms.
5. Plot train and validation loss separately.
6. Try learning-rate changes before architectural complexity.
7. Use finite-difference gradient checks for custom math.
8. Inspect actual errors, not only aggregate metrics.
9. Reproduce from a clean environment and fresh kernel.
10. Never tune against the final test set.
