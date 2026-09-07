# Forward Propagation, Logits, and Softmax

Forward propagation computes intermediate representations and final logits. Logits are unbounded class evidence, not calibrated probabilities. Softmax maps logits to a simplex while preserving argmax. Use a max-shift for numerical stability.

Prediction is a model output; business action may require thresholds, abstention, human review or policy rules.

## Learning questions
- What assumption does this stage make?
- What can fail silently?
- What evidence would you monitor?
- How does this affect business or production behavior?
