# ANN Big Picture

An artificial neural network is a parameterized function that learns representations by repeated forward evaluation, loss measurement, gradient computation and parameter updates. This chapter establishes the complete lifecycle from business objective to monitored production system.

## System map
`problem → data → split → preprocessing → representation → forward → loss → backprop → optimization → validation → test → inference → monitoring → retraining`.

A technically correct model is not a complete ML system. Data contracts, evaluation boundaries, artifact reproducibility and operational monitoring are part of the model's correctness.

## Learning questions
- What assumption does this stage make?
- What can fail silently?
- What evidence would you monitor?
- How does this affect business or production behavior?
