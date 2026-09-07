# Final Test, Error Analysis, and Calibration

Final test is used only after development choices are frozen. Evaluate accuracy plus per-class precision/recall/F1 and confusion matrices. Inspect actual failures; aggregate metrics cannot explain behavior.

Confidence is not automatically calibrated probability. Reliability diagrams and Expected Calibration Error compare confidence with empirical correctness. Calibration matters when confidence drives automation or escalation.

## Learning questions
- What assumption does this stage make?
- What can fail silently?
- What evidence would you monitor?
- How does this affect business or production behavior?
