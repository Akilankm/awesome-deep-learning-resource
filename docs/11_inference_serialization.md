# Serialization, Inference, and Performance

The deployed unit is a versioned artifact plus preprocessing and output contract. Verify predictions before and after reload. Separate single-item latency, batched throughput, warm-up and end-to-end service latency.

Inference code should be deterministic, shape-safe and explicit about device/dtype. Model timing alone excludes decoding, network calls, feature retrieval and downstream policy.

## Learning questions
- What assumption does this stage make?
- What can fail silently?
- What evidence would you monitor?
- How does this affect business or production behavior?
