# Preprocessing, Batching, and Data Contracts

Preprocessing is model logic. Scaling, normalization, encoding and reshaping must be identical across training, validation, test and serving. Mini-batches approximate full-dataset gradients while enabling memory-efficient matrix operations.

For a dense MNIST layer, a batch has shape $B\times784$. Batch size affects gradient noise, memory, throughput and sometimes generalization. Production input contracts should specify schema, dtype, range, shape, null policy and version.

## Learning questions
- What assumption does this stage make?
- What can fail silently?
- What evidence would you monitor?
- How does this affect business or production behavior?
