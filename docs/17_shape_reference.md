# Tensor and Shape Reference

For MNIST MLP 784 → 64 → 10 and batch size $B$:

| Object | Shape | Meaning |
|---|---:|---|
| $X$ | $B\times784$ | normalized flattened pixels |
| $W_1$ | $784\times64$ | first-layer weights |
| $b_1$ | $64$ | first-layer bias |
| $Z_1,A_1$ | $B\times64$ | pre/post activation |
| $W_2$ | $64\times10$ | output weights |
| logits | $B\times10$ | unnormalized class evidence |
| probabilities | $B\times10$ | softmax outputs |
| labels | $B$ | integer class IDs |

Shape reasoning should be performed before debugging numeric values.
