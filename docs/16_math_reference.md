# ANN Math Reference

## Dense layer
$Z=XW+b$. For batch $B$, input $D$, hidden $H$: $X\in\mathbb{R}^{B\times D}$, $W\in\mathbb{R}^{D\times H}$, $Z\in\mathbb{R}^{B\times H}$.

## ReLU
$ReLU(z)=\max(0,z)$ and derivative is 1 for positive pre-activation, 0 for negative values (the derivative at exactly zero is convention-dependent).

## Softmax and cross-entropy
$p_i=e^{z_i}/\sum_j e^{z_j}$; for the true class $y$, $L=-\log p_y$. With softmax-cross-entropy, the derivative with respect to logits simplifies to $p-y_{onehot}$.

## Gradient update
SGD: $\theta_{t+1}=\theta_t-\eta\nabla_\theta L$. Adam rescales updates using moving estimates of gradient moments.
