# Awesome Deep Learning Resource

A visual, executable, first-principles deep-learning learning repository.

The goal of this project is to make deep learning **observable**: not only equations and API calls, but live parameter changes, geometry, computational graphs, gradients, loss surfaces, learned representations, and training dynamics.

## Learning tracks

### 1. Artificial Neural Networks

Start here:

[`artificial-neural-networks/`](artificial-neural-networks/)

The ANN track contains executable Jupyter notebooks covering:

- artificial neurons, weights, bias, and activation functions
- feed-forward neural networks
- representation learning
- loss functions and learning objectives
- gradients and gradient descent
- backpropagation from first principles
- training a multilayer perceptron from scratch
- PyTorch autograd and `nn.Module`
- common training failure modes such as saturation, vanishing/exploding gradients, and overfitting

Each notebook is designed as an interactive lab rather than a static chapter.

## Repository direction

Future tracks can extend the same visual-first approach to:

- convolutional neural networks
- recurrent neural networks
- attention and Transformers
- embeddings and representation learning
- optimization and regularization
- generative models
- modern deep-learning systems

## Running locally

```bash
git clone https://github.com/Akilankm/awesome-deep-learning-resource.git
cd awesome-deep-learning-resource
python -m venv .venv
source .venv/bin/activate
pip install -r artificial-neural-networks/requirements.txt
jupyter lab
```
