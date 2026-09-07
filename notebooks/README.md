# Rendered MNIST ANN notebooks

This directory contains the committed, executed learning notebooks for this orphan branch. GitHub can render the Markdown, equations, code cells, plots, tables, metrics, and persisted outputs directly in the browser.

Expected learning sequence:

1. `00_mnist_learning_map.ipynb` — end-to-end learning map and experiment contract.
2. `01_pixels_to_neurons.ipynb` — MNIST pixels, flattening, normalization, neurons, weights, bias, activations.
3. `02_forward_propagation.ipynb` — complete forward propagation through the ANN.
4. `03_loss_gradients_backprop.ipynb` — loss, gradients, chain rule, and backpropagation.
5. `04_training_and_representation_learning.ipynb` — optimizer updates, training dynamics, and changing hidden representations.
6. `05_inference_debugging_business.ipynb` — inference, mistakes, debugging signals, and business interpretation.

The GitHub Action clears and re-executes every code cell, validates that execution counts and outputs are persisted with zero notebook errors, and commits the rendered `.ipynb` files back into this directory.
