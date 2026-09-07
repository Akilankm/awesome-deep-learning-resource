# ANN on MNIST — PyTorch

This branch is a **self-contained executable ANN course**. It uses the same official MNIST dataset, split contract, architecture and evaluation philosophy as the other implementation branches so learners can compare implementation mechanics rather than different experiments.

## Reproduce from a fresh clone
```bash
conda env create -f environment.yml
conda activate ann-mnist-pytorch
jupyter lab
```
Then open `notebooks/00_environment_and_learning_map.ipynb` and run notebooks in numerical order.

For a non-interactive verification identical to CI:
```bash
make verify
```

## Data contract
- official MNIST archive
- development subset sampled only from official training data
- 5,000 train / 1,000 validation, stratified and deterministic
- 1,000 final test examples sampled only from official test archive
- `float32`, 784 features, normalized `[0,1]`
- primary MLP: 784 → 64 → 10
- representation lab: 784 → 2 → 10

## Curriculum
1. [`00_environment_and_learning_map.ipynb`](notebooks/00_environment_and_learning_map.ipynb)
2. [`01_mnist_data_provenance_eda_splits.ipynb`](notebooks/01_mnist_data_provenance_eda_splits.ipynb)
3. [`02_preprocessing_batching_data_contract.ipynb`](notebooks/02_preprocessing_batching_data_contract.ipynb)
4. [`03_neuron_linear_algebra_architecture.ipynb`](notebooks/03_neuron_linear_algebra_architecture.ipynb)
5. [`04_activations_initialization_gradient_flow.ipynb`](notebooks/04_activations_initialization_gradient_flow.ipynb)
6. [`05_forward_logits_softmax.ipynb`](notebooks/05_forward_logits_softmax.ipynb)
7. [`06_loss_backprop_gradient_check.ipynb`](notebooks/06_loss_backprop_gradient_check.ipynb)
8. [`07_optimizers_end_to_end_training.ipynb`](notebooks/07_optimizers_end_to_end_training.ipynb)
9. [`08_validation_hyperparameters_diagnostics.ipynb`](notebooks/08_validation_hyperparameters_diagnostics.ipynb)
10. [`09_representation_learning_visualized.ipynb`](notebooks/09_representation_learning_visualized.ipynb)
11. [`10_test_evaluation_error_calibration.ipynb`](notebooks/10_test_evaluation_error_calibration.ipynb)
12. [`11_inference_serialization_performance.ipynb`](notebooks/11_inference_serialization_performance.ipynb)
13. [`12_production_monitoring_drift_retraining_business.ipynb`](notebooks/12_production_monitoring_drift_retraining_business.ipynb)

## Quality contract
Every code cell has a preceding technical explanation. CI clears all outputs, executes every notebook from zero, checks syntax/formatting, requires persisted outputs and zero error outputs, and commits the executed notebooks back to the branch. The learner-facing notebooks are the canonical source; scripts only verify them.
