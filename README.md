# Awesome Deep Learning Resource — Artificial Neural Networks

`master` is the **framework-neutral ANN textbook and visual knowledge base**. It contains no framework-specific training implementation. Hands-on execution lives in exactly three independent branches:

- [`ann-mnist-manual`](https://github.com/Akilankm/awesome-deep-learning-resource/tree/ann-mnist-manual) — NumPy / first principles
- [`ann-mnist-pytorch`](https://github.com/Akilankm/awesome-deep-learning-resource/tree/ann-mnist-pytorch) — production-style PyTorch
- [`ann-mnist-tensorflow`](https://github.com/Akilankm/awesome-deep-learning-resource/tree/ann-mnist-tensorflow) — production-style TensorFlow / Keras

## Recommended learning sequence
Read the conceptual chapter, then run the matching notebook in one implementation branch. The three implementation branches use the same official MNIST problem and data split contract so differences reflect implementation, not different experiments.

## Documentation map
- [ANN Big Picture](docs/00_ann_big_picture.md)
- [Data, EDA, Leakage, and Train/Validation/Test](docs/01_data_and_splits.md)
- [Preprocessing, Batching, and Data Contracts](docs/02_preprocessing_batching.md)
- [Neuron and Linear Algebra](docs/03_neuron_linear_algebra.md)
- [Activations and Initialization](docs/04_activations_initialization.md)
- [Forward Propagation, Logits, and Softmax](docs/05_forward_logits_softmax.md)
- [Loss, Chain Rule, and Backpropagation](docs/06_loss_backprop.md)
- [Optimization and Training Loops](docs/07_optimization_training.md)
- [Validation, Hyperparameters, and Failure Modes](docs/08_validation_diagnostics.md)
- [Representation Learning](docs/09_representation_learning.md)
- [Final Test, Error Analysis, and Calibration](docs/10_evaluation_calibration.md)
- [Serialization, Inference, and Performance](docs/11_inference_serialization.md)
- [Monitoring, Drift, and Retraining](docs/12_monitoring_drift.md)
- [Business and Production Mapping](docs/13_business_mapping.md)
- [Manual vs PyTorch vs TensorFlow Crosswalk](docs/14_framework_crosswalk.md)
- [Production-Grade ANN Checklist](docs/15_production_checklist.md)
- [ANN Math Reference](docs/16_math_reference.md)
- [Tensor and Shape Reference](docs/17_shape_reference.md)
- [Training Debug Playbook](docs/18_training_debug_playbook.md)
- [Production ANN Operating Model](docs/19_production_operating_model.md)

## Visual learning
- [ANN lifecycle](visual_learning/ann_lifecycle.svg)
- [Train / validation / test](visual_learning/train_val_test.svg)
- [Forward vs backward](visual_learning/forward_backward.svg)
- [Representation learning](visual_learning/representation_learning.svg)
- [Production feedback loop](visual_learning/production_loop.svg)

## Final learning objective
Understand ANN as an end-to-end system: **data → representation → forward → loss → backprop → optimization → validation → test → inference → monitoring → retraining → business action**.
