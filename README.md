# Awesome Deep Learning Resource

A visual, executable learning repository for deep learning and modern NLP.

## Learning tracks

### Artificial Neural Networks

`master` contains the **framework-neutral ANN textbook and visual knowledge base**. Hands-on ANN execution lives in three independent branches:

- [`ann-mnist-manual`](https://github.com/Akilankm/awesome-deep-learning-resource/tree/ann-mnist-manual) — NumPy / first principles
- [`ann-mnist-pytorch`](https://github.com/Akilankm/awesome-deep-learning-resource/tree/ann-mnist-pytorch) — production-style PyTorch
- [`ann-mnist-tensorflow`](https://github.com/Akilankm/awesome-deep-learning-resource/tree/ann-mnist-tensorflow) — production-style TensorFlow / Keras

### Natural Language Processing

The [`nlp/`](nlp/) track contains **30 executed notebooks** covering the complete path from raw text processing to transformers, semantic retrieval, end-to-end model development, production monitoring and responsible NLP. The notebooks use consistent mental models, change→effect reasoning and selective open-license reference diagrams where a mature architecture visual is genuinely useful.

Start here:

- [NLP curriculum and run guide](nlp/README.md)
- [NLP teaching standard](nlp/TEACHING_STANDARD.md)
- [NLP visual references and licenses](nlp/VISUAL_REFERENCES.md)
- [Conda environment](nlp/environment.yml)
- [Local NLP datasets](nlp/data/)
- [Optional pretrained NLP requirements](nlp/requirements-optional-transformers.txt)
- [Flagship end-to-end NLP project](nlp/projects/customer_support_intelligence/README.md)

Create the NLP environment from the repository root:

```bash
conda env create -f nlp/environment.yml
conda activate awesome-nlp
python -m ipykernel install --user --name awesome-nlp --display-name "Python (awesome-nlp)"
jupyter lab
```

The environment contains the core scientific/NLP/deep-learning stack plus Jupyter, Plotly/widgets and the Hugging Face ecosystem, so the notebooks can be rerun locally and extended beyond the committed offline examples.

## ANN recommended learning sequence

Read the conceptual chapter, then run the matching notebook in one implementation branch. The three ANN implementation branches use the same official MNIST problem and data split contract so differences reflect implementation, not different experiments.

## ANN documentation map
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
- [Interactive Learning: GitHub Static View vs VS Code/Jupyter](docs/20_interactive_learning.md)

## Visual learning
- [ANN lifecycle](visual_learning/ann_lifecycle.svg)
- [Train / validation / test](visual_learning/train_val_test.svg)
- [Forward vs backward](visual_learning/forward_backward.svg)
- [Representation learning](visual_learning/representation_learning.svg)
- [Production feedback loop](visual_learning/production_loop.svg)

## Static on GitHub, interactive after cloning

The learning artifacts deliberately support two modes. GitHub keeps executed notebooks readable as **static learning artifacts** with explanations, code, metrics and persisted outputs. After cloning into VS Code/Jupyter and running the appropriate environment, interactive Plotly/widget labs can be explored with hover, zoom, pan and point-level inspection.

See the [ANN interactive learning guide](docs/20_interactive_learning.md) and the [NLP run guide](nlp/README.md).

## Learning objective

Understand deep-learning and NLP systems end to end:

```text
data
  → representation
  → model
  → optimization
  → validation
  → test
  → inference
  → monitoring
  → retraining
  → business action
```

<!-- RNN_CNN_TRACKS -->
## Recurrent Neural Networks (TensorFlow/Keras)
The [`RNN/`](RNN/) track contains executed concept notebooks and a complete Jena-climate forecasting project. It covers sequence tensors, windowing, recurrence, BPTT, gradient pathologies, LSTM, GRU, architecture patterns, evaluation, serialization and monitoring.

- [RNN run guide](RNN/README.md)
- [RNN Conda environment](RNN/environment.yml)
- [RNN notebooks](RNN/notebooks/)
- [End-to-end Jena forecasting project](RNN/projects/jena_climate_forecasting/)

## Convolutional Neural Networks (TensorFlow/Keras)
The [`CNN/`](CNN/) track contains executed concept notebooks and a complete Fashion-MNIST project. It covers image tensors, manual convolution, kernels, padding/stride, pooling, Conv2D, regularization, augmentation, transfer learning, Grad-CAM, calibration, serialization and monitoring.

- [CNN run guide](CNN/README.md)
- [CNN Conda environment](CNN/environment.yml)
- [CNN notebooks](CNN/notebooks/)
- [End-to-end Fashion-MNIST project](CNN/projects/fashion_mnist_classifier/)
<!-- /RNN_CNN_TRACKS -->
