# Recurrent Neural Networks — TensorFlow/Keras

A complete, executed learning path from sequence tensors and recurrence to SimpleRNN, LSTM, GRU, **time-series forecasting**, **NLP sentence classification**, inference and monitoring.

## Run
```bash
conda env create -f RNN/environment.yml
conda activate awesome-rnn
jupyter lab
```

## Core RNN learning path
Start with `notebooks/00_rnn_learning_map.ipynb` and progress numerically through the recurrent foundations:

1. sequence data and tensor shapes
2. time-series windowing
3. recurrence and hidden state
4. TensorFlow SimpleRNN
5. unrolling
6. backpropagation through time
7. vanishing/exploding gradients
8. LSTM
9. GRU
10. sequence architectures
11. stacked/bidirectional RNNs
12. regularization/training
13. forecasting evaluation
14. inference and monitoring

## RNN for NLP — sentence classification

After the recurrent foundations, continue with:

- `notebooks/15_text_sequences_for_rnn.ipynb` — sentence → token IDs → padding/masking → embeddings → hidden states.
- `notebooks/16_sentence_classification_tensorflow.ipynb` — raw sentence → TensorFlow recurrent classifier → sentiment probability.

This path makes the mapping explicit:

```text
word sequence
   ↓
TextVectorization
   ↓
integer token sequence
   ↓
Embedding
   ↓
SimpleRNN / LSTM / GRU
   ↓
sentence representation
   ↓
classification head
```

## End-to-end projects

### 1. Time-series forecasting
`projects/jena_climate_forecasting/00_end_to_end_jena_forecasting.ipynb`

Real weather observations → chronological split → windowing → baseline → SimpleRNN/LSTM/GRU → validation selection → untouched test → serialization/inference → drift checks.

### 2. NLP sentence sentiment classification
`projects/sentence_sentiment_classification/00_end_to_end_sentence_sentiment_rnn.ipynb`

UCI Sentiment Labelled Sentences → data validation/EDA → domain-aware stratified split → TF-IDF baseline → training-only vocabulary → embedding/masking → SimpleRNN/LSTM/GRU → validation selection → untouched test → per-domain/error analysis → saved raw-string Keras model → reload/inference → monitoring.

Dataset documentation:
`data/SENTENCE_SENTIMENT_DATASET_CARD.md`

## Execution contract

Every committed notebook is executed by CI. Static outputs render on GitHub; rerunning locally in Jupyter/VS Code enables experimentation.

The teaching standard is:

`intuition → shapes → mathematics → TensorFlow implementation → observed result → interpretation → failure modes → production connection`.
