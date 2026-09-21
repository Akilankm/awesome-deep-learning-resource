# Sentence sentiment classification with RNNs

This project demonstrates the second major use of recurrence in the RNN track: **text sequences**.

## Problem
Given one natural-language review sentence, predict whether its sentiment is positive or negative.

## Dataset
UCI Sentiment Labelled Sentences, materialized locally at `RNN/data/sentence_sentiment_uci.csv`.

- 3,000 sentences
- Amazon, IMDb, Yelp
- balanced binary labels
- CC BY 4.0

## End-to-end lifecycle
```text
committed raw sentences
→ schema / duplicate / class checks
→ sentence-length EDA
→ domain-aware stratified train/validation/test
→ training-only vocabulary adaptation
→ TF-IDF logistic baseline
→ TextVectorization
→ Embedding + masking
→ SimpleRNN / LSTM / GRU
→ validation-only model selection
→ untouched test metrics
→ per-domain metrics
→ confusion matrix
→ high-confidence error analysis
→ save complete raw-string Keras model
→ reload
→ sentence inference
→ OOV / length / confidence monitoring ideas
```

Run the concept notebooks first:
1. `RNN/notebooks/15_text_sequences_for_rnn.ipynb`
2. `RNN/notebooks/16_sentence_classification_tensorflow.ipynb`

Then run:
`00_end_to_end_sentence_sentiment_rnn.ipynb`.
