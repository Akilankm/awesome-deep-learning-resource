# Dataset Card — UCI Sentiment Labelled Sentences

## Purpose
Sentence-level binary sentiment classification for teaching recurrent neural networks on textual sequences.

## Source
UCI Machine Learning Repository — **Sentiment Labelled Sentences** (dataset ID 331).

- DOI: 10.24432/C57604
- Original creators: Dimitrios Kotzias et al.
- License: CC BY 4.0
- Original domains: Amazon product reviews, IMDb movie reviews, Yelp restaurant reviews
- Original size: 3,000 labelled sentences
- Label semantics: `1 = positive`, `0 = negative`
- Missing values: none reported by UCI

## Materialized file
`sentence_sentiment_uci.csv`

Columns:
- `source`: amazon / imdb / yelp
- `text`: one review sentence
- `label`: 0 negative, 1 positive

The committed CSV is a local materialization of the original UCI dataset so the RNN/NLP notebooks can execute without runtime dataset downloads.

## Teaching value
This dataset is intentionally compact and balanced, making it suitable for showing:

`raw sentence → tokenization → integer sequence → padding/masking → embedding → recurrent state → sentence representation → binary classifier`

It also allows per-source error analysis to show that the same label task appears across different language domains.

## Attribution
Kotzias, D. (2015). *Sentiment Labelled Sentences* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C57604

License: Creative Commons Attribution 4.0 International (CC BY 4.0).
