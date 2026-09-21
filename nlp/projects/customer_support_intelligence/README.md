# Customer Support Intelligence — End-to-End NLP Project

This is the flagship project for the NLP track.

## Start here

Open and run:

`customer_support_intelligence_end_to_end.ipynb`

The notebook is designed to be independently executable from a clean clone using the repository Conda environment.

## Run

From the repository root:

```bash
conda env create -f nlp/environment.yml
conda activate awesome-nlp
python -m ipykernel install --user --name awesome-nlp --display-name "Python (awesome-nlp)"
jupyter lab
```

Or execute non-interactively:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=300 \
  nlp/projects/customer_support_intelligence/customer_support_intelligence_end_to_end.ipynb
```

## What the notebook does

The notebook implements the complete lifecycle:

```text
educational dataset generation
  → data contract / validation
  → EDA
  → duplicate + leakage analysis
  → normalization
  → stratified train / validation / test
  → majority baseline
  → Bag-of-Words + Naive Bayes
  → word TF-IDF + Logistic Regression
  → word + character TF-IDF
  → validation-only hyperparameter tuning
  → final untouched test evaluation
  → confidence / abstention
  → error analysis
  → entity extraction
  → knowledge-base retrieval
  → integrated inference
  → model serialization + reload verification
  → robustness tests
  → monitoring / drift
  → retraining / promotion policy
```

## Educational dataset

There is no hidden download step. The notebook deterministically generates a 473-row support-ticket dataset plus a 12-article knowledge base and materializes them under `data/raw/`.

The generated data deliberately includes class imbalance, duplicates, typos, noisy punctuation, mixed-intent examples, amounts, order IDs, emails and dates so that the quality, leakage, error-analysis and information-extraction sections are meaningful.

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Reproducibility

The project uses `RANDOM_SEED = 42`. The validated reference run was executed repeatedly with identical metrics and test predictions.

Reference classifier metrics:

- validation macro-F1: **0.9877**
- final untouched test accuracy: **0.8989**
- final untouched test macro-F1: **0.8984**
- final untouched test weighted-F1: **0.8999**

The reference run contains **9 deliberately useful test errors** for error analysis rather than an unrealistically perfect synthetic benchmark.

Retrieval reference results:

| Retriever | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| Word TF-IDF | 0.6404 | 0.8539 | 0.7623 |
| LSA dense | 0.6517 | 0.8427 | 0.7670 |

Entity extraction achieves exact-match F1 of 1.0 on this deliberately structured educational entity schema; the notebook explains why this should not be interpreted as real-world NER performance.

## Generated artifacts

A local execution generates:

- raw/interim/processed CSV data,
- `intent_classifier.joblib`,
- `metrics.json`,
- model comparison and tuning tables,
- classification report,
- test predictions,
- error-analysis CSV,
- entity extraction metrics,
- retrieval metrics/predictions,
- top features,
- monitoring snapshot,
- PNG + SVG figures,
- model metadata,
- run manifest.

The committed notebook contains rendered results, and selected lightweight SVGs/metrics are versioned under `artifacts/` for GitHub viewing.
