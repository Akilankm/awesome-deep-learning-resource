# Customer Support Intelligence — End-to-End NLP Project

This is the flagship project for the NLP track. It is intentionally **one deep project**, not a collection of shallow demos.

## Start here

Open:

`customer_support_intelligence_end_to_end.ipynb`

The notebook treats the committed CSV files as **ready-made business inputs**. The dataset-construction recipe is intentionally not part of the student-facing codebase: learners must discover patterns through EDA, form hypotheses, validate them, build baselines, inspect errors and justify each modeling decision.

The provenance is still explicit: this is an **educational synthetic/curated dataset**, not a public benchmark or production dataset.

## Run

From the repository root:

```bash
conda env create -f nlp/environment.yml
conda activate awesome-nlp
python -m ipykernel install --user --name awesome-nlp --display-name "Python (awesome-nlp)"
jupyter lab
```

Or execute end to end:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=300 \
  nlp/projects/customer_support_intelligence/customer_support_intelligence_end_to_end.ipynb
```

## Ready-made inputs

```text
data/raw/
├── support_tickets.csv   # 473 received support tickets
└── knowledge_base.csv    # 12 support articles
```

The notebook does **not** generate or overwrite these raw inputs.

During execution it creates inspectable derived data:

```text
data/interim/cleaned_tickets.csv
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

## How to study the project

Treat each stage as an engineering decision:

1. form a hypothesis from the data,
2. predict what a representation/model change should do,
3. run the experiment,
4. compare expected vs observed behavior,
5. debug the layer that actually failed.

The notebook uses compact decision tables instead of decorative diagrams for the project workflow.

## End-to-end lifecycle

```text
business problem
  → receive raw data
  → data contract
  → EDA
  → explicit hypotheses
  → normalization
  → duplicate/leakage control
  → stratified train/validation/test
  → majority baseline
  → Bag-of-Words + Naive Bayes
  → word TF-IDF + logistic regression
  → word + character TF-IDF
  → validation-only tuning
  → untouched final test
  → slice analysis
  → confidence / abstention
  → error analysis
  → learned-feature inspection
  → entity extraction
  → knowledge retrieval
  → integrated inference
  → serialization + reload verification
  → robustness tests
  → monitoring / drift
  → retraining / promotion policy
```

For major stages the notebook explains **what, why, when, how and when not to use the technique**.

## Reference execution

The refactored ready-made-data notebook was executed end to end before commit.

Classifier:
- validation macro-F1: **0.9582**
- final test accuracy: **0.8913**
- final test macro-F1: **0.8896**
- final weighted-F1: **0.8908**
- held-out errors: **10**

The imperfect score is intentional and useful educationally: the error analysis shows that the reference mistakes concentrate in multi-intent messages rather than presenting an unrealistically perfect synthetic benchmark.

Retrieval:

| Retriever | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| Word TF-IDF | 0.4674 | 0.8370 | 0.6562 |
| LSA dense | 0.4022 | 0.7500 | 0.6067 |

Entity extraction receives perfect exact-format scores on the controlled educational schema; the notebook explicitly explains why that is **not evidence of real-world NER performance**.

## Artifacts generated on rerun

- `intent_classifier.joblib`
- `metrics.json`
- `model_metadata.json`
- `model_comparison.csv`
- `hyperparameter_tuning.csv`
- `classification_report.csv`
- `test_predictions.csv`
- `errors.csv`
- `top_features_by_class.csv`
- `entity_extraction_metrics.csv`
- `retrieval_metrics.csv`
- `retrieval_predictions.csv`
- `monitoring_snapshot.csv`
- `run_manifest.json`

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for field semantics.
