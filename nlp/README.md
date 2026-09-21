# NLP — End-to-End Natural Language Processing Learning Track

This folder is a **30-notebook, executable NLP curriculum** that moves from raw text to production NLP systems.

The learning contract is:

```text
concept
  ↓
small inspectable example
  ↓
standard-library / canonical-library implementation
  ↓
intermediate representation
  ↓
rendered result
  ↓
failure modes
  ↓
production implication
```

The committed notebooks contain executed outputs so they remain useful when viewed directly on GitHub. After cloning, run them with the Conda environment below to reproduce and extend the experiments.

## Create the Conda environment

From the repository root:

```bash
conda env create -f nlp/environment.yml
conda activate awesome-nlp
python -m ipykernel install --user --name awesome-nlp --display-name "Python (awesome-nlp)"
```

Then launch either:

```bash
jupyter lab
```

or open the repository in VS Code and select the **Python (awesome-nlp)** kernel.

### Updating an existing environment

```bash
conda env update -f nlp/environment.yml --prune
conda activate awesome-nlp
```

## What the environment contains

The environment includes the full core stack used across the notebooks:

- NumPy, pandas, SciPy, scikit-learn, joblib and PyArrow
- NLTK, spaCy and Gensim
- PyTorch
- Matplotlib and Plotly
- JupyterLab, Notebook, ipykernel, nbformat, nbclient and nbconvert
- ipywidgets for interactive notebook controls
- Hugging Face Transformers, Sentence-Transformers, Datasets, Evaluate and Accelerate
- Kaleido for Plotly static export

The core curriculum is deliberately **offline-reproducible**. Modern pretrained checkpoints are an optional extension and may require internet access the first time model weights/tokenizers are downloaded.

No external NLTK corpus or spaCy model download is required for the core notebooks: the relevant examples use local/rule-based resources when necessary.

## Local datasets

The notebooks use small transparent datasets committed under `nlp/data/`:

- `sentiment_reviews.csv` — three-class sentiment examples
- `customer_support.csv` — four-class support-ticket routing dataset

These are educational datasets designed to make every stage inspectable. They are not benchmark datasets and should not be used to claim real-world model quality.

## Curriculum

| # | Notebook | Focus |
|---:|---|---|
| 00 | `00_nlp_learning_map.ipynb` | NLP lifecycle |
| 01 | `01_text_language_fundamentals.ipynb` | corpus, token, vocabulary, OOV |
| 02 | `02_text_cleaning_normalization.ipynb` | cleaning and normalization |
| 03 | `03_tokenization.ipynb` | word/subword tokenization |
| 04 | `04_morphology_stemming_lemmatization.ipynb` | morphology |
| 05 | `05_ngrams_collocations.ipynb` | n-grams |
| 06 | `06_pos_tagging.ipynb` | POS tagging |
| 07 | `07_named_entity_recognition.ipynb` | NER |
| 08 | `08_syntax_dependency_parsing.ipynb` | syntax and dependencies |
| 09 | `09_bag_of_words.ipynb` | sparse counts |
| 10 | `10_tfidf.ipynb` | TF-IDF |
| 11 | `11_distributional_embeddings.ipynb` | co-occurrence + SVD embeddings |
| 12 | `12_word2vec_gensim.ipynb` | Word2Vec |
| 13 | `13_sentence_document_embeddings.ipynb` | sentence/document vectors |
| 14 | `14_classical_text_classification.ipynb` | leakage-safe classical ML |
| 15 | `15_sentiment_analysis.ipynb` | sentiment analysis |
| 16 | `16_topic_modeling.ipynb` | topic modeling |
| 17 | `17_sequence_models_rnn_lstm_gru.ipynb` | recurrent models |
| 18 | `18_attention_mechanism.ipynb` | Q/K/V attention |
| 19 | `19_transformer_encoder.ipynb` | transformer encoder |
| 20 | `20_contextual_embeddings_bert_mechanics.ipynb` | contextual embeddings |
| 21 | `21_semantic_search.ipynb` | retrieval |
| 22 | `22_information_extraction_qa_summarization.ipynb` | IE, extractive QA, summarization |
| 23 | `23_evaluation_error_analysis.ipynb` | metrics and error analysis |
| 24 | `24_end_to_end_nlp_pipeline.ipynb` | complete ML lifecycle |
| 25 | `25_production_nlp_monitoring.ipynb` | monitoring and drift |
| 26 | `26_language_models_text_generation.ipynb` | language modeling/generation |
| 27 | `27_seq2seq_encoder_decoder.ipynb` | encoder-decoder |
| 28 | `28_modern_nlp_task_heads.ipynb` | pretrained task heads |
| 29 | `29_responsible_robust_nlp.ipynb` | privacy, robustness, responsible NLP |

## Flagship project

The concept track is complemented by one deep project under [`projects/customer_support_intelligence/`](projects/customer_support_intelligence/).

The project uses **one executed notebook** and a committed ready-made educational dataset to cover the complete lifecycle from receiving raw data through EDA, hypothesis formation, leakage prevention, splitting, baselines, model selection, final evaluation, slice/error analysis, entity extraction, retrieval, serialization, inference, robustness, monitoring and retraining policy. The dataset construction recipe is intentionally not exposed in the student-facing notebook.

Start here:

- [Project overview](projects/customer_support_intelligence/README.md)
- [End-to-end executed notebook](projects/customer_support_intelligence/customer_support_intelligence_end_to_end.ipynb)
- [Dataset dictionary](projects/customer_support_intelligence/DATA_DICTIONARY.md)
- [Reference execution metrics](projects/customer_support_intelligence/artifacts/)

Run it from the repository root after activating `awesome-nlp`:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=300 \
  nlp/projects/customer_support_intelligence/customer_support_intelligence_end_to_end.ipynb
```

## Re-run all notebooks

From the repository root, after activating `awesome-nlp`:

```bash
for nb in nlp/*.ipynb; do
  echo "Running $nb"
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=300 "$nb"
done
```

This command executes notebooks in lexical order and persists the fresh outputs back into each notebook.

## Optional pretrained NLP packages

If you are not using the Conda environment and only want the modern pretrained stack:

```bash
pip install -r nlp/requirements-optional-transformers.txt
```

## Reproducibility contract

- Random seeds are fixed where applicable.
- Core examples do not depend on remote APIs.
- Train/validation/test preprocessing is encapsulated in pipelines where appropriate.
- Final test evaluation is kept separate from model selection in the end-to-end project.
- Serialized inference examples save preprocessing and model logic together.
- Production notebooks cover inference contracts, drift, PII and robustness—not only model training.
