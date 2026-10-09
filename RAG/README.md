# RAG — Retrieval-Augmented Generation, from first principles to an end-to-end project

> **Branch:** `rag-end-to-end-curriculum`. This track is isolated from the ANN, CNN, RNN and NLP material on `master`.

## Student learning contract

Understand **what changes, why it changes, and how to prove it**. Each notebook provides a concept, a mental model, runnable cells, a change→effect challenge, a visible output, and interpretation. The fictional support-policy dataset is stored locally and needs **no API key, vector database, or download** for the core lessons.

**Important distinction:** The reproducible default is TF-IDF **sparse lexical retrieval** followed by an **extractive evidence preview**. That is a didactic *retrieval-and-grounding baseline*, not a claim that a template is a trained LLM. The optional local Ollama step performs genuine LLM-based RAG; its output is nondeterministic and is NOT included as a supposedly executed offline output.

## System map

```text
OFFLINE INDEX BUILD                                PER USER QUESTION
JSONL policies (fictional)                        question
    -> validate document IDs                           |
    -> chunk with overlap                         TF-IDF query vector
    -> sparse TF-IDF vectors                            |
    -> searchable chunk index -------------------- cosine ranking
                                                        |
                                                top-k with source IDs
                                                        |
                                +-----------------------+------------------+
                                |                                          |
                        offline evidence preview                 build grounded prompt
                        (deterministic baseline)                        |
                                                                     Ollama LLM
                                                                (optional local model)
                                                                        |
                                                            grounded answer + citations
                                                        -> evaluation, abstention, logs
```

### Environment / run

```bash
git clone https://github.com/Akilankm/awesome-deep-learning-resource.git
cd awesome-deep-learning-resource
git switch rag-end-to-end-curriculum
conda env create -f RAG/environment.yml
conda activate awesome-rag
python -m ipykernel install --user --name awesome-rag --display-name "Python (awesome-rag)"
jupyter lab
```

Open Jupyter **from repository root** and select `Python (awesome-rag)`. Notebook imports use `RAG.src.core` and need that working directory. For local validation:

```bash
python -m pytest -q RAG/tests
python RAG/scripts/execute_notebooks.py
python RAG/scripts/verify_notebooks.py
python -m RAG.src.cli --question "What is the free developer API rate limit?"
```

### Curriculum (open in this order)

| Lesson | Concept and experiment | Inspect |
|---|---|---|
| [00 – RAG system map](notebooks/00_system_map.ipynb) | parametric vs external knowledge; query/answer/data contracts | source metadata |
| [01 – ingest and chunk](notebooks/01_ingest_and_chunk.ipynb) | text size/overlap; effects on evidence boundaries | chunk counts and provenance |
| [02 – vectorize and retrieve](notebooks/02_retrieve_and_rank.ipynb) | TF-IDF weights, cosine similarity, top-k | ranked chunks and scores |
| [03 – grounding and generation](notebooks/03_grounded_generation.ipynb) | citations, prompt boundary, abstention, optional Ollama | prompt + preview |
| [04 – evaluation and failure modes](notebooks/04_evaluation.ipynb) | hit@k, MRR, nonanswerable abstention, leakage | metric dictionary |
| [05 – capstone pipeline](notebooks/05_capstone.ipynb) | end-to-end support assistant, CLI, review rubric | answers and sources |

### Follow-on reading

- [RAG engineering and architecture](docs/architecture.md) — indexing, embedding alternatives, vector stores, BM25, hybrid search, reranking, query rewrite, RAG vs fine-tuning
- [Evaluation, risks, and production](docs/evaluation_and_operations.md) — retrieval + generation metrics, tracing, prompt injection, ACLs, PII, observability
- [Capstone requirements](projects/support_policy_assistant/README.md) — acceptance criteria and exercises
- [Source data](data/support_policies.jsonl) and [retrieval gold set](data/evaluation_questions.jsonl)
- [Offline baseline / optional generator implementation](src/core.py)

### Genuine LLM generation: optional Ollama integration

After installing [Ollama](https://ollama.com/), run `ollama pull llama3.2:1b`, then make sure its local server is running. No external LLM API key is required; the model must be downloaded once. On a machine with the model:

```bash
python -m RAG.src.cli --generator ollama --question "How long can I return an unused item?"
```

**Before deployment:** add citation verification, robust semantic/hybrid retrieval, document ACL enforcement, redaction, reindexing/version control, monitoring and human evaluation. Neither the evidence preview nor a local small model is a production-ready helpdesk.

### CI and executed notebook policy

The branch's [GitHub Actions workflow](../.github/workflows/rag-notebooks.yml) runs the tests, executes all notebooks under a fresh Conda Python kernel, validates there are real output cells and no exceptions, and commits the **executed notebook results** back onto this branch when permissions allow. Code cells in each published notebook retain their output; Markdown and normal text results render directly on GitHub. Optional Ollama inference is deliberately **not** executed on CI.
