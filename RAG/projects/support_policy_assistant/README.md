# Capstone: fictional support-policy assistant

**Business brief.** Build a support assistant that answers standard shipping, returns, refund, account, privacy and developer API questions from a policy knowledge base. All policies are fictional learning data, not actual commerce advice.

**Inputs:** a JSONL policy corpus and one user question. **Outputs:** a response, source IDs, retrieval scores, generator mode, and a refusal when evidence is absent.

**Execution (from repo root):**

```bash
conda env create -f RAG/environment.yml
conda activate awesome-rag
python -m pytest -q RAG/tests
python -m RAG.src.cli --question "When can I get a refund?"
python -m RAG.src.cli --question "What is the CEO's birthday?"
# Optional: after starting Ollama and downloading llama3.2:1b
python -m RAG.src.cli --generator ollama --question "How long for a domestic order?"
```

**Delivery contract:** indexes load locally, no network required in baseline, document-level hit@k/MRR reported using a fixed gold set, inputs validated, result includes evidence provenance, out-of-domain query refuses, and no secrets are stored. For real deployment include access policy, effective dates, audit logs, citation verification, systematic LLM evaluation, monitoring, model selection and load tests.

## Classroom rubric (100 points)

- 20: Explain *RAG vs parametric training*; distinguish retrieval from generation.
- 20: Input schema, document IDs, chunk provenance and boundary experiment.
- 20: Retrieval experiments and quantitative hit@k/MRR analysis.
- 20: Grounded answer with citations; unknown-question refusal; injection discussion.
- 20: Reproducibility: Conda environment, saved notebook outputs, test/CI evidence, clear instructions.

## Extension exercises

1. Build a false document with the words 'ignore previous instructions'; assert it never becomes an instruction.
2. Add a date field and design stable precedence for a versioned shipping policy.
3. Compare TF-IDF with a pretrained dense encoder, keeping the evaluation set frozen.
4. Implement a FastAPI `/ask` endpoint, document request/response with OpenAPI, add authentication and rate limiting (optional; not needed for the offline curriculum).
5. Add response-grounding audits and show one false positive and one false negative.
