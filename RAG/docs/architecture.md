# RAG architecture: the why behind each stage

## The central equation (conceptual)

`answer = Generator(question, Retriever(question, corpus))`. The generator is **not** trained on the user's documents at query time. It receives retrieved evidence in its context window. Unlike training/fine-tuning, index changes can be reflected after reindexing without updating the LLM weights.

**Training:** changes model parameters. **RAG:** changes context at inference. **Fine-tuning + RAG:** complementary, not interchangeable.

## Indexing path

1. **Acquire documents:** permission, source, language, updated time and version. Our demonstration uses 10 fictional JSONL policy records.
2. **Clean and validate:** enforce schema, unique IDs, source URLs/IDs. In real systems handle PDF layout, scans, tables, encoding and deduplication; do not flatten a table without retaining row/column meaning.
3. **Chunk:** a chunk should answer a question without depending on missing neighboring text. Small chunks increase specificity but can lose context; large chunks preserve context but dilute ranking and consume tokens. Our lesson uses word chunks; production may require section-aware/sentence-aware splitters and token counting.
4. **Represent:** TF-IDF converts each chunk into sparse weighted word n-gram features. It is interpretable and offline; it does not understand paraphrases like a pretrained sentence-embedding model. Neural embeddings map to dense vectors capturing learned semantic similarity. Choose a model appropriate to language/domain and record its version.
5. **Index:** the matrix here is in memory. At scale: FAISS, Qdrant, pgvector, Milvus or hosted search, including metadata filtering and access controls. Ensure source-to-chunk lineage.
6. **Retrieve:** cosine similarity between query and chunks, sorted highest first. `top-k` determines context breadth. More is not automatically better. BM25 is a useful lexical alternative; hybrid sparse+dense search and reciprocal rank fusion often improve coverage. Rerank candidate passages with a cross-encoder for accuracy/latency tradeoffs.

## Query path

```text
question -> normalization/rewrite? -> retrieve candidates
         -> tenant/ACL filter -> rerank/dedupe -> trim to context budget
         -> quote as UNTRUSTED EVIDENCE -> LLM -> citation verifier -> response
```

Avoid answer generation when no trusted passage meets an evidence policy. Our threshold is only a **retrieval score threshold**, not a calibrated probability of truth.

## Similarity math

With TF-IDF vectors `q` and `d`, cosine similarity is:

`cos(q,d) = (q·d) / (||q|| * ||d||)`.

`TfidfVectorizer(norm="l2")` normalizes vectors so the dot product matches cosine similarity. Word n-grams emphasize matching phrases; unknown vocabulary terms have zero weight. Replacing TF-IDF with pretrained embeddings changes what 'similarity' means and typically trades interpretability for semantic recall.

## Engineering architecture and scales

- **Prototype**: Python + local JSONL + TF-IDF + Ollama.
- **Pilot**: ingestion jobs, document versioning, ACL filter, embeddings, vector store, reranker, evaluation suite, HTTP service, traces.
- **Production**: incremental index versioning, queue/retry/dead letter, data residency, PII handling, prompt-injection defenses, cost/token budgets, SLA/SLO dashboards, drift checks, feedback/human escalation, security review.

## Design decisions students must defend

- Why not fine-tune on constantly changing policies?
- When could a relevant chunk be retrieved but an answer still be wrong?
- What is the difference between similarity score, answer correctness and factual faithfulness?
- How do permissions apply before, during and after retrieval?
- What metrics change if chunk size or `top-k` changes?

For the runnable pathway see [RAG README](../README.md).
