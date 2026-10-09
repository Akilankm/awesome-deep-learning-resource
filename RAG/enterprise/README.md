# Enterprise case: internal banking risk & compliance policy copilot

**This is a realistic enterprise architecture with FICTIONAL policies and simulated principals**, not an authenticated production banking service. It demonstrates the code path, test gates and trade-offs students should understand before handling private data.

## Business brief

A bank has changing internal policies for KYC onboarding, AML investigation, sanctions screening, credit-risk exceptions, model-risk governance and incident response. Employees must only retrieve their tenant's current documents permitted for their department. The assistant should cite the **policy ID and version** it used and refuse unsupported questions rather than guessing.

**Scenario:** a risk analyst asks "Who approves credit-risk limit overrides?" → search filtered to the Northstar tenant, risk-authorized documents and active versions → retrieve [RISK-2026] → optionally send evidence to a local Ollama chat model → cited answer. The same analyst must **not** receive the restricted AML compliance playbook.

## Real LangChain components

| Stage | Component | Why |
|---|---|---|
| Document representation | `langchain_core.documents.Document` | full text + typed provenance metadata |
| Chunking | `langchain_text_splitters.RecursiveCharacterTextSplitter` | position-aware source splits |
| Embeddings | `langchain_ollama.OllamaEmbeddings` with `nomic-embed-text` | REAL neural semantic vectorization (optional download) |
| Offline CI embedding | `langchain_core.embeddings.Embeddings` adapter using lexical hashing | deterministic tests; **NOT neural/semantic embeddings** |
| Vector DB | `langchain_chroma.Chroma` | in-memory demo or persistent local index |
| Secure filtering | `Chroma.similarity_search_with_score(filter=...)` | scope enforced *before* nearest-neighbor retrieval |
| Prompt | `langchain_core.prompts.ChatPromptTemplate` | explicit evidence trust boundary |
| Generation | `langchain_ollama.ChatOllama(model="llama3.2:1b")` | REAL local LLM (optional download) |
| Orchestration | LCEL `prompt | llm | StrOutputParser()` | composable prompt / model / parsing |
| Tests | pytest, `FakeListChatModel` | tests prompt wiring and citation rejection **without** claiming real LLM inference |

See the [LangChain Chroma](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma) and [ChatOllama](https://docs.langchain.com/oss/python/integrations/chat/ollama) integrations.

## Two execution modes (be precise!)

**Mode 1 — offline reproducible:** default uses hashing-vector lexical vectors in Chroma + an **extractive evidence preview**. All notebooks, tests and security filters execute without model downloads, Ollama or API keys. This verifies retrieval logic and orchestration, but does not demonstrate learned semantic similarity or LLM reasoning.

**Mode 2 — genuine neural RAG:** start Ollama and pull both models. The same pipeline switches to REAL embedding inference and REAL LLM generation through LangChain.

```bash
# Prerequisite: install Ollama from https://ollama.com/
ollama pull nomic-embed-text
ollama pull llama3.2:1b
# If Ollama is not already running: ollama serve

# From the project root, in another terminal:
conda activate awesome-rag
python -m RAG.enterprise.cli \
  --embedding ollama --generator ollama \
  --tenant northstar-bank --role risk \
  --question "Who approves credit-risk limit overrides?"
```

Note: the `--role` and `--tenant` flags **simulate an authenticated request**. In real systems, these must be derived solely from verified SSO/JWT identity claims by a server-side authorization service. Never accept role or tenant as user-controlled authorization inputs.

## Setup and assessment

```bash
conda env create -f RAG/environment.yml
conda activate awesome-rag
python -m pytest -q RAG/tests
python -m RAG.enterprise.cli --question "What evidence is needed to activate an account?"
python RAG/scripts/execute_notebooks.py
python RAG/scripts/verify_notebooks.py
```

Read [06 enterprise access](../notebooks/06_bfsi_access_control.ipynb) and [07 LangChain generation](../notebooks/07_bfsi_langchain_end_to_end.ipynb) after the core track.

## What the design controls

- **Tenant isolation:** every Chroma search includes `tenant_id`.
- **Role groups:** department-specific `access_group` enforced *in the store query*.
- **Policy freshness:** effective and expiry numeric dates enforced *in the store query*.
- **Provenance:** policy IDs, versions, chunk IDs, source URIs and retrieval distances returned.
- **Prompt injection:** source text is untrusted; prompts are not a substitute for external security.
- **Hallucination:** no evidence → abstain; unknown/foreign policy IDs in generated text → refuse. Citation syntax is **not sufficient to prove grounding**.
- **Testing:** unit/integration tests of access boundaries and LCEL with a mocked chat model, and a fixed retrieval evaluation set.

## What must be added to call this production-ready

Verified SSO/OIDC + a trusted, policy-enforced authorization service, encrypted tenant-specific vector storage, ingestion ACL controls, document revocation propagation, versioned reindexing, calibrated semantic distance thresholds, source-span claim verification, adversarial prompt-injection testing, PII redaction, regulatory review, human escalation, structured audit events, drift and latency/cost monitoring, and end-to-end load tests. No production data should enter the fictional public repository.

**Do not interpret this synthetic policy data as actual RBI/AML regulatory advice or statutory deadlines.**
