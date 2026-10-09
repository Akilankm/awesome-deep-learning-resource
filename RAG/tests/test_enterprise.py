"""Offline LangChain/Chroma integration regression: permission boundaries first."""
from datetime import date
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from RAG.enterprise.langchain_bfsi import (
    ABSTAIN, OfflineLexicalEmbeddings, Principal, ask_policy, authorized_search,
    build_generation_chain, build_store, chunk_policies, evaluate_access_and_retrieval,
    get_embeddings, load_evaluation, load_policies, scoped_filter,
)

TODAY = date(2026, 10, 9)


@pytest.fixture(scope="module")
def store():
    return build_store(load_policies(), OfflineLexicalEmbeddings())


def test_langchain_document_loading_and_chunk_metadata():
    docs = load_policies()
    assert len(docs) == 9
    chunks = chunk_policies(docs, chunk_size=160, chunk_overlap=30)
    assert len(chunks) >= len(docs)
    assert all(c.metadata.get("policy_id") and c.metadata.get("chunk_id") for c in chunks)


def test_reject_unknown_principal_and_embedding_provider():
    with pytest.raises(ValueError):
        Principal("northstar-bank", "anonymous")
    with pytest.raises(ValueError):
        get_embeddings("invalid")


def test_access_filter_is_explicit():
    scope = scoped_filter(Principal("northstar-bank", "employee"), TODAY)
    assert len(scope["$and"]) == 4


def test_employee_sees_only_public_docs(store):
    hits = authorized_search(store, "identity verification government ID and address proof",
                             Principal("northstar-bank", "employee"), TODAY)
    assert hits
    assert all(d.metadata["access_group"] == "all" for d, _ in hits)
    assert hits[0][0].metadata["policy_id"] == "KYC-2026"


def test_department_isolation(store):
    hits = authorized_search(store, "AML suspicious alert triage four business hours",
                             Principal("northstar-bank", "risk"), TODAY, max_l2_distance=2.1)
    assert all(d.metadata["access_group"] != "compliance" for d, _ in hits)
    compliance_hits = authorized_search(
        store, "AML suspicious alert triage four business hours",
        Principal("northstar-bank", "compliance"), TODAY)
    assert any(d.metadata["policy_id"] == "AML-2026" for d, _ in compliance_hits)


def test_tenant_isolation_even_on_exact_other_tenant_query(store):
    hits = authorized_search(store, "HarborCase Harbor Identity Desk KYC",
                             Principal("northstar-bank", "admin"), TODAY, max_l2_distance=2.1)
    assert all(d.metadata["tenant_id"] == "northstar-bank" for d, _ in hits)


def test_deprecated_policy_excluded(store):
    hits = authorized_search(store, "onboarding without address proof arrived later",
                             Principal("northstar-bank", "admin"), TODAY, max_l2_distance=2.1)
    assert not any(d.metadata["policy_id"] == "KYC-2024" for d, _ in hits)


def test_authorized_preview_has_citation(store):
    result = ask_policy(store, "identity verification government ID address proof",
                        Principal("northstar-bank", "employee"), TODAY)
    assert result["mode"] == "offline-evidence"
    assert "[KYC-2026]" in result["answer"]
    assert any(s["policy_id"] == "KYC-2026" for s in result["sources"])


def test_unknown_question_refuses(store):
    result = ask_policy(store, "What is the CEO's favorite basketball team?",
                        Principal("northstar-bank", "employee"), TODAY)
    assert result["answer"] == ABSTAIN


def test_langchain_prompt_runnable_with_fake_model():
    chain = build_generation_chain(FakeListChatModel(responses=["Documented procedure [KYC-2026]"]))
    output = chain.invoke({"question": "What is KYC?", "context": "[KYC-2026] Verify ID."})
    assert "[KYC-2026]" in output


def test_generated_citation_guard(store):
    principal = Principal("northstar-bank", "employee")
    good = ask_policy(store, "identity verification government ID address proof", principal, TODAY,
                      llm=FakeListChatModel(responses=["Verify identity documents [KYC-2026]"]))
    assert good["mode"] == "llm-generated"
    bad = ask_policy(store, "identity verification government ID address proof", principal, TODAY,
                     llm=FakeListChatModel(responses=["Approved by fictional policy [SECRET-2026]"]))
    assert bad["mode"] == "citation-rejected"
    assert bad["answer"] == ABSTAIN


def test_evaluation_contract(store):
    metrics = evaluate_access_and_retrieval(store, load_evaluation(), TODAY, k=3)
    assert metrics["answerable_n"] == 8
    assert metrics["restricted_or_unknown_n"] == 3
    assert 0 <= metrics["hit_at_k"] <= 1
    assert 0 <= metrics["mrr_at_k"] <= 1
    assert 0 <= metrics["negative_refusal_rate"] <= 1
