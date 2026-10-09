"""Unit tests for ingestion, chunk boundaries, retrieval, refusal and citation plumbing."""
from pathlib import Path
import pytest
from RAG.src.core import (
    Document, Retriever, answer_offline, build_prompt, chunk_documents,
    evaluate_retrieval, load_documents, load_questions, run_query,
)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture
def retriever():
    return Retriever(chunk_documents(load_documents(DATA / "support_policies.jsonl")))


def test_load_and_unique_ids():
    docs = load_documents(DATA / "support_policies.jsonl")
    assert len(docs) == 10
    assert len({d.doc_id for d in docs}) == len(docs)


def test_chunk_overlap_and_boundaries():
    doc = Document("a", "Example", "one two three four five six seven", "internal")
    chunks = chunk_documents([doc], size=4, overlap=2)
    assert [(c.start_word, c.end_word) for c in chunks] == [(0, 4), (2, 6), (4, 7)]
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]


@pytest.mark.parametrize("size,overlap", [(0, 0), (3, 3), (5, -1)])
def test_invalid_chunks(size, overlap):
    with pytest.raises(ValueError):
        chunk_documents(load_documents(DATA / "support_policies.jsonl"), size, overlap)


def test_retrieval_source(retriever):
    hits = retriever.search("password reset link", k=2)
    assert hits and hits[0].chunk.doc_id == "account"
    assert all(h.score > 0 for h in hits)


def test_refusal_on_out_of_domain_question(retriever):
    r = run_query("What is the CEO's birthday?", retriever)
    assert r["mode"] == "extractive-preview"
    assert not r["sources"]
    assert "do not know" in r["answer"]


def test_prompt_contains_trust_boundary(retriever):
    prompt = build_prompt("When can I return?", retriever.search("return unused items"))
    assert "untrusted" in prompt and "<evidence>" in prompt and "[returns" in prompt


def test_evaluation_ranges(retriever):
    metrics = evaluate_retrieval(retriever, load_questions(DATA / "evaluation_questions.jsonl"))
    assert metrics["answerable_n"] == 10
    assert metrics["unanswerable_n"] == 2
    assert 0 <= metrics["hit_at_k"] <= 1
    assert 0 <= metrics["mrr_at_k"] <= 1
    assert 0 <= metrics["unanswerable_abstention"] <= 1


def test_bad_generator(retriever):
    with pytest.raises(ValueError):
        run_query("test", retriever, generator="imaginary")
