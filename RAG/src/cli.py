"""Run from repository root: python -m RAG.src.cli --question '...'"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .core import Retriever, chunk_documents, load_documents, run_query

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fictional support-policy RAG learning project")
    parser.add_argument("--question", required=True)
    parser.add_argument("--generator", choices=["offline", "ollama"], default="offline")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.12)
    args = parser.parse_args()
    chunks = chunk_documents(load_documents(ROOT / "data" / "support_policies.jsonl"))
    result = run_query(args.question, Retriever(chunks), k=args.k,
                       threshold=args.threshold, generator=args.generator)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
