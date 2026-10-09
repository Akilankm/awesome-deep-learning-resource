"""Enterprise RAG CLI. Run from the repo root with python -m RAG.enterprise.cli."""
from __future__ import annotations
import argparse
import json
from datetime import date
from .langchain_bfsi import (Principal, ask_policy, build_store, get_embeddings,
                             load_policies, make_ollama_chat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fictional BFSI policy copilot")
    parser.add_argument("--question", required=True)
    parser.add_argument("--tenant", choices=["northstar-bank", "harbor-finance"], default="northstar-bank")
    parser.add_argument("--role", choices=["employee", "risk", "compliance", "admin"], default="employee")
    parser.add_argument("--embedding", choices=["offline", "ollama"], default="offline")
    parser.add_argument("--generator", choices=["offline", "ollama"], default="offline")
    parser.add_argument("--as-of", default="2026-10-09", help="YYYY-MM-DD; fixed teaching date by default")
    args = parser.parse_args()
    # IMPORTANT: CLI role/tenant are classroom simulations, NOT authenticated claims.
    store = build_store(load_policies(), get_embeddings(args.embedding))
    llm = make_ollama_chat() if args.generator == "ollama" else None
    result = ask_policy(store, args.question, Principal(args.tenant, args.role),
                        date.fromisoformat(args.as_of), llm=llm)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
