"""Execute and persist every RAG teaching notebook with real Jupyter outputs.

Usage from repository root: python RAG/scripts/execute_notebooks.py
"""
from __future__ import annotations
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
RAG = ROOT / "RAG"
NOTEBOOKS = sorted((RAG / "notebooks").glob("*.ipynb"))


def main() -> None:
    if not NOTEBOOKS:
        raise SystemExit("No notebooks found")
    report = []
    for path in NOTEBOOKS:
        nb = nbformat.read(path, as_version=4)
        # Setting the notebook execution directory to the repo root makes
        # RAG.src imports and relative corpus paths work on CI and locally.
        client = NotebookClient(nb, kernel_name="python3", timeout=180,
                                allow_errors=False, resources={"metadata": {"path": str(ROOT)}})
        result = client.execute()
        outputs = 0
        for cell in result.cells:
            if cell.cell_type != "code":
                continue
            cell.metadata.pop("execution", None)  # no volatile wall-clock data in git
            outputs += len(cell.get("outputs", []))
        nbformat.write(result, path)
        report.append({"notebook": path.name, "code_cells": sum(c.cell_type == "code" for c in result.cells),
                       "outputs": outputs, "status": "passed"})
        print(f"PASS {path.name} ({outputs} saved outputs)", flush=True)
    report_path = RAG / "reports" / "execution_report.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved report to {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
