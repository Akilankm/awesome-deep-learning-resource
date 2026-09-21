from __future__ import annotations
import argparse, json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

def execute(path: Path, repo_root: Path) -> dict:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=1200, kernel_name="python3", allow_errors=False,
                            resources={"metadata": {"path": str(repo_root)}})
    client.execute()
    nbformat.write(nb, path)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    return {
        "notebook": str(path.relative_to(repo_root)),
        "code_cells": len(code_cells),
        "executed": sum(c.execution_count is not None for c in code_cells),
        "output_cells": sum(bool(c.get("outputs")) for c in code_cells),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--report", required=True)
    args = p.parse_args()
    root = Path.cwd().resolve()
    files = []
    for raw in args.paths:
        path = (root / raw).resolve()
        if path.is_dir():
            files.extend(sorted(path.rglob("*.ipynb")))
        elif path.suffix == ".ipynb":
            files.append(path)
    report = []
    for f in files:
        print(f"EXECUTING {f.relative_to(root)}", flush=True)
        report.append(execute(f, root))
    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
