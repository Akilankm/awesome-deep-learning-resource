"""Fail if published notebooks are unexecuted or contain error outputs."""
from __future__ import annotations
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[2]
paths = sorted((ROOT / "RAG" / "notebooks").glob("*.ipynb"))
if len(paths) < 6:
    raise SystemExit(f"Expected at least 6 notebooks, found {len(paths)}")
for path in paths:
    nb = nbformat.read(path, as_version=4)
    codes = [c for c in nb.cells if c.cell_type == "code"]
    if not codes or any(c.execution_count is None for c in codes):
        raise SystemExit(f"NOT EXECUTED: {path}")
    outputs = [o for c in codes for o in c.outputs]
    if not outputs or any(o.output_type == "error" for o in outputs):
        raise SystemExit(f"MISSING OUTPUT OR ERROR: {path}")
    print(f"VERIFIED {path.name} | {len(codes)} executed code cells | {len(outputs)} outputs")
