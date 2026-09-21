from __future__ import annotations
import json, sys
from pathlib import Path
import nbformat

roots = [Path("RNN/notebooks"), Path("RNN/projects"), Path("CNN/notebooks"), Path("CNN/projects")]
failures = []
summary = []
for root in roots:
    for path in sorted(root.rglob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        unexecuted = [i for i, c in enumerate(code_cells) if c.execution_count is None]
        errors = [o for c in code_cells for o in c.get("outputs", []) if o.output_type == "error"]
        output_count = sum(bool(c.get("outputs")) for c in code_cells)
        ok = bool(code_cells) and not unexecuted and not errors and output_count > 0
        summary.append({"notebook": str(path), "code_cells": len(code_cells), "output_cells": output_count, "ok": ok})
        if not ok:
            failures.append({"path": str(path), "unexecuted": unexecuted, "errors": len(errors), "outputs": output_count})
Path("notebook_execution_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
if failures:
    print("FAILURES", json.dumps(failures, indent=2), file=sys.stderr)
    raise SystemExit(1)
