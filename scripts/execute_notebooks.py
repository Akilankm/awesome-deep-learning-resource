from pathlib import Path
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
root = Path(__file__).resolve().parents[1]
notebooks = sorted((root / "notebooks").glob("[0-9][0-9]_*.ipynb"))
if not notebooks:
    raise SystemExit("No curriculum notebooks found.")
for path in notebooks:
    print("Executing", path.name)
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    ExecutePreprocessor(
        timeout=1200, kernel_name="python3", allow_errors=False
    ).preprocess(nb, {"metadata": {"path": str(path.parent)}})
    nbformat.write(nb, path)
print(f"Executed {len(notebooks)} notebooks successfully.")
