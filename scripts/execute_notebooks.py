from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

root = Path(__file__).resolve().parents[1]
notebooks = sorted((root / "notebooks").glob("*.ipynb"))
if len(notebooks) != 2:
    raise SystemExit(f"Expected 2 notebooks, found {len(notebooks)}")

for path in notebooks:
    print(f"Executing {path.name}")
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3", allow_errors=False)
    ep.preprocess(nb, {"metadata": {"path": str(root)}})
    nbformat.write(nb, path)
    print(f"Executed {path.name}")
