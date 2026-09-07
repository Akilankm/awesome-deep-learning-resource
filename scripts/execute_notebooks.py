from pathlib import Path
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO / 'notebooks'
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob('*.ipynb'))

if not NOTEBOOKS:
    raise SystemExit('No notebooks found under notebooks/.')

# Use Jupyter's inline backend so figures are persisted inside .ipynb outputs.
os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'

for path in NOTEBOOKS:
    print(f'Executing {path.name} ...')
    nb = nbformat.read(path, as_version=4)

    # CI must prove reproducibility. Never rely on previously committed outputs.
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.execution_count = None
            cell.outputs = []

    executor = ExecutePreprocessor(
        timeout=1200,
        kernel_name='python3',
        allow_errors=False,
    )
    executor.preprocess(nb, {'metadata': {'path': str(NOTEBOOK_DIR)}})
    nbformat.write(nb, path)

print(f'All {len(NOTEBOOKS)} notebooks executed successfully from cleared state.')
