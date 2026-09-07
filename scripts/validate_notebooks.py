from pathlib import Path
import ast
import nbformat
import sys

root = Path(__file__).resolve().parents[1]
notebooks = sorted((root / "notebooks").glob("*.ipynb"))
problems = []

if len(notebooks) != 2:
    problems.append(f"Expected exactly 2 notebooks, found {len(notebooks)}")

required = [
    "train", "validation", "test", "preprocess", "loss", "accuracy",
    "classification_report", "confusion", "save", "reload", "inference",
]

for path in notebooks:
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    combined = "\n".join(cell.source for cell in nb.cells).lower()
    for token in required:
        if token.lower() not in combined:
            problems.append(f"{path.name}: missing lifecycle concept {token}")
    markdown_chars = sum(len(c.source) for c in nb.cells if c.cell_type == "markdown")
    if markdown_chars < 2500:
        problems.append(f"{path.name}: documentation too thin ({markdown_chars} chars)")
    output_count = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        try:
            ast.parse(cell.source)
        except SyntaxError as exc:
            problems.append(f"{path.name} cell {i}: syntax error {exc}")
        if cell.execution_count is None:
            problems.append(f"{path.name} cell {i}: not executed")
        for out in cell.outputs:
            output_count += 1
            if out.output_type == "error":
                problems.append(f"{path.name} cell {i}: {out.ename}: {out.evalue}")
    if output_count == 0:
        problems.append(f"{path.name}: no persisted outputs")

if problems:
    print("\n".join(problems))
    sys.exit(1)

print("Validated both E2E notebooks: complete lifecycle, executed outputs, zero runtime errors.")
