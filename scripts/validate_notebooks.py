from pathlib import Path
import ast
import sys
import nbformat

root = Path(__file__).resolve().parents[1]
notebooks = sorted((root / "notebooks").glob("[0-9][0-9]_*.ipynb"))
expected = 13
problems = []

if len(notebooks) != expected:
    problems.append(f"Expected {expected} notebooks, found {len(notebooks)}")

for path in notebooks:
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    markdown_chars = sum(len(c.source) for c in nb.cells if c.cell_type == "markdown")
    if markdown_chars < 1400:
        problems.append(f"{path.name}: explanation too thin ({markdown_chars} chars)")

    previous = None
    output_count = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            if (
                previous is None
                or previous.cell_type != "markdown"
                or len(previous.source.strip()) < 120
            ):
                problems.append(
                    f"{path.name} cell {i}: code lacks a substantial preceding explanation"
                )
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
        previous = cell

    if output_count == 0:
        problems.append(f"{path.name}: no persisted outputs")

    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown").lower()
    for token in ["business", "production"]:
        if token not in text:
            problems.append(f"{path.name}: missing {token} interpretation")

if problems:
    print("\n".join(problems))
    sys.exit(1)

print(
    f"Validated {len(notebooks)} executed notebooks: documentation, syntax, "
    "outputs and zero runtime errors."
)
