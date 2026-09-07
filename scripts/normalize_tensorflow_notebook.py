from pathlib import Path
import ast
import json

path = Path("notebooks/02_mnist_e2e_tensorflow.ipynb")
nb = json.loads(path.read_text())

replacements = {
    "SEEED": "SEED",
}

changed = False
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    for old, new in replacements.items():
        if old in source:
            source = source.replace(old, new)
            changed = True
    ast.parse(source)
    cell["source"] = source.splitlines(keepends=True)

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"TensorFlow notebook normalized; changed={changed}; all code cells parse.")
