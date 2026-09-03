from pathlib import Path

parts_dir = Path(__file__).with_name('bootstrap_parts')
parts = sorted(parts_dir.glob('part*.txt'))
if not parts:
    raise SystemExit('No ANN bootstrap payload parts found.')
source = ''.join(p.read_text(encoding='utf-8') for p in parts)
exec(compile(source, 'bootstrap_ann_notebooks_embedded.py', 'exec'), {'__name__': '__main__'})
