from pathlib import Path
import argparse
import nbformat

parser=argparse.ArgumentParser()
parser.add_argument('--require-executed',action='store_true')
args=parser.parse_args()
root=Path('artificial-neural-networks')
paths=sorted(root.glob('*.ipynb'))
if len(paths)<12: raise SystemExit(f'Expected at least 12 ANN notebooks, found {len(paths)}')
errors=[]
for path in paths:
    nb=nbformat.read(path,as_version=4)
    try: nbformat.validate(nb)
    except Exception as e: errors.append(f'{path}: nbformat invalid: {e}')
    md='\n'.join(c.source for c in nb.cells if c.cell_type=='markdown')
    if not md.lstrip().startswith('# '): errors.append(f'{path}: missing H1 title')
    if 'TODO' in md or 'TBD' in md: errors.append(f'{path}: contains TODO/TBD')
    if '\\(' in md or '\\[' in md: errors.append(f'{path}: renderer-fragile math delimiter')
    code=[c for c in nb.cells if c.cell_type=='code']
    if not code: errors.append(f'{path}: no code cells')
    for index,c in enumerate(code):
        try: compile(c.source,f'{path}:code-cell-{index}','exec')
        except SyntaxError as e: errors.append(f'{path}: code cell {index} does not compile: {e}')
    if len(md) < 500: errors.append(f'{path}: explanatory Markdown is unexpectedly thin')
    if args.require_executed:
        if any(c.execution_count is None for c in code): errors.append(f'{path}: unexecuted code cell')
        if any(any(o.output_type=='error' for o in c.get('outputs',[])) for c in code): errors.append(f'{path}: error output committed')
        if sum(len(c.get('outputs',[])) for c in code)==0: errors.append(f'{path}: no rendered outputs')
if errors: raise SystemExit('\n'.join(errors))
print(f'Validated {len(paths)} ANN notebooks. require_executed={args.require_executed}')
