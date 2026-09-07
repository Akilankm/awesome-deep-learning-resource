from pathlib import Path
import ast, sys, nbformat
files=sorted(Path('.').glob('*.ipynb'))
if len(files)!=6:
    raise SystemExit(f'Expected 6 notebooks, found {len(files)}')
problems=[]
for p in files:
    nb=nbformat.read(p,as_version=4)
    nbformat.validate(nb)
    if not nb.cells or nb.cells[0].cell_type!='markdown' or not nb.cells[0].source.lstrip().startswith('# '):
        problems.append(f'{p}: missing H1')
    outputs=0
    markdown='\n'.join(c.source for c in nb.cells if c.cell_type=='markdown')
    if len(markdown)<500:
        problems.append(f'{p}: explanation too thin')
    for bad in ['TODO','TBD','\\[','\\(','\\]','\\)']:
        if bad in markdown:
            problems.append(f'{p}: renderer-fragile token {bad}')
    for i,c in enumerate(nb.cells):
        if c.cell_type=='code':
            try:
                ast.parse(c.source)
            except SyntaxError as e:
                problems.append(f'{p} cell {i}: syntax {e}')
            if c.execution_count is None:
                problems.append(f'{p} cell {i}: not executed')
            for o in c.outputs:
                outputs+=1
                if o.output_type=='error':
                    problems.append(f'{p} cell {i}: {o.ename}: {o.evalue}')
    if outputs==0:
        problems.append(f'{p}: no persisted outputs')
if problems:
    print('\n'.join(problems))
    sys.exit(1)
print(f'Validated {len(files)} executed notebooks: structure, math markdown, code syntax, execution counts, persisted outputs, zero errors.')
