from pathlib import Path
import os, nbformat
from nbconvert.preprocessors import ExecutePreprocessor
os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
for path in sorted(Path('.').glob('*.ipynb')):
    print('Executing', path)
    nb=nbformat.read(path,as_version=4)
    for c in nb.cells:
        if c.cell_type=='code':
            c.execution_count=None
            c.outputs=[]
    ep=ExecutePreprocessor(timeout=900,kernel_name='python3',allow_errors=False)
    ep.preprocess(nb,{'metadata':{'path':'.'}})
    nbformat.write(nb,path)
print('All notebooks executed successfully.')
