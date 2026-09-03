from pathlib import Path
import argparse, os, shutil
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

parser=argparse.ArgumentParser()
parser.add_argument('--in-place',action='store_true')
parser.add_argument('--output-dir',default='executed-notebooks')
args=parser.parse_args()
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('PYTHONHASHSEED','0')
root=Path('artificial-neural-networks')
out=root if args.in_place else Path(args.output_dir)
out.mkdir(parents=True,exist_ok=True)
for path in sorted(root.glob('*.ipynb')):
    print(f'Executing {path}')
    nb=nbformat.read(path,as_version=4)
    for cell in nb.cells:
        if cell.cell_type=='code': cell.execution_count=None;cell.outputs=[]
    ep=ExecutePreprocessor(timeout=240,kernel_name='python3',allow_errors=False)
    ep.preprocess(nb,{'metadata':{'path':str(root)}})
    target=path if args.in_place else out/path.name
    nbformat.write(nb,target)
print('Notebook execution completed successfully.')
