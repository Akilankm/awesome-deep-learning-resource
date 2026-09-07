from pathlib import Path
import ast
import json
import sys
import nbformat

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO / 'notebooks'
FRAMEWORK = json.loads((REPO / 'track.json').read_text())['framework']

EXPECTED = [
    '00_environment_and_learning_map.ipynb',
    '01_mnist_data_and_preprocessing.ipynb',
    '02_neuron_tensor_math.ipynb',
    '03_forward_propagation_end_to_end.ipynb',
    '04_activations_initialization_gradient_flow.ipynb',
    '05_softmax_cross_entropy_loss.ipynb',
    '06_backpropagation_and_autodiff.ipynb',
    '07_end_to_end_training_loop.ipynb',
    '08_representation_learning_visualized.ipynb',
    '09_evaluation_error_analysis_calibration.ipynb',
    '10_inference_serialization_reload.ipynb',
    '11_production_business_monitoring.ipynb',
]

actual = sorted(p.name for p in NOTEBOOK_DIR.glob('*.ipynb'))
problems = []
if actual != EXPECTED:
    problems.append(f'Notebook curriculum mismatch. Expected {EXPECTED}; found {actual}')

for name in EXPECTED:
    path = NOTEBOOK_DIR / name
    if not path.exists():
        continue

    nb = nbformat.read(path, as_version=4)
    try:
        nbformat.validate(nb)
    except Exception as exc:
        problems.append(f'{name}: invalid notebook schema: {exc}')
        continue

    if not nb.cells or nb.cells[0].cell_type != 'markdown':
        problems.append(f'{name}: first cell must be Markdown')
    elif not nb.cells[0].source.lstrip().startswith('# '):
        problems.append(f'{name}: first Markdown cell must contain an H1 title')

    markdown = '\n'.join(c.source for c in nb.cells if c.cell_type == 'markdown')
    if len(markdown) < 900:
        problems.append(f'{name}: teaching explanation too thin ({len(markdown)} chars)')

    for forbidden in ['TODO', 'TBD', '\\[', '\\]', '\\(', '\\)']:
        if forbidden in markdown:
            problems.append(f'{name}: renderer-fragile or unfinished token: {forbidden}')

    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    if len(code_cells) < 2:
        problems.append(f'{name}: expected at least two executable code cells')

    output_count = 0
    image_output_count = 0
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue

        if not cell.source.strip():
            problems.append(f'{name} cell {index}: empty code cell')
            continue

        try:
            ast.parse(cell.source)
        except SyntaxError as exc:
            problems.append(f'{name} cell {index}: Python syntax error: {exc}')

        if cell.execution_count is None:
            problems.append(f'{name} cell {index}: not executed')

        for output in cell.outputs:
            output_count += 1
            if output.output_type == 'error':
                problems.append(f'{name} cell {index}: {output.ename}: {output.evalue}')
            if output.output_type in {'display_data', 'execute_result'}:
                data = getattr(output, 'data', {})
                if 'image/png' in data:
                    image_output_count += 1

    if output_count == 0:
        problems.append(f'{name}: no persisted outputs')

    if name in {
        '01_mnist_data_and_preprocessing.ipynb',
        '04_activations_initialization_gradient_flow.ipynb',
        '08_representation_learning_visualized.ipynb',
        '09_evaluation_error_analysis_calibration.ipynb',
        '11_production_business_monitoring.ipynb',
    } and image_output_count == 0:
        problems.append(f'{name}: expected at least one persisted image output')

training_text = (NOTEBOOK_DIR / '07_end_to_end_training_loop.ipynb').read_text() if (NOTEBOOK_DIR / '07_end_to_end_training_loop.ipynb').exists() else ''
backprop_text = (NOTEBOOK_DIR / '06_backpropagation_and_autodiff.ipynb').read_text() if (NOTEBOOK_DIR / '06_backpropagation_and_autodiff.ipynb').exists() else ''
for token in ['train_one_epoch', 'evaluate', 'optimizer']:
    if token not in training_text:
        problems.append(f'07_end_to_end_training_loop.ipynb: missing required concept {token}')
for token in ['finite_difference', 'dL_dw']:
    if token not in backprop_text:
        problems.append(f'06_backpropagation_and_autodiff.ipynb: missing required manual-gradient concept {token}')

if FRAMEWORK == 'pytorch':
    for token in ['nn.Module', 'DataLoader', 'loss.backward()', 'optimizer.step()']:
        if token not in training_text:
            problems.append(f'PyTorch training notebook missing {token}')
elif FRAMEWORK == 'tensorflow':
    for token in ['tf.keras.Model', 'tf.data.Dataset', 'tf.GradientTape', 'apply_gradients']:
        if token not in training_text:
            problems.append(f'TensorFlow training notebook missing {token}')
else:
    problems.append(f'Unsupported framework in track.json: {FRAMEWORK}')

if problems:
    print('\n'.join(problems))
    sys.exit(1)

print(
    f'Validated {len(EXPECTED)} executed {FRAMEWORK} notebooks: '
    'schema, teaching depth, Python syntax, execution counts, persisted outputs, '
    'visual outputs, framework training semantics, and zero execution errors.'
)
