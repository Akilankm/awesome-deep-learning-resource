from pathlib import Path
import base64, io, tarfile
parts = sorted((Path(__file__).parent / 'bootstrap_final_parts').glob('*.txt'))
if len(parts) != 6:
    raise SystemExit(f'Expected 6 payload parts, found {len(parts)}')
payload = ''.join(p.read_text().strip() for p in parts)
root = Path(__file__).resolve().parents[1]
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode='r:gz') as tf:
    tf.extractall(root)
print('Materialized final canonical branch contents.')
