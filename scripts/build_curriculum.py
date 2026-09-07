from pathlib import Path
import base64, lzma

parts = sorted((Path(__file__).parent / 'curriculum_payload').glob('part-*.txt'))
if len(parts) != 6:
    raise SystemExit(f'Expected 6 curriculum payload parts, found {len(parts)}')
payload = ''.join(p.read_text().strip() for p in parts)
source = lzma.decompress(base64.b64decode(payload))
exec(compile(source, 'build_curriculum_source.py', 'exec'))
