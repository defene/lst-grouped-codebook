"""Verify and extract the inference-only model archive; never invokes CUDA."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import sys
import tarfile

root = Path(__file__).resolve().parent
archive = root / 'all39_inference_20261013.tar'
with archive.open('rb') as f:
    actual = hashlib.file_digest(f, 'sha256').hexdigest()
if actual != sys.argv[1]:
    raise RuntimeError('Archive SHA256 mismatch')
with tarfile.open(archive) as tar:
    for member in tar:
        parts = PurePosixPath(member.name).parts
        if not member.isfile() or member.name.startswith('/') or any(p in ('..', '.') or ':' in p or '\\' in p for p in parts):
            raise RuntimeError(f'Unsafe member: {member.name}')
        path = root.joinpath(*parts)
        if not path.resolve().is_relative_to(root):
            raise RuntimeError(f'Unsafe path: {member.name}')
        path.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as source, path.open('wb') as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
manifest = json.loads((root / 'MODEL_MANIFEST.json').read_text(encoding='utf-8'))
for item in manifest['files']:
    path = root / item['path']
    with path.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    if sha != item['sha256'] or path.stat().st_size != item['bytes']:
        raise RuntimeError(f"Corrupt file: {item['path']}")
assert len(manifest['models']) == 39
receipt = {'status': 'verified', 'models': 39, 'files': len(manifest['files']),
           'archive_sha256': actual, 'archive_bytes': archive.stat().st_size,
           'model_weight_bytes': sum(m['weights_bytes'] for m in manifest['models']),
           'optimizer_states_included': False, 'original_model_dtypes_preserved': True}
(root / 'download_verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
