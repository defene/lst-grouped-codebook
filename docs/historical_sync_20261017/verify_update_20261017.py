"""Verify the source/report update without models, datasets, network, or training."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
manifest=json.loads((root/'UPDATE_MANIFEST_20261017.json').read_text(encoding='utf-8'))
for rel,entry in manifest['files'].items():
    path=root/rel
    assert path.resolve().is_relative_to(root),rel
    assert path.suffix.lower() not in {'.pt','.pth','.ckpt','.safetensors','.npz','.npy','.eoz','.zip','.sqlite'},rel
    raw=path.read_bytes()
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],rel
    if path.suffix=='.py':ast.parse(raw,filename=rel)
    if path.suffix=='.json':json.loads(raw)
for item in json.loads((root/'reports/local_vq_ema_20261017/evidence.json').read_text()):
    raw=(root/item['receipt']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==item['receipt_sha256']
    receipt=json.loads(raw)
    assert receipt['checkpoint_sha256']==item['selection']['checkpoint_sha256']
    pixel=next(c for c in receipt['cells'] if c['profile']['kind']=='pixel10')
    assert pixel['raw']['pixel_weighted_rmse']==item['selection']['score']
    assert all(c['unique_records']==11684 and c['views']==2 for c in receipt['cells'])
for folder in ('ema_local_full_20261015_attempt01','ema_local_k512_full_20261015_attempt01'):
    assert (root/'outputs'/folder/'STOP').exists()
print('PASS: source/report hashes, syntax, validation receipts and STOP markers;',len(manifest['files']),'files')
