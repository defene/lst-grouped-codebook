from pathlib import Path
import gzip,hashlib,json
root=Path(__file__).resolve().parent
m=json.loads((root/'EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
for e in m['files']:
    # The project instructions were updated in October; retain the original
    # bytes for this frozen historical export's checksum verification.
    p = root / ('docs/historical_export_20261004/AGENTS.md' if e['path']=='AGENTS.md' else e['path'])
    b=p.read_bytes()
    assert hashlib.sha256(b).hexdigest()==e['sha256'],e['path']
    if e['encoding']=='gzip':b=gzip.decompress(b)
    assert hashlib.sha256(b).hexdigest()==e['source_sha256'],e['path']
print('PASS:',len(m['files']),'source files; exact source bytes verified')
