from pathlib import Path
import gzip,hashlib,json
root=Path(__file__).resolve().parent
m=json.loads((root/'EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
for e in m['files']:
    b=(root/e['path']).read_bytes()
    assert hashlib.sha256(b).hexdigest()==e['sha256'],e['path']
    if e['encoding']=='gzip':b=gzip.decompress(b)
    assert hashlib.sha256(b).hexdigest()==e['source_sha256'],e['path']
print('PASS:',len(m['files']),'source files; exact source bytes verified')
