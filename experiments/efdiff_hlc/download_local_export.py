"""Stream exported inference models from Modal and verify server-side SHA256."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import modal

PREFIX = 'hlc-synthetic-pilot-16384-512-v1/local_export_v1'
OUT = Path(__file__).resolve().parents[2]/'output/efdiff_hlc_pilot_20260915/models'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    volume = modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
    manifest = json.loads(b''.join(volume.read_file(f'{PREFIX}/manifest.json')))
    def fetch(item):
        name,expected = item
        target = OUT/name
        if target.exists() and target.stat().st_size==expected['bytes']:
            with target.open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()==expected['sha256']:
                    return name
        temp = target.with_suffix('.download')
        n = last = 0
        def progress(*args, **kwargs):
            nonlocal n,last
            n += kwargs.get('advance',0)
            if n-last >= 64*1024*1024:
                print(f'{name}: {n/1e9:.2f}/{expected["bytes"]/1e9:.2f} GB',flush=True)
                last = n
        with temp.open('wb') as stream:
            count = volume._read_file_into_fileobj(f'{PREFIX}/{name}',stream,progress_cb=progress,concurrency=4)
        with temp.open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
        assert count==expected['bytes'] and digest==expected['sha256'],name
        temp.replace(target)
        print('Verified',name,flush=True)
        return name
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch,manifest['files'].items()))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('All inference weights downloaded and SHA256 verified',flush=True)


if __name__=='__main__':
    main()
