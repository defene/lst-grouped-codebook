"""CPU-only remote file hashes and verified download of existing EMA weights."""
import hashlib
import json
from pathlib import Path
import modal

RUN='hlc-raw-mmdit-v-b256-fullval-epoch2-20261014-v2'
OUT=(Path(__file__).resolve().parents[2]/'output/efdiff_hlc_raw_mmdit_b256_v_20261014') if modal.is_local() else Path('/tmp/local-output-unused')
vol=modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
app=modal.App('hlc-mmdit-completed-artifact-audit')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


@app.function(image=modal.Image.debian_slim(python_version='3.12'),volumes={'/runs':vol},cpu=2,memory=1024,timeout=600)
def manifest():
    vol.reload()
    root=Path('/runs')/RUN/'train/v'
    complete=json.loads((root/'complete.json').read_text())
    assert complete['phase']=='complete' and complete['training_records']==5217544 and complete['updates']==30573
    result={}
    for name in ['best_ema.pt','final_ema.pt']:
        p=root/name
        result[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
    return result


@app.local_entrypoint()
def main():
    OUT.mkdir(exist_ok=True)
    hashes=manifest.remote()
    (OUT/'remote_weight_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for name,meta in hashes.items():
        dest=OUT/'train/v'/name
        if not (dest.exists() and dest.stat().st_size==meta['bytes'] and digest(dest)==meta['sha256']):
            tmp=dest.with_suffix('.download')
            with tmp.open('wb') as f:
                for block in vol.read_file(f'{RUN}/train/v/{name}'):f.write(block)
            assert tmp.stat().st_size==meta['bytes'] and digest(tmp)==meta['sha256']
            tmp.replace(dest)
        print(json.dumps(dict(file=name,verified=True,**meta)),flush=True)
    (OUT/'weight_download_verified.json').write_text(json.dumps(dict(files=hashes,remote_local_sha256_equal=True),indent=2)+'\n')
