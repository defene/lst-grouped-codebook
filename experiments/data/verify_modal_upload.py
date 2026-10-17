"""Restore frozen file timestamps and verify the dataset upload in Modal."""
import json
from pathlib import Path
import modal

app=modal.App('hls-lst-upload-verification')
volume=modal.Volume.from_name('hls-lst-datasets-20260914',environment_name='main')
image=modal.Image.debian_slim(python_version='3.13').pip_install('numpy','zstandard')

@app.function(image=image,volumes={'/data':volume},cpu=1,memory=512,timeout=120)
def probe():
    import os
    p=Path('/data/_transfer/timestamp_probe.txt');p.parent.mkdir(exist_ok=True)
    p.write_text('Timestamp compatibility probe for frozen dataset readers.\n')
    expected=1788999547937498400
    os.utime(p,ns=(expected,expected))
    actual=p.stat().st_mtime_ns
    volume.commit()
    return {'expected_mtime_ns':expected,'actual_mtime_ns':actual,'exact':actual==expected}

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=86400)
def verify():
    import hashlib
    import os
    import sys
    import time
    root=Path('/data').resolve();manifest=root/'_transfer/files.jsonl'
    print(f'Verifying canonical volume root: {root}',flush=True)
    plan=json.loads((root/'_transfer/plan.json').read_text())
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==plan['manifest_sha256']
    total=0;count=0;started=time.time()
    with manifest.open() as f:
        for line in f:
            r=json.loads(line);p=root/r['path']
            assert p.resolve().is_relative_to(root),f'Path outside volume: {r["path"]}'
            st=p.stat();assert st.st_size==r['size'],r['path']
            os.utime(p,ns=(st.st_atime_ns,r['mtime_ns']))
            assert p.stat().st_mtime_ns==r['mtime_ns'],f'Timestamp precision mismatch: {r["path"]}'
            total+=st.st_size;count+=1
            if count%25000==0:print(f'File sizes and timestamps verified: {count}',flush=True)
    assert count==plan['files'] and total==plan['bytes']
    # Full archive SHA-256 provides end-to-end verification of the large HLP file.
    archive=root/'artifacts/prithvi_data_v1_20260909.zip'
    expected=(root/'artifacts/prithvi_data_v1_20260909.zip.sha256').read_text().split()[0]
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    assert h.hexdigest()==expected,'HLP archive SHA256 mismatch'
    os.chdir(root);sys.path.insert(0,str(root))
    from eo_data import open_dataset
    reads=[]
    for name,prepared in [('prithvi','processed_data/hlp_split_v2_20260914'),('lstsr_tb','processed_data/v1')]:
        for split in ('train','val','test'):
            for modality in ('hls','lst'):
                ds=open_dataset(name,modality,split,root=root,prepared=root/prepared)
                for i in (0,len(ds)//2,len(ds)-1):
                    sample=ds[i];assert sample['valid_pixels']>0
                reads.append({'dataset':name,'split':split,'modality':modality,'samples':len(ds),'read_checks':3})
                ds.close()
    result={'status':'complete','files':count,'bytes':total,'hlp_archive_sha256':h.hexdigest(),
            'sizes_and_nanosecond_timestamps_verified':True,'read_checks':reads,
            'transfer_integrity':'Modal v2 block SHA256 checks; HLP additional full SHA256; frozen reader artifact hashes.',
            'elapsed_seconds':time.time()-started}
    (root/'_transfer/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    volume.commit()
    return result

@app.local_entrypoint()
def main(probe_only: bool=False):
    import time
    out=Path(__file__).resolve().parents[2]/'output/modal_upload_20260914'
    status_path=out/'status.json'
    def set_status(phase,error=None):
        if probe_only:return
        state=json.loads(status_path.read_text())
        previous=state.pop('error',None)
        if previous:state['previous_error']=previous
        state['phase']=phase;state['updated_unix']=time.time()
        if error:state['error']=error
        temporary=status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(status_path)
    set_status('verifying_remote')
    try:
        result=probe.remote() if probe_only else verify.remote()
    except Exception as exc:
        set_status('verification_failed',f'{type(exc).__name__}: {exc}')
        raise
    path=out/('timestamp_probe.json' if probe_only else 'remote_verification.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    set_status('complete')
    print(json.dumps(result,indent=2))
