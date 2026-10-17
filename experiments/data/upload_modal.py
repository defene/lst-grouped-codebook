"""Resumable upload of the reviewed dataset manifest, Modal SDK 1.5.5."""
import concurrent.futures
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/modal_upload_20260914'
os.environ['MODAL_PROFILE'] = 'runlongyu-target'
os.environ['MODAL_ENVIRONMENT'] = 'main'
os.environ['PYTHONIOENCODING'] = 'utf-8'
import modal
from modal.client import Client
from modal.volume import VolumeUploadContextManager2
from modal_proto import api_pb2

LOCK = threading.Lock()
STOP = threading.Event()
STATE = {'phase':'starting','committed_files':0,'committed_bytes':0,'sent_bytes_this_run':0,'active':{}}

def save_json(path, data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)

def progress_loop():
    while not STOP.wait(20):
        with LOCK:
            STATE['updated_unix']=time.time()
            save_json(OUT/'status.json',STATE)
            print(json.dumps(STATE,ensure_ascii=False),flush=True)

def upload_batch(volume, batch_id, rows, manifest_hash):
    checkpoint=OUT/'completed'/f'{batch_id:05d}.json'
    identity=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    size=sum(r['size'] for r in rows)
    if checkpoint.exists():
        done=json.loads(checkpoint.read_text())
        assert done['identity']==identity and done['manifest']==manifest_hash and done['volume_id']==volume.object_id
        with LOCK:
            STATE['committed_files']+=len(rows);STATE['committed_bytes']+=size
        return
    for row in rows:
        p=ROOT/row['path'];st=p.stat()
        if (st.st_size,st.st_mtime_ns)!=(row['size'],row['mtime_ns']):
            raise ValueError(f'Source changed since preflight: {row["path"]}')
    with LOCK: STATE['active'][str(batch_id)]={'files':len(rows),'bytes':size,'first':rows[0]['path']}
    def callback(**kwargs):
        if 'advance' in kwargs:
            with LOCK: STATE['sent_bytes_this_run']+=kwargs['advance']
        return kwargs.get('name','')
    for attempt in range(4):
        try:
            # Pinned SDK backend supplies block checksums and accurate byte callbacks.
            with VolumeUploadContextManager2(volume.object_id,volume.client,progress_cb=callback,
                                            force=True,hash_concurrency=4,put_concurrency=32) as batch:
                for row in rows:batch.put_file(ROOT/row['path'],'/'+row['path'])
            break
        except Exception as exc:
            if attempt==3:raise
            print(f'Batch {batch_id}: retry {attempt+1}, {type(exc).__name__}: {exc}',flush=True)
            time.sleep(2**attempt)
    for row in rows:
        st=(ROOT/row['path']).stat()
        assert (st.st_size,st.st_mtime_ns)==(row['size'],row['mtime_ns']),row['path']
    save_json(checkpoint,{'identity':identity,'manifest':manifest_hash,'volume_id':volume.object_id,'files':len(rows),'bytes':size})
    with LOCK:
        STATE['active'].pop(str(batch_id),None)
        STATE['committed_files']+=len(rows);STATE['committed_bytes']+=size
        save_json(OUT/'status.json',STATE)

def main():
    import msvcrt
    lock_file=(OUT/'upload.lock').open('a+b');lock_file.seek(0);lock_file.write(b'0');lock_file.flush();lock_file.seek(0)
    msvcrt.locking(lock_file.fileno(),msvcrt.LK_NBLCK,1)
    from modal.config import config, _lookup_workspace
    info=asyncio.run(_lookup_workspace(config.get('server_url'),config.get('token_id'),config.get('token_secret')))
    if info.username!='runlongyu':raise RuntimeError(f'Wrong workspace: {info.username}')
    plan=json.loads((OUT/'plan.json').read_text())
    raw=(OUT/'files.jsonl').read_bytes();manifest_hash=hashlib.sha256(raw).hexdigest()
    assert manifest_hash==plan['manifest_sha256']
    rows=[json.loads(line) for line in raw.splitlines()]
    receipt=OUT/'volume.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text());assert saved['manifest']==manifest_hash
        volume=modal.Volume.from_name(saved['name'],environment_name='main');volume.hydrate()
        assert volume.object_id==saved['id']
    else:
        name=plan['proposed_volume']
        modal.Volume.objects.create(name,version=2,environment_name='main')
        volume=modal.Volume.from_name(name,environment_name='main');volume.hydrate()
        save_json(receipt,{'name':name,'id':volume.object_id,'workspace':'runlongyu','environment':'main','manifest':manifest_hash})
    (OUT/'completed').mkdir(exist_ok=True)
    STATE.update(phase='uploading',pid=os.getpid(),started_unix=time.time(),total_files=len(rows),total_bytes=plan['bytes'],volume_id=volume.object_id)
    save_json(OUT/'status.json',STATE)
    print('Verified runlongyu/main; uploading to '+volume.object_id,flush=True)
    # Small reader/metadata batches first; the large HLP ZIP runs alongside HLC batches.
    priority={'reader':0,'documentation':1,'HLP':2,'HLP-history':3,'HLC':4}
    rows.sort(key=lambda r:(priority[r['group']],r['size']>10**10,r['path']))
    batches=[];batch=[];size=0
    for row in rows:
        if batch and (len(batch)>=256 or size+row['size']>1024**3):
            batches.append(batch);batch=[];size=0
        batch.append(row);size+=row['size']
    if batch:batches.append(batch)
    watcher=threading.Thread(target=progress_loop,daemon=True);watcher.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            pending={};next_batch=0
            while next_batch<len(batches) or pending:
                while next_batch<len(batches) and len(pending)<3:
                    future=pool.submit(upload_batch,volume,next_batch,batches[next_batch],manifest_hash)
                    pending[future]=next_batch;next_batch+=1
                done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:future.result();del pending[future]
        with volume.batch_upload(force=True) as batch:
            batch.put_file(OUT/'files.jsonl','/_transfer/files.jsonl')
            batch.put_file(OUT/'plan.json','/_transfer/plan.json')
        STATE['phase']='verifying_remote';save_json(OUT/'status.json',STATE)
        subprocess.run([sys.executable,'-m','modal','run','--env','main',str(ROOT/'experiments/data/verify_modal_upload.py')],cwd=ROOT,check=True)
        STATE['phase']='complete';save_json(OUT/'status.json',STATE)
        print('UPLOAD AND REMOTE VERIFICATION COMPLETE',flush=True)
    except BaseException as exc:
        STATE['phase']='failed';STATE['error']=f'{type(exc).__name__}: {exc}';save_json(OUT/'status.json',STATE)
        raise
    finally:
        STOP.set();watcher.join(timeout=2);lock_file.close()

if __name__=='__main__':main()
