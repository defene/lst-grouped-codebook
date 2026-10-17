"""Extend the reviewed color census while newly verified downloads arrive."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import concurrent.futures as cf
import hashlib,json,sqlite3,time,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.openimages_hlp_color import analyze as base
from eo_data import open_dataset
NAT=ROOT/'data/openimages_train218975_20260917'
OUT=ROOT/'runs/openimages_hlp_color_matched_20260917'
OLD=base.OUT
COUNT=218975
PREP=base.PREP
natural=base.natural
hlp=base.hlp
moments=base.moments
linearize=base.linearize
dump=base.dump

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def init_worker():base.NAT=NAT

def main():
    OUT.mkdir(exist_ok=True);chunks=OUT/'chunks';chunks.mkdir(exist_ok=True)
    selection=json.loads((NAT/'selection.json').read_text(encoding='utf-8'))
    assert selection['count']==COUNT
    rows=[json.loads(s) for s in (NAT/'selected_metadata.jsonl').read_text(encoding='utf-8').splitlines()]
    ids=[r['ImageID'] for r in rows];assert len(ids)==len(set(ids))==COUNT
    rows=[dict(image_id=rid,path=f'images/{rid[:2]}/{rid}.jpg') for rid in ids]
    old=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
    ds=open_dataset('prithvi','hls','train',prepared=base.PREP)
    lst=open_dataset('prithvi','lst','train',prepared=base.PREP)
    assert ds.fingerprint==old['hls_fingerprint'] and lst.fingerprint==old['lst_fingerprint']
    assert len(ds)==COUNT and np.array_equal(ds.indices[:,:2],lst.indices[:,:2])
    old_ids=[json.loads(s)['image_id'] for s in (base.NAT/'download_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    assert ids[:100000]==old_ids
    # Immutable moments are reused only after input identity and per-chunk membership checks.
    reused={'hlp':0,'natural':0}
    for source in sorted((OLD/'chunks').glob('*.npz')):
        kind,start=source.stem.split('_');start=int(start)
        with np.load(source) as z:
            n=len(z['ids'])
            if kind=='natural':
                if n!=min(256,COUNT-start):continue
                assert z['ids'].tolist()==ids[start:start+n]
            target=chunks/source.name
            if not target.exists():os.link(source,target)
            reused[kind]+=n
    assert reused=={'hlp':COUNT,'natural':99840}
    jobs=[];complete=0;cached=[]
    for start in range(0,COUNT,256):
        subset=rows[start:start+256];p=chunks/f'natural_{start:06d}.npz'
        if p.exists():
            with np.load(p) as z:assert z['ids'].tolist()==ids[start:start+len(subset)]
            complete+=len(subset);cached.append(start)
        else:jobs.append(('natural',start,subset,str(chunks)))
    protocol=dict(old)
    protocol.update(status='running',n_natural=COUNT,n_hlp=COUNT,workers=8,
        natural_manifest_sha256=None,natural_selection_sha256=sha(NAT/'selection.json'),
        script_sha256=sha(Path(__file__)),statistic_implementation_sha256=sha(Path(base.__file__)),
        parent_report=str(OLD),parent_protocol_sha256=sha(OLD/'protocol.json'),
        reused_moments=reused,scope='same-size full census; unchanged statistics and HLP members')
    base.dump(OUT/'protocol.json',protocol)
    started=time.time();last=0;queued=set(cached);futures={}
    db=sqlite3.connect((NAT/'downloads.sqlite3').as_uri()+'?mode=ro',uri=True)
    with cf.ProcessPoolExecutor(max_workers=8,initializer=init_worker) as pool:
        while True:
            ready={r[0] for r in db.execute('SELECT image_id FROM images')}
            for job in jobs:
                if len(futures)>=16:break
                if job[1] not in queued and all(r['image_id'] in ready for r in job[2]):
                    futures[pool.submit(base.chunk,job)]=job;queued.add(job[1])
            done=[f for f in futures if f.done()]
            for f in done:
                kind,n,_=f.result();complete+=n;del futures[f]
            download=json.loads((NAT/'progress.json').read_text(encoding='utf-8'))
            state=dict(status='analyzing' if complete<COUNT else 'awaiting_download_audit',
                completed={'natural':complete,'hlp':COUNT},total={'natural':COUNT,'hlp':COUNT},
                reused=reused,download_completed=len(ready),download_status=download['status'],seconds=time.time()-started)
            if time.time()-last>15:
                base.dump(OUT/'progress.json',state);print(json.dumps(state),flush=True);last=time.time()
            if complete==COUNT and download['status']=='complete':break
            if (NAT/'last_error.json').exists():raise RuntimeError((NAT/'last_error.json').read_text(encoding='utf-8'))
            time.sleep(2)
    manifest=[json.loads(s) for s in (NAT/'download_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['image_id'] for r in manifest]==ids
    assert sum(p.stat().st_size for p in (NAT/'images').glob('*/*.jpg'))==sum(r['bytes'] for r in manifest)
    protocol.update(status='complete',natural_manifest_sha256=sha(NAT/'download_manifest.jsonl'),elapsed_seconds=time.time()-started)
    base.dump(OUT/'protocol.json',protocol)
    state['status']='complete';base.dump(OUT/'progress.json',state);print(json.dumps(state),flush=True)

if __name__=='__main__':main()
