"""Full LST census and matched synthetic/native 32x interpolation baselines.

Read-only inputs; atomic resumable Parquet chunks. No GPU or model training.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
import traceback
import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from eo_data import codec
from eo_data.archive import ArchiveFrames
from eo_data.core import json_write, sha256_file

OUT = ROOT/'data_inspection/lst_sr_census_20260913'
VERSION = 'lst-sr-census-v1'
FIELDS = ['valid_pixels','mean_c','std_c','iqr_c','entropy_bits','gradient_c',
          'gradient4_c','gradient16_c','synthetic_lr_valid_fraction',
          'synthetic_pixels','synthetic_mae_c','synthetic_rmse_c','synthetic_bias_c',
          'synthetic_nearest_rmse_c','synthetic_target_std_c','synthetic_nrmse',
          'native_lr_valid_fraction','common_pixels','native_mae_c','native_rmse_c',
          'native_bias_c','native_bias_removed_rmse_c','synthetic_common_rmse_c',
          'coarse_common_pixels','coarse_bias_c','coarse_rmse_c','native_lr_entropy_bits',
          'native_lr_odd_fraction']
_CACHE = {}

@njit(cache=True)
def hist_quantile(hist, n, q):
    pos=(n-1)*q; lo=int(math.floor(pos)); hi=int(math.ceil(pos))
    cum=0; lv=0; hv=0; found=False
    for k in range(65536):
        cum+=hist[k]
        if not found and cum>lo: lv=k-32768; found=True
        if cum>hi: hv=k-32768; break
    return (lv+(hv-lv)*(pos-lo))/100.

@njit(cache=True)
def basic_gradient(raw, valid, scale):
    size=256//scale
    a=np.zeros((size,size),np.float64); m=np.zeros((size,size),np.bool_)
    for y in range(size):
        for x in range(size):
            total=0.; n=0
            for dy in range(scale):
                for dx in range(scale):
                    yy=y*scale+dy; xx=x*scale+dx
                    if valid[yy,xx]: total+=raw[yy,xx]; n+=1
            if n==scale*scale: a[y,x]=total/(n*100.); m[y,x]=True
    total=0.; n=0
    for y in range(size):
        for x in range(size):
            if m[y,x]:
                if y+1<size and m[y+1,x]: total+=abs(a[y,x]-a[y+1,x]); n+=1
                if x+1<size and m[y,x+1]: total+=abs(a[y,x]-a[y,x+1]); n+=1
    return total/n if n else np.nan

@njit(cache=True)
def interpolate(a, m, y, x):
    fy=max(0.,min(7.,(y+.5)/32.-.5)); fx=max(0.,min(7.,(x+.5)/32.-.5))
    y0=int(fy); x0=int(fx); y1=min(y0+1,7); x1=min(x0+1,7)
    wy=fy-y0; wx=fx-x0
    w00=(1-wy)*(1-wx); w01=(1-wy)*wx; w10=wy*(1-wx); w11=wy*wx
    ok=(w00==0 or m[y0,x0]) and (w01==0 or m[y0,x1]) and (w10==0 or m[y1,x0]) and (w11==0 or m[y1,x1])
    return a[y0,x0]*w00+a[y0,x1]*w01+a[y1,x0]*w10+a[y1,x1]*w11, ok

@njit(cache=True)
def metrics(raw, valid, lr_raw, lr_valid, native):
    out=np.full(28,np.nan,np.float64)
    hist=np.zeros(65536,np.int64); ent=np.zeros(656,np.int64)
    block_sum=np.zeros((8,8),np.float64); block_n=np.zeros((8,8),np.int64)
    n=0; total=0.; squares=0.; grad=0.; edges=0
    for y in range(256):
        for x in range(256):
            if not valid[y,x]: continue
            code=int(raw[y,x]); v=code/100.; n+=1; total+=v; squares+=v*v
            hist[code+32768]+=1; ent[code//100+328]+=1
            block_sum[y//32,x//32]+=v; block_n[y//32,x//32]+=1
            if y+1<256 and valid[y+1,x]: grad+=abs(code-int(raw[y+1,x]))/100.; edges+=1
            if x+1<256 and valid[y,x+1]: grad+=abs(code-int(raw[y,x+1]))/100.; edges+=1
    out[0]=n
    if n==0: return out
    out[1]=total/n; out[2]=math.sqrt(max(0.,squares/n-(total/n)**2))
    out[3]=hist_quantile(hist,n,.75)-hist_quantile(hist,n,.25)
    entropy=0.
    for count in ent:
        if count: p=count/n; entropy-=p*math.log2(p)
    out[4]=entropy; out[5]=grad/edges if edges else np.nan
    out[6]=basic_gradient(raw,valid,4); out[7]=basic_gradient(raw,valid,16)
    small=np.zeros((8,8),np.float64); smask=block_n>=922
    for y in range(8):
        for x in range(8):
            if smask[y,x]: small[y,x]=block_sum[y,x]/block_n[y,x]
    out[8]=smask.sum()/64.
    native_values=lr_raw.astype(np.float64)/100.
    out[16]=lr_valid.sum()/64. if native else np.nan
    # Sum error, absolute error, squared error and target moments per image.
    sn=0; sa=0.; ss=0.; sb=0.; nearest_ss=0.; sy=0.; sy2=0.
    cn=0; ca=0.; cs=0.; cb=0.; common_synth_ss=0.
    for y in range(256):
        for x in range(256):
            if not valid[y,x]: continue
            target=raw[y,x]/100.
            pred,sok=interpolate(small,smask,y,x)
            if sok:
                error=pred-target; sn+=1; sa+=abs(error); ss+=error*error; sb+=error
                sy+=target; sy2+=target*target
                ne=small[y//32,x//32]-target; nearest_ss+=ne*ne
            if native and sok:
                npred,nok=interpolate(native_values,lr_valid,y,x)
                if nok:
                    error=npred-target; cn+=1; ca+=abs(error); cs+=error*error; cb+=error
                    se=pred-target; common_synth_ss+=se*se
    out[9]=sn
    if sn:
        out[10]=sa/sn; out[11]=math.sqrt(ss/sn); out[12]=sb/sn; out[13]=math.sqrt(nearest_ss/sn)
        sd=math.sqrt(max(0.,sy2/sn-(sy/sn)**2)); out[14]=sd
        if sd>=.1: out[15]=out[11]/sd
    out[17]=cn if native else np.nan
    if cn:
        out[18]=ca/cn; out[19]=math.sqrt(cs/cn); out[20]=cb/cn
        out[21]=math.sqrt(max(0.,cs/cn-(cb/cn)**2)); out[22]=math.sqrt(common_synth_ss/cn)
    if native:
        bn=0; bs=0.; bb=0.; ln=0; odd=0; le=np.zeros(656,np.int64)
        for y in range(8):
            for x in range(8):
                if lr_valid[y,x]:
                    ln+=1; code=int(lr_raw[y,x]); odd+=code%2!=0; le[code//100+328]+=1
                    if smask[y,x]:
                        e=native_values[y,x]-small[y,x]; bn+=1; bs+=e*e; bb+=e
        out[23]=bn
        if bn: out[24]=bb/bn; out[25]=math.sqrt(bs/bn)
        if ln:
            e=0.
            for count in le:
                if count: p=count/ln; e-=p*math.log2(p)
            out[26]=e; out[27]=odd/ln
    return out

def sanity():
    from scipy.ndimage import map_coordinates
    mask=np.ones((256,256),bool); lr=np.zeros((8,8),np.int16); lm=np.ones((8,8),bool)
    raw=np.full((256,256),2500,np.int16); lr[:]=2500
    got=metrics(raw,mask,lr,lm,True)
    assert got[3]==got[4]==got[5]==got[11]==got[19]==0
    lr[:]=2700; got=metrics(raw,mask,lr,lm,True)
    assert abs(got[19]-2)<1e-10 and got[21]==0 and got[20]==2
    raw=np.random.default_rng(99).integers(-2200,5400,(256,256),dtype=np.int16)
    got=metrics(raw,mask,lr,lm,True)
    q=np.quantile(raw.astype(float)/100,[.25,.75])
    assert abs(got[3]-(q[1]-q[0]))<1e-10
    small=(raw.astype(float)/100).reshape(8,32,8,32).mean(axis=(1,3))
    coords=np.meshgrid((np.arange(256)+.5)/32-.5,(np.arange(256)+.5)/32-.5,indexing='ij')
    ref=map_coordinates(small,coords,order=1,mode='nearest',prefilter=False)
    assert abs(got[11]-np.sqrt(np.mean((ref-raw/100.)**2)))<1e-9
    valid=mask.copy(); valid[:,128:]=False
    a=metrics(raw,valid,lr,lm,True); raw[:,128:]=-32768
    b=metrics(raw,valid,lr,lm,True)
    assert np.allclose(a,b,equal_nan=True)
    return 'Passed constant, known bias, exact IQR, independent SciPy bilinear reference, masked-fill invariance.'

def state(dataset):
    if dataset not in _CACHE:
        prepared=ROOT/'processed_data/v1'/dataset
        bundle=json.loads((prepared/'bundle.json').read_text(encoding='utf-8'))
        db=sqlite3.connect((prepared/'index.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True)
        source=ROOT/bundle['source_root']
        reader=ArchiveFrames(ROOT,json.loads((prepared/'storage.json').read_text())) if dataset=='prithvi' else None
        _CACHE[dataset]=(db,source,reader)
    return _CACHE[dataset]

def run_chunk(job):
    dataset,start,end,outdir=job; target=Path(outdir)/f'{dataset}_{start:07d}_{end:07d}.parquet'
    db,source,reader=state(dataset); rows=[]
    for rid,key,split,tile,meta_text,loc_text in db.execute('SELECT id,key,split,tile,meta,loc FROM records WHERE id>=? AND id<? ORDER BY id',(start,end)):
        loc=json.loads(loc_text); meta=json.loads(meta_text)
        if dataset=='prithvi':
            parts=codec.decode(reader.read(loc))
            raw=np.load(io.BytesIO(parts[2]),allow_pickle=False)[None]
            masks=raw!=-32767; slots=[0]
            lrs=np.zeros((1,8,8),np.int16); lms=np.zeros((1,8,8),bool)
        else:
            path=(source/loc['lst']).resolve()
            if not path.is_relative_to(source.resolve()): raise ValueError('Source escape')
            stat=path.stat(); expected=loc['source_signatures']['lst']
            if stat.st_size!=expected['size'] or stat.st_mtime_ns!=expected['mtime_ns']: raise ValueError(f'Source changed: {key}')
            with np.load(path,allow_pickle=False) as z:
                raw=z['hr_lst']; masks=z['hr_valid'].astype(bool)&(raw!=-32768)
                lrs=z['lr_lst']; lms=z['lr_valid'].astype(bool)&(lrs!=-32768)
            slots=loc['slots']
        for slot in slots:
            m=metrics(raw[slot],masks[slot],lrs[slot],lms[slot],dataset=='lstsr_tb')
            date=meta['datetime'][:10] if dataset=='prithvi' else meta['month_dates'][f'm{slot+1:02d}']
            row={'dataset':dataset,'record_id':rid,'sequence_id':key,'split':'test' if dataset=='prithvi' and split=='val' else split,
                 'source_split':split,'tile':tile,'date':date,'year':int(date[:4]),'month':int(date[5:7]),
                 'slot':slot+1 if dataset=='lstsr_tb' else 0}
            row.update(zip(FIELDS,m.tolist())); rows.append(row)
    df=pd.DataFrame(rows)
    temporary=target.with_suffix('.tmp.parquet'); df.to_parquet(temporary,index=False,compression='zstd'); temporary.replace(target)
    return {'file':target.name,'records':end-start,'samples':len(df),'dataset':dataset}

def input_identity():
    identity={}
    for name in ('prithvi','lstsr_tb'):
        p=ROOT/'processed_data/v1'/name
        b=json.loads((p/'bundle.json').read_text())
        actual=sha256_file(p/'index.sqlite')
        if actual!=b['artifacts']['index.sqlite']: raise ValueError('Index differs from frozen bundle')
        identity[name]={'bundle_sha256':sha256_file(p/'bundle.json'),'index_sha256':actual,'records':b['records']}
    return identity

def run(args):
    OUT.mkdir(parents=True,exist_ok=True)
    checks=sanity(); identity=input_identity()
    outdir=OUT/('benchmark_chunks' if args.benchmark else 'chunks'); outdir.mkdir(exist_ok=True)
    jobs=[]
    for name in ('prithvi','lstsr_tb'):
        n=identity[name]['records']; batch=512 if name=='prithvi' else 64
        if args.benchmark: n=min(n,512 if name=='prithvi' else 256)
        for start in range(1,n+1,batch): jobs.append((name,start,min(n+1,start+batch),str(outdir)))
    script_sha=sha256_file(Path(__file__))
    protocol={'version':VERSION,'script_sha256':script_sha,'source_identity':identity,'checks':checks,'scale':32,
              'degradation':'Area average of valid HR pixels in 32x32 blocks; >=90% (922/1024) block coverage.',
              'upsampling':'Bilinear half-pixel centers, border replication, only valid interpolation stencil.',
              'native_comparison':'Native MODIS and synthetic area-average input compared on exactly the same valid target pixels.',
              'normalization':'Physical Celsius, no training normalization. Native LR /100 follows existing encoding evidence; not an independent full source-unit validation.',
              'bias_removed':'Per-image ground-truth-derived offset removal for diagnosis only; not a deployable baseline.',
              'scope':'All frozen LST indices, all splits; Prithvi source val displayed as test; no HLS or learned models.',
              'weights':'Per-image distributions, pixel pooled RMSE, sequence means and tile means; no trained-model hardness claim.'}
    p=OUT/('benchmark_protocol.json' if args.benchmark else 'protocol.json')
    if p.exists():
        old=json.loads(p.read_text())
        if old['script_sha256']!=script_sha or old['source_identity']!=identity:
            raise RuntimeError('Resume identity changed; use a separate output directory/version.')
    else: json_write(p,protocol)
    finished={}
    for job in jobs:
        name,start,end,_=job; path=outdir/f'{name}_{start:07d}_{end:07d}.parquet'
        if path.exists():
            import pyarrow.parquet as pq
            finished[path.name]={'file':path.name,'dataset':name,'records':end-start,'samples':pq.read_metadata(path).num_rows}
    pending_jobs=[j for j in jobs if f'{j[0]}_{j[1]:07d}_{j[2]:07d}.parquet' not in finished]
    started=time.time(); initial_samples=sum(v['samples'] for v in finished.values())
    status_path=OUT/('benchmark_status.json' if args.benchmark else 'status.json')
    def status(phase):
        totals={n:{'records':0,'samples':0} for n in ('prithvi','lstsr_tb')}
        for r in finished.values():
            for key in ('records','samples'): totals[r['dataset']][key]+=r[key]
        elapsed=time.time()-started; samples=sum(r['samples'] for r in finished.values())
        report={'status':phase,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':os.getpid(),
                'workers':args.workers,'completed_chunks':len(finished),'total_chunks':len(jobs),'datasets':totals,
                'elapsed_seconds_this_run':elapsed,'samples_per_second_this_run':(samples-initial_samples)/max(elapsed,1)}
        json_write(status_path,report); print(json.dumps(report),flush=True)
    status('running')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        it=iter(pending_jobs); active={}
        for _ in range(args.workers*2):
            j=next(it,None)
            if j: active[pool.submit(run_chunk,j)]=j
        last=0
        while active:
            done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                job=active.pop(future)
                try: r=future.result()
                except Exception:
                    json_write(OUT/'failure.json',{'job':job,'traceback':traceback.format_exc()}); status('failed'); raise
                finished[r['file']]=r
                j=next(it,None)
                if j: active[pool.submit(run_chunk,j)]=j
            if time.time()-last>=30: status('running'); last=time.time()
    status('scan_complete')
    json_write(OUT/('benchmark_inventory.json' if args.benchmark else 'chunk_inventory.json'),list(finished.values()))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=4); parser.add_argument('--benchmark',action='store_true')
    run(parser.parse_args())
