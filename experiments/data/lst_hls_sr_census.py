"""Full LR-LST + HLS guided baseline; no target-HR fitting or hyperparameter search."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import io,json,math,sqlite3,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import lst_sr_census as base
ROOT=base.ROOT
OUT=ROOT/'data_inspection/lst_hls_sr_census_20260913'
FIELDS=['valid_pixels','hls_valid_pixels','hls_coarse_valid_cells','synthetic_pixels','synthetic_baseline_rmse_c',
        'synthetic_guided_rmse_c','synthetic_guided_mae_c','synthetic_guided_pixels','synthetic_fit_cells',
        'synthetic_guided_bias_c','common_pixels','native_baseline_rmse_c','native_guided_rmse_c',
        'native_guided_mae_c','native_guided_pixels','native_fit_cells','native_guided_bias_c',
        'synthetic_common_guided_rmse_c','synthetic_common_baseline_rmse_c','max_abs_hls_lst_correlation']

@njit(cache=True)
def optical_coarse(opt,mask):
    coarse=np.zeros((6,8,8),np.float64);counts=np.zeros((8,8),np.int64)
    for y in range(256):
        for x in range(256):
            if mask[y,x]:
                by=y//32;bx=x//32;counts[by,bx]+=1
                for c in range(6):coarse[c,by,bx]+=opt[c,y,x]*.0001
    for y in range(8):
        for x in range(8):
            if counts[y,x]:
                for c in range(6):coarse[c,y,x]/=counts[y,x]
    return coarse,counts>=922

@njit(cache=True)
def fit_coefficients(coarse,cmask,small,smask):
    mask=cmask&smask;n=int(mask.sum());beta=np.zeros(6,np.float64)
    if n<16:return beta,n,False
    features=np.zeros((n,6),np.float64);target=np.zeros(n,np.float64);i=0
    for y in range(8):
        for x in range(8):
            if mask[y,x]:
                for c in range(6):features[i,c]=coarse[c,y,x]
                target[i]=small[y,x];i+=1
    means=np.zeros(6,np.float64);stds=np.zeros(6,np.float64)
    for c in range(6):
        for i in range(n):means[c]+=features[i,c]/n
        for i in range(n):stds[c]+=(features[i,c]-means[c])**2/n
        stds[c]=max(math.sqrt(stds[c]),.0001)
        for i in range(n):features[i,c]=(features[i,c]-means[c])/stds[c]
    target=target-target.mean()
    # Minimize mean squared LR residual + 1.0 * ||standardized coefficients||².
    gram=features.T@features/n+np.eye(6)
    rhs=features.T@target/n
    scaled=np.linalg.solve(gram,rhs)
    for c in range(6):beta[c]=scaled[c]/stds[c]
    return beta,n,True

@njit(cache=True)
def guided_metrics(raw,valid,lr,lm,opt,omask,coarse,cmask,has_native):
    out=np.full(20,np.nan,np.float64);out[0]=valid.sum();out[1]=omask.sum();out[2]=cmask.sum()
    sums=np.zeros((8,8),np.float64);counts=np.zeros((8,8),np.int64)
    for y in range(256):
        for x in range(256):
            if valid[y,x]:sums[y//32,x//32]+=raw[y,x]/100.;counts[y//32,x//32]+=1
    small=np.zeros((8,8),np.float64);smask=counts>=922
    for y in range(8):
        for x in range(8):
            if smask[y,x]:small[y,x]=sums[y,x]/counts[y,x]
    bs,ns,fit_s=fit_coefficients(coarse,cmask,small,smask)
    native=lr.astype(np.float64)/100.
    bn=np.zeros(6,np.float64);nn=0;fit_n=False
    if has_native:bn,nn,fit_n=fit_coefficients(coarse,cmask,native,lm)
    # Only LR-derived coefficients enter prediction. HR is used below for scoring only.
    coarse_s=np.zeros((8,8),np.float64);coarse_n=np.zeros((8,8),np.float64)
    for c in range(6):
        coarse_s+=bs[c]*coarse[c]
        coarse_n+=bn[c]*coarse[c]
    sn=0;sbse=0.;sgse=0.;sgae=0.;sgbias=0.;sgpixels=0
    cn=0;nbse=0.;ngse=0.;ngae=0.;ngbias=0.;ngpixels=0;scgse=0.;scbse=0.
    cor_n=0;xt=np.zeros(6,np.float64);xt2=np.zeros(6,np.float64);xy=np.zeros(6,np.float64);yt=0.;yt2=0.
    for y in range(256):
        for x in range(256):
            if not valid[y,x]:continue
            truth=raw[y,x]/100.
            pred,sok=base.interpolate(small,smask,y,x)
            if not sok:continue
            guided=pred;guide_ok=False;hsv=0.;hnv=0.
            if omask[y,x] and (fit_s or fit_n):
                csv,cok=base.interpolate(coarse_s,cmask,y,x)
                if cok:
                    guide_ok=True
                    for c in range(6):
                        value=opt[c,y,x]*.0001
                        hsv+=bs[c]*value;hnv+=bn[c]*value
                    if fit_s:guided=pred+hsv-csv;sgpixels+=1
            sn+=1;error=guided-truth;sgse+=error*error;sgae+=abs(error);sgbias+=error
            sbse+=(pred-truth)**2
            if omask[y,x]:
                cor_n+=1;yt+=truth;yt2+=truth*truth
                for c in range(6):
                    value=opt[c,y,x]*.0001;xt[c]+=value;xt2[c]+=value*value;xy[c]+=value*truth
            if has_native:
                npred,nok=base.interpolate(native,lm,y,x)
                if nok:
                    nguided=npred
                    if guide_ok and fit_n:
                        cnv,_=base.interpolate(coarse_n,cmask,y,x)
                        nguided=npred+hnv-cnv;ngpixels+=1
                    cn+=1;error=nguided-truth;ngse+=error*error;ngae+=abs(error);ngbias+=error
                    nbse+=(npred-truth)**2;scgse+=(guided-truth)**2;scbse+=(pred-truth)**2
    out[3]=sn;out[7]=sgpixels;out[8]=ns
    if sn:
        out[4]=math.sqrt(sbse/sn);out[5]=math.sqrt(sgse/sn);out[6]=sgae/sn;out[9]=sgbias/sn
    if has_native:
        out[10]=cn;out[14]=ngpixels;out[15]=nn
        if cn:
            out[11]=math.sqrt(nbse/cn);out[12]=math.sqrt(ngse/cn);out[13]=ngae/cn;out[16]=ngbias/cn
            out[17]=math.sqrt(scgse/cn);out[18]=math.sqrt(scbse/cn)
    if cor_n>1:
        vy=yt2-yt*yt/cor_n;best=0.;good=False
        for c in range(6):
            vx=xt2[c]-xt[c]*xt[c]/cor_n
            if vx>1e-12 and vy>1e-10:
                r=(xy[c]-xt[c]*yt/cor_n)/math.sqrt(vx*vy);best=max(best,abs(r));good=True
        if good:out[19]=min(1.,best)
    return out

def lower_priority():
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x00004000)

def sanity():
    rng=np.random.default_rng(100)
    opt=rng.integers(100,7000,(6,256,256),dtype=np.int16);om=np.ones((256,256),bool)
    coarse,cm=optical_coarse(opt,om)
    assert np.allclose(coarse,opt.reshape(6,8,32,8,32).mean(axis=(2,4))*.0001)
    raw=np.full((256,256),2500,np.int16);valid=np.ones((256,256),bool);lr=np.full((8,8),2500,np.int16);lm=np.ones((8,8),bool)
    got=guided_metrics(raw,valid,lr,lm,opt,om,coarse,cm,True)
    assert got[4]==got[5]==got[11]==got[12]==0
    target=coarse[0]*20-coarse[2]*10+5
    beta,n,ok=fit_coefficients(coarse,cm,target,lm)
    x=coarse.reshape(6,-1).T;mean=x.mean(0);std=np.maximum(x.std(0),.0001);z=(x-mean)/std
    ref=np.linalg.solve(z.T@z/64+np.eye(6),z.T@(target.ravel()-target.mean())/64)/std
    assert ok and n==64 and np.allclose(beta,ref,atol=1e-10)
    raw=rng.integers(-2000,5000,(256,256),dtype=np.int16)
    a=guided_metrics(raw,valid,lr,lm,opt,om,coarse,cm,True)
    b=base.metrics(raw,valid,lr,lm,True)
    assert a[3]==b[9] and a[10]==b[17] and abs(a[4]-b[11])<1e-9 and abs(a[11]-b[19])<1e-9
    none=np.zeros((256,256),bool);c,cm=optical_coarse(opt,none)
    a=guided_metrics(raw,valid,lr,lm,opt,none,c,cm,True)
    assert a[4]==a[5] and a[11]==a[12]
    return 'Passed optical block means, constant field, independent NumPy ridge fit, baseline parity, no-optics fallback.'

def run_chunk(job):
    lower_priority()
    dataset,start,end,outdir=job;target=Path(outdir)/f'{dataset}_{start:07d}_{end:07d}.parquet'
    db,source,reader=base.state(dataset);rows=[]
    for rid,key,split,tile,meta_text,loc_text in db.execute('SELECT id,key,split,tile,meta,loc FROM records WHERE id>=? AND id<? ORDER BY id',(start,end)):
        loc=json.loads(loc_text);meta=json.loads(meta_text)
        if dataset=='prithvi':
            parts=base.codec.decode(reader.read(loc))
            opt=np.load(io.BytesIO(parts[0]),allow_pickle=False);fm=np.load(io.BytesIO(parts[1]),allow_pickle=False)
            raw=np.load(io.BytesIO(parts[2]),allow_pickle=False)[None];valid=raw!=-32767
            om=~np.isin(opt,(-9999,-32767,-32768)).any(axis=0)&(fm!=255)&((fm&14)==0)
            slots=[0];lr=np.zeros((1,8,8),np.int16);lm=np.zeros((1,8,8),bool)
            stamp=meta['hls_image_id'].split('_')[-1][:8];hdate=f'{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}'
        else:
            for kind in ('lst','hls'):
                path=(source/loc[kind]).resolve();st=path.stat();e=loc['source_signatures'][kind]
                if not path.is_relative_to(source.resolve()) or st.st_size!=e['size'] or st.st_mtime_ns!=e['mtime_ns']:raise ValueError('Changed source '+key)
            with np.load(source/loc['lst'],allow_pickle=False) as z:
                raw=z['hr_lst'];valid=z['hr_valid'].astype(bool)&(raw!=-32768);lr=z['lr_lst'];lm=z['lr_valid'].astype(bool)&(lr!=-32768)
            with np.load(source/loc['hls'],allow_pickle=False) as z:
                opt=z['hls_reflectance_i16'];fm=z['hls_fmask'];om=z['hls_valid'].astype(bool)&(fm!=255)&((fm&14)==0)&~(opt==-32768).any(axis=0)
            slots=loc['slots'];hdate=meta['hls_datetime'][:10]
        coarse,cm=optical_coarse(opt,om)
        for slot in slots:
            result=guided_metrics(raw[slot],valid[slot],lr[slot],lm[slot],opt,om,coarse,cm,dataset=='lstsr_tb')
            date=meta['datetime'][:10] if dataset=='prithvi' else meta['month_dates'][f'm{slot+1:02d}']
            gap=abs(int((np.datetime64(date)-np.datetime64(hdate))/np.timedelta64(1,'D')))
            row={'dataset':dataset,'record_id':rid,'sequence_id':key,'split':'test' if dataset=='prithvi' and split=='val' else split,
                 'source_split':split,'tile':tile,'date':date,'hls_date':hdate,'hls_date_gap_days':gap,
                 'year':int(date[:4]),'month':int(date[5:7]),'slot':slot+1 if dataset=='lstsr_tb' else 0}
            row.update(zip(FIELDS,result.tolist()));rows.append(row)
    frame=pd.DataFrame(rows);temp=target.with_suffix('.tmp.parquet');frame.to_parquet(temp,index=False,compression='zstd');temp.replace(target)
    return {'file':target.name,'dataset':dataset,'records':end-start,'samples':len(rows)}

def run(args):
    OUT.mkdir(exist_ok=True);checks=sanity();identity=base.input_identity()
    directory=OUT/('benchmark_chunks' if args.benchmark else 'chunks');directory.mkdir(exist_ok=True)
    protocol={'source_identity':identity,'script_sha256':base.sha256_file(Path(__file__)),
       'base_script_sha256':base.sha256_file(HERE/'lst_sr_census.py'),'checks':checks,'scale':32,'ridge_lambda':1.,'minimum_fit_cells':16,
       'fit':'Per-image ridge fitted ONLY to LR LST and coarse HLS; six features standardized across valid fitting cells, std floor 0.0001 reflectance. Objective mean squared LR error + lambda*coefficient_norm_squared.',
       'prediction':'bilinear(LR) + beta*HLS_HR - bilinear(beta*coarse_HLS). Missing optics or insufficient LR fitting cells fall back to bilinear LR; no target fitting.',
       'scope':'Full LR LST + single HLS image task, both synthetic and real MODIS inputs. Same target pixels as base census.',
       'dates':'Calendar-day gap between HLS date and target metadata date. Prithvi target date uses metadata.datetime, not independently confirmed LST acquisition time.'}
    pp=OUT/('benchmark_protocol.json' if args.benchmark else 'protocol.json')
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Resume protocol mismatch')
    if not pp.exists():base.json_write(pp,protocol)
    jobs=[]
    for name in ('prithvi','lstsr_tb'):
        n=identity[name]['records'];batch=256 if name=='prithvi' else 32
        if args.benchmark:n=min(n,256 if name=='prithvi' else 128)
        jobs.extend((name,a,min(a+batch,n+1),str(directory)) for a in range(1,n+1,batch))
    finished={};pending=[]
    import pyarrow.parquet as pq
    for j in jobs:
        name,a,b,_=j;p=directory/f'{name}_{a:07d}_{b:07d}.parquet'
        if p.exists():finished[p.name]={'file':p.name,'dataset':name,'records':b-a,'samples':pq.read_metadata(p).num_rows}
        else:pending.append(j)
    initial=sum(r['samples'] for r in finished.values());started=time.time()
    statuspath=OUT/('benchmark_status.json' if args.benchmark else 'status.json')
    def status(phase):
        totals={n:{'records':0,'samples':0} for n in ('prithvi','lstsr_tb')}
        for r in finished.values():
            for k in ('records','samples'):totals[r['dataset']][k]+=r[k]
        elapsed=time.time()-started;count=sum(r['samples'] for r in finished.values())
        report={'status':phase,'pid':os.getpid(),'workers':args.workers,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'completed_chunks':len(finished),'total_chunks':len(jobs),'datasets':totals,'elapsed_seconds_this_run':elapsed,
                'samples_per_second_this_run':(count-initial)/max(1,elapsed)}
        base.json_write(statuspath,report);print(json.dumps(report),flush=True)
    status('running')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        it=iter(pending);active={}
        for _ in range(args.workers*2):
            j=next(it,None)
            if j:active[pool.submit(run_chunk,j)]=j
        last=0
        while active:
            done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                j=active.pop(future)
                try:r=future.result()
                except Exception:base.json_write(OUT/'failure.json',{'job':j,'traceback':traceback.format_exc()});status('failed');raise
                finished[r['file']]=r;j=next(it,None)
                if j:active[pool.submit(run_chunk,j)]=j
            if time.time()-last>=30:status('running');last=time.time()
    status('scan_complete')
    base.json_write(OUT/('benchmark_inventory.json' if args.benchmark else 'chunk_inventory.json'),list(finished.values()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--benchmark',action='store_true');run(p.parse_args())
