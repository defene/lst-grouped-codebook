"""Read-only census of color/spatial moments; each image receives equal weight."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse, concurrent.futures as cf, hashlib, io, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from eo_data import open_dataset
from eo_data.core import canonical
PREP=ROOT/'processed_data/hlp_split_v2_20260914'
NAT=ROOT/'data/openimages_train100k_20260916'
OUT=ROOT/'runs/openimages_hlp_color_20260917'
DS=None

def linearize(x):
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def natural(path):
    with Image.open(path) as im:
        icc=bool(im.info.get('icc_profile'))
        im=im.convert('RGB')  # Same default MPO frame / no EXIF transform as prior VAR evaluation.
        w,h=im.size
        size=(288,int(288*h/w)) if w<=h else (int(288*w/h),288)
        im=im.resize(size,Image.Resampling.LANCZOS)
        left=round((size[0]-256)/2); top=round((size[1]-256)/2)
        im=im.crop((left,top,left+256,top+256))
        x=np.asarray(im,dtype=np.float32).transpose(2,0,1)/255
    return x,icc

def hlp(index,check=False):
    global DS
    if DS is None: DS=open_dataset('prithvi','hls','train',prepared=PREP,verify=False)
    _,rid,slot,count,row=DS._record(index)
    parts=DS._parts(json.loads(row[3]),row[4])  # Verify each frame's original byte digest.
    raw=[np.load(io.BytesIO(parts[k]),allow_pickle=False) for k in range(3)]
    x,m=canonical(raw[0],'hls','prithvi',fmask=raw[1])
    y,n=canonical(raw[2],'lst_hr','prithvi')
    assert int(m.sum())==count
    if check:
        pub=DS[index]; assert np.array_equal(m,pub['reference_mask'])
        assert np.allclose(x[:,m[0]],DS.denormalize(pub['image'])[:,m[0]],atol=1e-7)
        lst=open_dataset('prithvi','lst','train',prepared=PREP,verify=False)
        assert int(lst.indices[index,0])==rid
        pub=lst[index]; assert np.array_equal(n,pub['reference_mask'])
        assert np.allclose(y[:,n[0]],lst.denormalize(pub['image'])[:,n[0]],atol=.002)
        lst.close()
    return x,m[0],y/100,n[0],row[0],row[1]

def moments(x,mask,rng):
    z=x[:,mask].astype(np.float64)
    assert z.shape[1]>0 and np.isfinite(z).all()
    mean=z.mean(axis=1); centered=z-mean[:,None]
    cov=centered@centered.T/z.shape[1]
    dx=x[:,:,1:]-x[:,:,:-1]; dy=x[:,1:,:]-x[:,:-1,:]
    mx=mask[:,1:]&mask[:,:-1]; my=mask[1:,:]&mask[:-1,:]
    edge=(np.square(dx[:,mx],dtype=np.float64).sum(axis=1)+np.square(dy[:,my],dtype=np.float64).sum(axis=1))/(mx.sum()+my.sum())
    # Pixel sample is only used for density plots; exact moments use every valid pixel.
    sampled=z[:,rng.integers(0,z.shape[1],size=64)].T.astype(np.float32)
    return dict(mean=mean,cov=cov,edge=edge,pixels=np.int64(z.shape[1]),
                at_zero=(z==0).mean(axis=1),at_one=(z==1).mean(axis=1),
                minimum=z.min(axis=1),maximum=z.max(axis=1),sampled=sampled)

def chunk(job):
    kind,start,items,outdir=job
    target=Path(outdir)/f'{kind}_{start:06d}.npz'
    if target.exists():
        with np.load(target) as z: assert len(z['ids'])==len(items)
        return kind,len(items),'cached'
    data={}; ids=[]; tiles=[]; profiles=[]
    def append(prefix,stats):
        for k,v in stats.items():data.setdefault(prefix+'_'+k,[]).append(v)
    for i,item in enumerate(items):
        if kind=='natural':
            ident=item['image_id']; x,icc=natural(NAT/item['path']); mask=np.ones((256,256),bool)
            seed=int.from_bytes(hashlib.sha256(('20260917:'+ident).encode()).digest()[:8],'little')
            append('encoded',moments(x,mask,np.random.default_rng(seed)))
            append('linear',moments(linearize(x),mask,np.random.default_rng(seed)))
            profiles.append(icc); tiles.append('')
        else:
            x,mask,y,lmask,ident,tile=hlp(item,check=(start==0 and i<3))
            seed=int.from_bytes(hashlib.sha256(('20260917:'+ident).encode()).digest()[:8],'little')
            append('hls',moments(x,mask,np.random.default_rng(seed)))
            append('lst',moments(y,lmask,np.random.default_rng(seed)))
            tiles.append(tile); profiles.append(False)
        ids.append(ident)
    np.savez_compressed(target,ids=np.asarray(ids),tiles=np.asarray(tiles),icc=np.asarray(profiles),**{k:np.asarray(v) for k,v in data.items()})
    return kind,len(items),'computed'

def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    out=OUT if not a.limit else OUT/f'smoke_{a.limit}'
    chunks=out/'chunks';chunks.mkdir(parents=True,exist_ok=True)
    ds=open_dataset('prithvi','hls','train',prepared=PREP)
    lst=open_dataset('prithvi','lst','train',prepared=PREP)
    assert np.array_equal(ds.indices[:,:2],lst.indices[:,:2])
    rows=[json.loads(s) for s in (NAT/'download_manifest.jsonl').read_text().splitlines()]
    assert len(rows)==100000 and len({r['image_id'] for r in rows})==100000
    n_hlp=len(ds); n_nat=len(rows)
    if a.limit: n_hlp=min(n_hlp,a.limit); rows=rows[:a.limit];n_nat=len(rows)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    dump(out/'protocol.json',dict(status='running',hls_fingerprint=ds.fingerprint,lst_fingerprint=lst.fingerprint,
        natural_manifest_sha256=sha(NAT/'download_manifest.jsonl'),script_sha256=sha(Path(__file__)),
        n_hlp=n_hlp,n_natural=n_nat,workers=a.workers,weighting='equal images; population covariance ddof=0 within each valid mask',
        natural_preprocessing='PIL RGB default first frame; no ICC/EXIF transform; short edge 288 Lanczos; center crop 256, same as prior VAR evaluation',
        natural_linearization='Assume sRGB encoding; c<=0.04045:c/12.92 else ((c+0.055)/1.055)^2.4. Sensitivity analysis, not radiometric calibration.',
        hls_rgb=[2,1,0],hls_scale=.0001,lst_scale_to_celsius=.01,
        density_sampling='64 independently sampled valid pixels with replacement per image, deterministic SHA256-derived seed; exact moments use ALL pixels',
        sources=['https://www.w3.org/TR/css-color-4/','https://hls.gsfc.nasa.gov/products-description/l30/']))
    jobs=[]
    for kind,items in [('natural',rows),('hlp',list(range(n_hlp)))]:
        for start in range(0,len(items),256):jobs.append((kind,start,items[start:start+256],str(chunks)))
    # Interleave sources to observe both sources early without random archive seeks.
    jobs.sort(key=lambda j:(j[1],j[0]))
    started=time.time();counts={'natural':0,'hlp':0}
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(chunk,j) for j in jobs]
        for f in cf.as_completed(futures):
            kind,n,_=f.result();counts[kind]+=n
            progress=dict(status='running',completed=counts,total={'natural':n_nat,'hlp':n_hlp},seconds=time.time()-started)
            dump(out/'progress.json',progress)
            if sum(counts.values())%4096<n:print(json.dumps(progress),flush=True)
    progress['status']='complete';dump(out/'progress.json',progress)
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    protocol['status']='complete';protocol['elapsed_seconds']=time.time()-started
    dump(out/'protocol.json',protocol)
    print(json.dumps(progress),flush=True)

if __name__=='__main__':main()
