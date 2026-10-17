"""Reproducible manifest census and deterministic array sample; never modifies data."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data_inspection/extracted_metadata/lstsr_tb_final_dataset_20260719'
OUT = ROOT/'data_inspection/normalization'
SEED = 20260905
VALUES = np.arange(-32768,32768,dtype=np.float64)

def summarize(h, scale=1):
    n=int(h.sum()); c=h.cumsum(); v=VALUES*scale
    mean=float(h@v/n); std=float(np.sqrt(h@((v-mean)**2)/n))
    qs=[0,.001,.01,.05,.25,.5,.75,.95,.99,.999,1]
    return dict(n=n,mean=mean,std=std,quantiles={str(q):float(v[np.searchsorted(c,max(1,np.ceil(q*n)))]) for q in qs},negative_fraction=float(h[VALUES<0].sum()/n),above_one_fraction=float(h[v>1].sum()/n) if scale==.0001 else None)

def one(row):
    hs=[]; checks={}; monthly=[]
    with np.load(DATA/row.archive,allow_pickle=False) as z:
        available=np.array([i+1 in set(map(int,row.available_months.split(','))) for i in range(12)])
        for key,mask in [('hr_lst','hr_valid'),('lr_lst','lr_valid')]:
            x=z[key]; m=z[mask].astype(bool)
            checks[key]={'masked_nodata':int((m&(x==-32768)).sum()),'valid':int(m.sum()),'total':int(m.size),'eligible_valid':int(m[available].sum()),'eligible_total':int(m[available].size)}
            m=m & (x!=-32768) & available[:,None,None]
            hs.append(np.bincount(x[m].astype(np.int32)+32768,minlength=65536))
            if key=='hr_lst':
                for i in np.flatnonzero(available):
                    v=x[i][m[i]].astype(float)
                    monthly.append(dict(month=int(i+1),n=len(v),sum=float(v.sum()),sum_sq=float(v@v)))
    with np.load(DATA/'hls_l8_conditions_2019_2025_lc08/arrays'/row.mgrs_tile/(row.sequence_id+'.npz'),allow_pickle=False) as z:
        x=z['hls_reflectance_i16']; m=np.broadcast_to(z['hls_valid'].astype(bool),x.shape)
        checks['hls']={'masked_nodata':int((m&(x==-32768)).sum()),'valid':int(m.sum()),'total':int(m.size)}
        for b in range(6):
            good=m[b] & (x[b]!=-32768)
            hs.append(np.bincount(x[b][good].astype(np.int32)+32768,minlength=65536))
    return np.stack(hs),checks,monthly

def main():
    OUT.mkdir(exist_ok=True)
    p=DATA/'phase4_dataset_manifest.csv'; a=pd.read_csv(p)
    b=pd.read_csv(DATA/'hls_l8_conditions_2019_2025_lc08/hls_condition_manifest.csv')
    result={'seed':SEED,'lst_unit':'raw int16; physical scale/offset unconfirmed','hls_scale':.0001,'sampling':'uniform sequence sample within each split; all valid pixels in manifest-available LST months; HLS counted once per sequence; pixel-weighted statistics','manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'census':{'sequences':len(a),'tiles':a.mgrs_tile.nunique(),'months':int(a.available_month_count.sum()),'duplicate_lst_ids':int(a.sequence_id.duplicated().sum()),'duplicate_hls_ids':int(b.sequence_id.duplicated().sum()),'unmatched_ids':len(set(a.sequence_id)^set(b.sequence_id)),'tile_split_overlap':int((a.groupby('mgrs_tile').split.nunique()>1).sum()),'splits':a.groupby('split').agg(sequences=('sequence_id','size'),tiles=('mgrs_tile','nunique'),months=('available_month_count','sum')).to_dict('index'),'month_frequency':a.available_months.str.split(',').explode().value_counts().sort_index().to_dict(),'month_count_distribution':a.available_month_count.value_counts().sort_index().to_dict(),'hls_quality':b[['valid_fraction','snow_fraction','water_fraction']].describe(percentiles=[.01,.5,.99]).to_dict(),'hls_rule_failures':int(((b.valid_fraction<.98)|((b.snow_fraction+b.water_fraction)>=.98)).sum())},'samples':{}}
    names=['hr_lst','lr_lst','B2','B3','B4','B5','B6','B7']; records=[]
    for split,n in [('train',2048),('val',256),('test',256)]:
        sample=a[a.split==split].sample(n=n,random_state=SEED).sort_values('sequence_id')
        sample[['sequence_id','mgrs_tile','split','archive','available_months']].to_csv(OUT/f'{split}_sample.csv',index=False)
        hist=np.zeros((8,65536),dtype=np.int64); checks={}; months=np.zeros((12,3))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,(h,ch,mm) in enumerate(pool.map(one,sample.itertuples(index=False)),1):
                hist+=h
                for k,d in ch.items():
                    acc=checks.setdefault(k,{})
                    for t,v in d.items(): acc[t]=acc.get(t,0)+v
                for d in mm: months[d['month']-1]+=[d['n'],d['sum'],d['sum_sq']]
                if i%256==0: print(split,i,flush=True)
        stats={k:summarize(h,.0001 if j>=2 else 1) for j,(k,h) in enumerate(zip(names,hist))}
        monthly=[{'month':i+1,'n':int(v[0]),'mean':float(v[1]/v[0]),'std':float(np.sqrt(v[2]/v[0]-(v[1]/v[0])**2))} for i,v in enumerate(months) if v[0]]
        result['samples'][split]={'sequences':n,'tiles':sample.mgrs_tile.nunique(),'statistics':stats,'checks':checks,'hr_monthly':monthly}
        np.savez_compressed(OUT/f'{split}_histograms.npz',hist=hist,names=names)
        for k,v in stats.items(): records.append({'split':split,'variable':k,**{t:v[t] for t in ['n','mean','std','negative_fraction','above_one_fraction']},**{'q'+q:x for q,x in v['quantiles'].items()}})
        (OUT/'statistics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame(records).to_csv(OUT/'distribution_summary.csv',index=False)
    print(json.dumps(result['samples'],indent=2),flush=True)

if __name__=='__main__': main()
