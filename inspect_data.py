from pathlib import Path
import io, json, zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
P = ROOT / 'lstsr_tb_final_upload_20260721'
out = ROOT / 'data_inspection'
out.mkdir(exist_ok=True)
with zipfile.ZipFile(P / 'lst_hls_metadata_and_reports.zip') as z:
    def read(suffix):
        return pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith(suffix))))
    a = read('phase4_dataset_manifest.csv')
    b = read('hls_condition_manifest.csv')
    c = read('phase3_tile_variant_months_merged.csv')
r = {'sequences':len(a),'hls_rows':len(b),'tiles':a.mgrs_tile.nunique(),
     'usable_months':int(a.available_month_count.sum()),
     'duplicate_lst_ids':int(a.sequence_id.duplicated().sum()),
     'duplicate_hls_ids':int(b.sequence_id.duplicated().sum()),
     'missing_hls':len(set(a.sequence_id)-set(b.sequence_id)),
     'extra_hls':len(set(b.sequence_id)-set(a.sequence_id)),
     'splits':a.groupby('split').agg(sequences=('sequence_id','size'),tiles=('mgrs_tile','nunique'),months=('available_month_count','sum')).to_dict('index'),
     'tiles_in_multiple_splits':int((a.groupby('mgrs_tile').split.nunique()>1).sum()),
     'month_count_distribution':a.available_month_count.value_counts().sort_index().to_dict(),
     'source_year_count_distribution':a.distinct_source_year_count.value_counts().sort_index().to_dict(),
     'sequence_modes':a.sequence_mode.value_counts().to_dict(),
     'hls_quality':b[['valid_fraction','snow_fraction','water_fraction']].describe(percentiles=[.01,.5,.95,.99]).to_dict(),
     'hls_snow_above_half':int((b.snow_fraction>.5).sum()),
     'hls_quality_rule_failures':int(((b.valid_fraction<.98)|((b.snow_fraction+b.water_fraction)>=.98)).sum()),
     'hls_dates':[b.hls_datetime_utc.min(),b.hls_datetime_utc.max()],
     'phase3_resampling':c.modis_lst_resampling.value_counts(dropna=False).to_dict(),
     'phase3_primary_scene_examples':c.primary_scene_id.dropna().head(3).tolist(),
     'phase3_modis_examples':c.modis_image_id.dropna().head(3).tolist()}
j=a.merge(b,on='sequence_id',suffixes=('_lst','_hls'),validate='one_to_one')
r['join_mismatches']={k:int((j[k+'_lst']!=j[k+'_hls']).sum()) for k in ['mgrs_tile','split','x0','y0','patch_size']}
r['month_frequency']=a.available_months.str.split(',').explode().value_counts().sort_index().to_dict()
archives={}
for pattern,label in [('lst_phase4_arrays*.zip','lst'),('hls_condition_arrays*.zip','hls')]:
    index={}
    duplicates=0
    for path in sorted(P.glob(pattern)):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith('.npz'):
                    sid=Path(n).stem
                    duplicates+=int(sid in index)
                    index[sid]=(path,n)
    archives[label]=index
    r[label+'_archive_inventory']={'files':len(index),'duplicates':duplicates,'missing_manifest_ids':len(set(a.sequence_id)-set(index)),'extra_ids':len(set(index)-set(a.sequence_id))}
samples=a.sample(16,random_state=20260905)
checks=[]
for row in samples.itertuples():
    item={'sequence_id':row.sequence_id,'split':row.split}
    for label in ['lst','hls']:
        path,n=archives[label][row.sequence_id]
        with zipfile.ZipFile(path) as z, np.load(io.BytesIO(z.read(n)),allow_pickle=False) as npz:
            item[label+'_schema']={k:{'shape':list(npz[k].shape),'dtype':str(npz[k].dtype)} for k in npz.files}
            for key,mask in ([('hr_lst','hr_valid'),('lr_lst','lr_valid')] if label=='lst' else [('hls_reflectance_i16','hls_valid')]):
                x=npz[key]; m=np.broadcast_to(npz[mask].astype(bool),x.shape)
                item[key]={'valid_fraction':float(m.mean()),'valid_raw_quantiles':np.quantile(x[m],[0,.01,.5,.99,1]).tolist(),'nodata_in_valid':int((x[m]==-32768).sum()),'non_nodata_outside_mask':int((x[~m]!=-32768).sum())}
            if label=='lst':
                months=np.flatnonzero(npz['hr_valid'].any(axis=(1,2)) & npz['lr_valid'].any(axis=(1,2)))+1
                item['available_months_match']=set(months)==set(map(int,row.available_months.split(',')))
    checks.append(item)
r['array_samples']=checks
(out/'inspection.json').write_text(json.dumps(r,indent=2,default=int),encoding='utf-8')
print(json.dumps({k:v for k,v in r.items() if k!='array_samples'},indent=2,default=int))
print('Sample checks:',len(checks),'month mismatches',sum(not x['available_months_match'] for x in checks))
