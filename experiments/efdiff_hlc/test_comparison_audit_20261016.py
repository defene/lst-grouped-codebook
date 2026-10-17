"""Additional member/month and finite-value audit, plus fixed-case residuals."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/hlc_unet_mmdit_test_20261015'
OUT=BASE/'results'
NAMES=['bicubic','unet_x0','unet_epsilon','mmdit_5','mmdit_50']
LABELS=['Bicubic','UNet x0 / 15','UNet epsilon / 50','MMDiT / 5','MMDiT / 50']
idx=np.load(ROOT/'processed_data/v1/lstsr_tb/indices/test.lst_hr.npy')
seen=set();wins={k:0 for k in ['mmdit_5_vs_unet_x0','mmdit_5_vs_unet_epsilon','mmdit_5_vs_mmdit_50']}
with (OUT/'per_record_metrics.csv').open() as f:
    for pos,row in enumerate(csv.DictReader(f)):
        pair=(int(row['record_id']),int(row['month']))
        assert pair==(int(idx[pos,0]),int(idx[pos,1])+1) and pair not in seen
        seen.add(pair)
        for k in NAMES:
            assert int(row[k+'_pixels'])>0
            assert all(np.isfinite(float(row[k+'_'+x])) and float(row[k+'_'+x])>=0 for x in ['sse','sae'])
        for k in wins:
            other=k.split('_vs_')[1]
            wins[k]+=float(row['mmdit_5_sse'])<float(row[other+'_sse'])
assert len(seen)==141524
caps=[]
for p in sorted((BASE/'test/chunks').glob('*.npz')):
    with np.load(p) as z:
        caps.extend({k:z[k][i] for k in z.files} for i in range(len(z['index'])))
caps.sort(key=lambda s:int(s['index']))
assert [int(s['index']) for s in caps]==np.linspace(0,141523,16,dtype=int).tolist()
cases=[]
fig,axes=plt.subplots(8,4,figsize=(11,22))
for ci,j in enumerate(np.linspace(0,15,8,dtype=int)):
    s=caps[j];mask=s['mask'][0].astype(bool);target=s['target_c'][0].astype(np.float64)
    case={'case':ci+1,'index':int(s['index']),'record_id':int(s['record_id']),'month':int(s['month'])}
    for k in NAMES:
        pred=s['condition_c'][0] if k=='bicubic' else s[k][0]
        d=pred.astype(np.float64)[mask]-target[mask]
        assert np.isfinite(d).all()
        case[k]={'rmse_c':float(np.sqrt(np.mean(d*d))),'mae_c':float(np.abs(d).mean()),'bias_c':float(d.mean())}
    cases.append(case)
    for col,(k,label) in enumerate(zip(NAMES[1:],LABELS[1:])):
        ax=axes[ci,col]
        im=ax.imshow(np.where(mask,s[k][0]-target,np.nan),cmap='RdBu_r',vmin=-5,vmax=5)
        ax.set_xticks([]);ax.set_yticks([])
        if ci==0:ax.set_title(label+' NFE',fontsize=10)
        if col==0:ax.set_ylabel(f'Case {ci+1}\nrecord {case["record_id"]}',fontsize=9)
        ax.set_xlabel(f'RMSE {case[k]["rmse_c"]:.3f} | bias {case[k]["bias_c"]:+.3f} C',fontsize=8)
fig.subplots_adjust(left=.09,right=.9,bottom=.03,top=.96,hspace=.17,wspace=.04)
fig.colorbar(im,cax=fig.add_axes([.93,.35,.014,.3]),label='Prediction - reference (C)')
fig.suptitle('HLC full test | 8 fixed cases | common residual scale [-5, 5] C',fontsize=13)
fig.savefig(OUT/'residuals_8cases.png',dpi=150);plt.close(fig)
(OUT/'case_metrics.json').write_text(json.dumps(cases,indent=2))
sources={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'test/complete.json',BASE/'test/protocol.json',OUT/'verification.json',OUT/'per_record_metrics.csv']}
audit=dict(records=141524,chunks=len(list((BASE/'test/chunks').glob('*.json'))),record_month_pairs_unique=True,record_month_order_matches_source=True,all_per_record_metrics_finite=True,per_record_wins=wins,fixed_cases=cases,sources_sha256=sources)
(OUT/'extended_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({k:v for k,v in audit.items() if k not in ['fixed_cases','sources_sha256']},indent=2))
