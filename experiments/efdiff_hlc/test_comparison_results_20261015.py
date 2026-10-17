"""Verify complete coverage and render fixed test cases; refuses partial scores."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/hlc_unet_mmdit_test_20261015'
NAMES=['bicubic','unet_x0','unet_epsilon','mmdit_5','mmdit_50']
LABELS=['Bicubic','UNet x0 | 15 NFE','UNet epsilon | 50 NFE','MMDiT | 5 NFE','MMDiT | 50 NFE']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    scope='smoke' if args.smoke else 'test';folder=BASE/scope
    done=json.loads((folder/'complete.json').read_text());protocol=json.loads((folder/'protocol.json').read_text())
    count=44 if args.smoke else 141524
    assert done['phase']=='complete' and done['records']==count and done['split']==('val' if args.smoke else 'test')
    index=np.load(ROOT/f'processed_data/v1/lstsr_tb/indices/{done["split"]}.lst_hr.npy',mmap_mode='r')
    sums={name:dict(pixels=0,sse=0.,sae=0.) for name in NAMES};rows=[];captures=[];parts=[]
    for path in sorted((folder/'chunks').glob('*.json')):
        part=json.loads(path.read_text());assert part['protocol']==done['protocol'] and part['start']==len(rows)
        cp=path.parent/part['csv'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==part['csv_sha256']
        with cp.open() as f:
            for row in csv.DictReader(f):
                pos=int(row['index']);assert pos==len(rows) and int(row['record_id'])==int(index[pos,0])
                assert len({int(row[k+'_pixels']) for k in NAMES})==1
                for k in NAMES:
                    for s in ['pixels','sse','sae']:sums[k][s]+=int(row[k+'_'+s]) if s=='pixels' else float(row[k+'_'+s])
                rows.append(row)
        assert part['end']==len(rows) and part['records']==part['end']-part['start'];parts.append(part)
        npz=path.with_suffix('.npz')
        if npz.exists():
            with np.load(npz) as z:
                for i in range(len(z['index'])):captures.append({k:z[k][i] for k in z.files})
    assert len(rows)==count and len({r['index'] for r in rows})==count
    metrics={}
    for k,s in sums.items():
        m=done['metrics'][k]
        assert s['pixels']==m['valid_pixels']
        assert math.isclose(s['sse'],m['squared_error_sum'],rel_tol=1e-11)
        assert math.isclose(s['sae'],m['absolute_error_sum'],rel_tol=1e-11)
        metrics[k]=dict(records=count,valid_pixels=s['pixels'],rmse_c=math.sqrt(s['sse']/s['pixels']),mae_c=s['sae']/s['pixels'],nfe=m['nfe'])
    out=BASE/('smoke_verified' if args.smoke else 'results');out.mkdir(exist_ok=True)
    (out/'verification.json').write_text(json.dumps(dict(complete_coverage=True,unique_ordered_members=True,per_record_sums_verified=True,records=count,metrics=metrics),indent=2))
    if args.smoke:print('Smoke coverage and sums verified:',count);return
    with (out/'main_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['model','records','valid_pixels','rmse_c','mae_c','nfe']);writer.writeheader()
        writer.writerows(dict(model=k,**v) for k,v in metrics.items())
    with (out/'per_record_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    captures.sort(key=lambda x:int(x['index']))
    assert [int(c['index']) for c in captures]==np.linspace(0,count-1,16,dtype=int).tolist()
    selected=np.linspace(0,15,8,dtype=int)
    for case,j in enumerate(selected,1):
        sample=captures[j];mask=sample['mask'][0].astype(bool);target=sample['target_c'][0]
        lo,hi=np.percentile(target[mask],[1,99]);rgb=np.clip(sample['HLS_raw'][[2,1,0]].transpose(1,2,0)/.3,0,1)**(1/2.2)
        rgb[~sample['HLS_valid'][0].astype(bool)]=.65
        fig,axes=plt.subplots(3,3,figsize=(11,11.7));fig.subplots_adjust(left=.025,right=.875,bottom=.035,top=.9,wspace=.045,hspace=.21)
        axes[0,0].imshow(rgb);axes[0,0].set_title('HLS RGB (associated date)')
        views=[(sample['synthetic_coarse_c'][0],'Synthetic coarse 7x7'),(target,'Reference fine LST')]
        for ax,(v,title) in zip(axes[0,1:],views):
            im=ax.imshow(v if v.shape==(7,7) else np.where(mask,v,np.nan),cmap='inferno',vmin=lo,vmax=hi);ax.set_title(title)
        for ax,name,label in zip(list(axes[1])+[axes[2,0]],['unet_x0','unet_epsilon','mmdit_50','mmdit_5'],['UNet x0 | 15 NFE','UNet epsilon | 50 NFE','MMDiT | 50 NFE','MMDiT | 5 NFE']):
            v=sample[name][0];rmse=np.sqrt(np.mean((v[mask].astype('float64')-target[mask])**2))
            ax.imshow(np.where(mask,v,np.nan),cmap='inferno',vmin=lo,vmax=hi);ax.set_title(f'{label}\nRMSE {rmse:.3f} C')
        for ax,name in zip(axes[2,1:],['mmdit_50','mmdit_5']):
            err=ax.imshow(np.where(mask,sample[name][0]-target,np.nan),cmap='RdBu_r',vmin=-5,vmax=5);ax.set_title(name+' prediction - reference')
        for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(im,cax=fig.add_axes([.9,.4,.018,.4]),label='Temperature (C)')
        fig.colorbar(err,cax=fig.add_axes([.9,.065,.018,.23]),label='Error (C)')
        fig.suptitle(f'HLC TEST | case {case} | record {int(sample["record_id"])} | month {int(sample["month"])}\nFixed checkpoints; different historical training protocols',fontsize=13)
        fig.savefig(out/f'case_{case:02d}.png',dpi=150);plt.close(fig)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
