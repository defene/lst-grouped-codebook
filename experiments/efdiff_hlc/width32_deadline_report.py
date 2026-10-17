"""Report only matched training milestones; never label a partial pass an epoch."""
import json
from pathlib import Path
import numpy as np


def build(root,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    out=root/'comparisons'/str(records)
    out.mkdir(parents=True,exist_ok=True)
    metrics={v:json.loads((root/'train'/v/f'validation_{records}.json').read_text()) for v in ('x0','epsilon')}
    provenance={v:json.loads((root/'train'/v/'provenance.json').read_text()) for v in metrics}
    for key in ('stats_hash','fingerprint','stage_members_sha256'):
        assert provenance['x0'][key]==provenance['epsilon'][key],key
    arrays={v:np.load(root/'train'/v/f'predictions_{records}.npz') for v in metrics}
    a,b=arrays['x0'],arrays['epsilon']
    for key in ('index','record_id','month','target_c','condition_c','mask'):
        assert np.array_equal(a[key],b[key]),key
    assert all(m['records']==512 and m['training_records']==records for m in metrics.values())
    lines=['# HLC narrowed-UNet matched-stage comparison','',
        f'Each model: {records:,} unique training records, {records/2608772:.2%} of one full epoch. '+
        f'Complete epochs: {int(records==2608772)}. Equal membership, order rule, effective batch 64.','',
        'One H200 shared sequentially. UNet widths [32,64,128,256,512]; 66,900,097 trainable parameters. '+
        'Frozen Prithvi encoder unchanged. Synthetic Wald x32; HLC v1; full-train normalization.','',
        'Fixed stage EMA; 512 fixed validation records; no test evaluation. '+
        'This deadline-limited comparison is separate from the default five-epoch formal budget.','',
        '| Model | RMSE C | MAE C | Bicubic RMSE C |','|---|---:|---:|---:|']
    for v,m in metrics.items():
        lines.append(f"| {v} | {m['rmse_c']:.4f} | {m['mae_c']:.4f} | {m['bicubic_rmse_c']:.4f} |")
    for page in range(4):
        fig,axes=plt.subplots(4,6,figsize=(18,12),layout='constrained')
        for row,i in enumerate(range(page*4,page*4+4)):
            mask=a['mask'][i,0]
            target,base,x0,eps=(x[i,0] for x in (a['target_c'],a['condition_c'],a['prediction_c'],b['prediction_c']))
            lo,hi=np.percentile(target[mask],[1,99])
            lim=max(float(np.percentile(np.abs(x0-target)[mask],98)),float(np.percentile(np.abs(eps-target)[mask],98)),.1)
            for col,(value,title) in enumerate(zip((target,base,x0,eps,x0-target,eps-target),
                ('Reference','Bicubic','EFDiff-x0','EFDiff-epsilon','x0 error','epsilon error'))):
                im=axes[row,col].imshow(np.where(mask,value,np.nan),cmap='coolwarm' if col>=4 else 'turbo',
                    vmin=-lim if col>=4 else lo,vmax=lim if col>=4 else hi)
                axes[row,col].set_title(title)
                axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                fig.colorbar(im,ax=axes[row,col],fraction=.046,pad=.02,label='C')
            axes[row,0].set_ylabel(f"record {int(a['record_id'][i])}, month {int(a['month'][i])}")
        name=f'comparison_{page+1}.png'
        fig.savefig(out/name,dpi=120)
        plt.close(fig)
        lines.extend(['',f'![Predictions and signed errors]({name})'])
    result={'phase':'complete','training_records_per_model':records,'epoch_fraction':records/2608772,
        'complete_epochs':int(records==2608772),'validation_records':512,'test_used':False,'metrics':metrics}
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'latest_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
