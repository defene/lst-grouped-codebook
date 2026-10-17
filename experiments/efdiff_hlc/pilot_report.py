"""Compare the fixed epoch-five checkpoints and identical validation images."""
import json
from pathlib import Path
import numpy as np


def build(directory, local=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    if local:
        run = json.loads((directory/'metrics.json').read_text())
        assert run['status']=='passed'
        metrics = run['metrics']
        data = {v:np.load(directory/f'{v}_predictions.npz') for v in metrics}
    else:
        metrics = {v:json.loads((directory/v/'validation_epoch_5.json').read_text()) for v in ('x0','epsilon')}
        data = {v:np.load(directory/v/'predictions_epoch5.npz') for v in metrics}
    a,b = data['x0'],data['epsilon']
    for key in ('index','record_id','month','target_c','condition_c','mask'):
        assert np.array_equal(a[key],b[key]), key
    expected = 16 if local else 512
    assert metrics['x0']['records'] == metrics['epsilon']['records'] == expected
    lines = ['# HLC local inference check' if local else '# HLC synthetic-degradation pilot', '',
        f'Evaluation shown: {expected} fixed validation examples. Training: 16,384 train patch-months; 5 complete subset epochs, 1,280 updates, 81,920 record presentations per model. No test evaluation.', '',
        'Frozen Prithvi-EO-2.0-300M + EFDiff UNet; synthetic Wald ×32 degradation. Single HLS condition per calendar-month bundle; HLS and fine LST need not be synchronous.', '',
        'Primary comparison: fixed epoch-five EMA. Shared pilot changes: 100-update warmup and EMA decay 0.99 starting at update 100. This short, tile-balanced pilot is not a full-data or converged reproduction.', '',
        '| Model | Validation RMSE (°C) | Validation MAE (°C) | Bicubic RMSE (°C) |',
        '|---|---:|---:|---:|']
    if local:
        lines[2:2] = ['Local end-to-end inference: data read from the local HLC source; CPU float32 frozen encoder + RTX 5090 batch-one bf16 diffusion. File SHA256 hashes match cloud exports. Noise seeds, batch size and encoder precision differ from cloud validation; these 16-example scores must not replace the cloud 512-example scores.', '']
    for v,m in metrics.items():
        lines.append(f"| EFDiff-{v} | {m['rmse_c']:.4f} | {m['mae_c']:.4f} | {m['bicubic_rmse_c']:.4f} |")
    for page in range(4):
        fig,axes = plt.subplots(4,6,figsize=(18,12),layout='constrained')
        for row,i in enumerate(range(page*4,page*4+4)):
            mask = a['mask'][i,0]
            target,base,x0,eps = (x[i,0] for x in (a['target_c'],a['condition_c'],a['prediction_c'],b['prediction_c']))
            lo,hi = np.percentile(target[mask],[1,99])
            lim = max(float(np.percentile(np.abs(x0-target)[mask],98)),float(np.percentile(np.abs(eps-target)[mask],98)),0.1)
            for col,(value,title) in enumerate(zip((target,base,x0,eps,x0-target,eps-target),
                      ('Fine LST reference','Bicubic','EFDiff-x0','EFDiff-epsilon','x0 error','epsilon error'))):
                im = axes[row,col].imshow(np.where(mask,value,np.nan),cmap='coolwarm' if col>=4 else 'turbo',
                                          vmin=-lim if col>=4 else lo,vmax=lim if col>=4 else hi)
                axes[row,col].set_title(title)
                axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
                fig.colorbar(im,ax=axes[row,col],fraction=.046,pad=.02,label='°C')
            axes[row,0].set_ylabel(f"record {int(a['record_id'][i])}\nmonth {a['month'][i]}")
        name = f'comparison_{page+1}.png'
        fig.savefig(directory/name,dpi=120)
        plt.close(fig)
        lines.extend(['',f'![Matched validation examples {page+1}]({name})'])
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    result = {'phase':'complete','scope':'pilot','checkpoint':'fixed_epoch5_ema','test_used':False,'metrics':metrics}
    (directory/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
