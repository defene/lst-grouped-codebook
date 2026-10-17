"""Reproduce comparison tables and scientific training plots from local logs."""
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import ROOT,save

METHODS=['baseline','ema','ema_maint']
LABELS={'baseline':'Gradient codebook','ema':'EMA only','ema_maint':'EMA + K-means + resets'}
COLORS={'baseline':'#525866','ema':'#2463a6','ema_maint':'#c67217'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    folder=ROOT/args.output;results=json.loads((folder/'results.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text());assert len(results)==6
    assert json.loads((folder/'complete.json').read_text())['completed']
    for name,digest in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    rows=[];logs={}
    for r in results:
        family,method=r['arm'].split(':');path=folder/(family+'_'+method)
        log=[json.loads(line) for line in (path/'training.jsonl').read_text().splitlines()];logs[r['arm']]=log
        assert [x['step'] for x in log]==list(range(1,129))
        assert all(np.isfinite(x['total']) and np.isfinite(x['gradient_norm']) for x in log)
        metrics=r['final']['metrics'];usage=metrics['pixel10']['code_usage'];groups=usage.get('per_group',[usage])
        evaluations=json.loads((path/'evaluations.json').read_text());assert [x['step'] for x in evaluations]==[0,64,128]
        chosen=min(evaluations,key=lambda x:x['metrics']['pixel10']['rmse_c'])
        assert chosen['step']==r['best_step']
        rows.append(dict(family=family,method=method,step=128,pixel10_rmse_c=metrics['pixel10']['rmse_c'],
            clean_rmse_c=metrics['clean']['rmse_c'],missing_rmse_c=metrics['pixel10']['missing_rmse_c'],
            mae_c=metrics['pixel10']['mae_c'],best_step=r['best_step'],best_rmse_c=r['best_pixel10_rmse_c'],
            used_codes=[g['used_codes'] for g in groups],perplexity=[g['perplexity'] for g in groups],
            top1=[g['top1_fraction'] for g in groups],resets=r['reset_counts'],peak_gib=r['peak_allocated_gib'],seconds=r['seconds']))
    for family in ['vq','grouped_g8k256']:
        entries=[r for r in results if r['arm'].startswith(family+':')]
        assert len({r['initial_backbone_sha256'] for r in entries})==1
        assert len({r['final']['metrics']['pixel10']['valid_pixels'] for r in entries})==1
    save(folder/'comparison.json',rows)
    save(folder/'selected_checkpoints.json',[
        dict(arm=r['arm'],step=r['best_step'],subset_rmse_c=r['best_pixel10_rmse_c'],
             path=r['arm'].replace(':','_')+f"/step_{r['best_step']:04d}.pt",
             trained=r['best_step']>0) for r in results])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for row,family in enumerate(['vq','grouped_g8k256']):
        for method in METHODS:
            log=logs[family+':'+method];steps=[x['step'] for x in log]
            pixel=np.array([x['pixel'] for x in log]);q=np.array([x.get('quantizer/quantization_mse',x.get('quantizer/commitment_loss')) for x in log])
            for col,ys in [(0,pixel),(1,q)]:
                axes[row,col].plot(steps,ys,color=COLORS[method],alpha=.20,linewidth=.7)
                axes[row,col].plot(steps[7:],np.convolve(ys,np.ones(8)/8,mode='valid'),color=COLORS[method],label=LABELS[method],linewidth=1.8)
            ev=json.loads((folder/(family+'_'+method)/'evaluations.json').read_text())
            axes[row,2].plot([x['step'] for x in ev],[x['metrics']['pixel10']['rmse_c'] for x in ev],
                marker={'baseline':'o','ema':'s','ema_maint':'^'}[method],color=COLORS[method],label=LABELS[method])
        axes[row,1].set_yscale('log')
        for col,title in enumerate(['Training physical MSE / sigma^2','Training quantization MSE (log)','Fixed validation pixel10 RMSE (C)']):
            axes[row,col].set_title(('VQ-VAE' if row==0 else 'Grouped')+' | '+title)
            axes[row,col].set_xlabel('Optimizer updates');axes[row,col].set_xlim(0,128);axes[row,col].set_xticks([0,32,64,96,128])
            axes[row,col].grid(axis='y',color='#dddddd',linewidth=.5)
        axes[row,0].legend(fontsize=9)
    fig.suptitle('Local EMA pilot: G, seed 17, 2,048 training records per arm\n64 validation records x 2 fixed pixel10 views; faint = raw, bold = trailing 8 updates',fontsize=13)
    fig.savefig(folder/'curves.png',dpi=170);plt.close(fig)
    # Separate group dictionaries and scales; do not pool numerical IDs.
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,method in zip(axes,METHODS):
        r=next(x for x in results if x['arm']=='grouped_g8k256:'+method)
        groups=r['final']['metrics']['pixel10']['code_usage']['per_group']
        values=np.array([[s['perplexity'] for s in g['scales']] for g in groups])
        im=ax.imshow(values,aspect='auto',cmap='cividis',vmin=1,vmax=256)
        ax.set_title(LABELS[method]);ax.set_xticks(range(10),[s['scale'] for s in groups[0]['scales']]);ax.set_yticks(range(8))
        ax.set_xlabel('Scale grid side');ax.set_ylabel('Group')
    fig.colorbar(im,ax=axes,shrink=.8,label='Per-scale perplexity (1-256)')
    fig.suptitle('Grouped code usage on fixed pixel10 validation views, update 128')
    fig.savefig(folder/'grouped_perplexity.png',dpi=170);plt.close(fig)
    lines=['LOCAL EMA PILOT — COMPLETED','',
        'HLP v2 train/val; G representation; scratch seed17; no test data.',
        '128 updates x effective batch16 = 2048 training records/model (0.00935 full train epochs).',
        'Microbatch2; BF16; AdamW lr1e-4, 10-update warmup then cosine; clip1.',
        'Validation: fixed random 64 records, clean1view and pixel10 2views; raw valid-pixel weighted Celsius error.',
        'All reported final values are step128. Best candidates restricted to steps0/64/128 on this subset.',
        'This is an early technical pilot, not a formal 5-epoch comparison or full validation selection.','',
        'family | method | pixel10 RMSE C | clean RMSE C | missing RMSE C | best step | used codes | perplexity']
    for r in rows:
        lines.append(f"{r['family']} | {r['method']} | {r['pixel10_rmse_c']:.6f} | {r['clean_rmse_c']:.6f} | {r['missing_rmse_c']:.6f} | {r['best_step']} | {r['used_codes']} | "+str([round(v,2) for v in r['perplexity']]))
    lines.extend(['','Interpretation limits:',
        '- Maintenance arm bundles K-means and resets: cannot attribute improvement to one alone.',
        '- Ordinary EMA removes gradient codebook loss; compare physical error / quantization MSE, not total loss.',
        '- Raw quantization MSE also depends on learned latent scale; it is diagnostic, not a standalone quality ranking.',
        '- Grouped retains fitting gradients to Phi; EMA aggregates its shared per-group dictionaries across scales.',
        '- Grouped has more parameters and 17x nominal index bits than ordinary VQ; this is not equal-rate architecture attribution.',
        '- Existing oracle scalar-mean side information is retained.',
        '- One seed and small random validation subset; no convergence or population significance claim.',
        '- Training batches change each step; raw fluctuations conflate sample difficulty and optimization.',
        '- Usage includes all transmitted latent positions, including reference-invalid positions.',
        '', 'Checks: six complete logs; finite objectives/gradient norms; matched backbone initialization within family;',
        'identical validation pixel denominators; checkpoint selection recomputed; execution source hashes unchanged.',
        'Checkpoints: each arm has step_0000.pt, step_0064.pt, step_0128.pt and last_resume.pt.'])
    (folder/'RESULTS.txt').write_text('\n'.join(lines),encoding='utf-8')
    receipt={'schemaVersion':1,'items':[{'id':'local-ema-pilot','title':'本机 VQ-VAE 与 Grouped EMA 短程试验','queries':[{
        'id':'local-six-arms','source':{'label':'Local EMA pilot','files':['comparison.json','evaluations.json','manifest.json'],
        'metricDefinitions':[{'name':'Pixel10 RMSE','definition':'Pixel10 RMSE 按固定验证子集的有效像素加权，以摄氏度计算。'}],
        'filters':['G 表示','随机初始化 seed 17','每组训练 2048 条记录、128 次更新','固定验证 64 条记录、每条 2 个 pixel10 视图'],
        'caveats':['这是短程技术试验，尚不能判断收敛后效果。','K-means 与死码维护合并在一组，不能区分二者各自贡献。','Grouped 和普通 VQ 的模型大小与码率不同。']},
        'columns':['family','method','pixel10_rmse_c','clean_rmse_c','best_step','peak_gib'],
        'rows':[{k:r[k] for k in ['family','method','pixel10_rmse_c','clean_rmse_c','best_step','peak_gib']} for r in rows]}]}]}
    save(folder/'sources.json',receipt)
    with zipfile.ZipFile(folder/'source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        names=set(manifest['source_sha256'])
        names.update(str(p.relative_to(ROOT)) for p in Path(__file__).parent.glob('*.py'))
        names.update(['tests/test_lst_ema_pilot_20261014.py','experiments/paper1/lst_ema_pilot_20261014/README.txt'])
        for name in sorted(names):archive.write(ROOT/name,name)
    save(folder/'analysis_verification.json',dict(passed=True,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_archive_sha256=hashlib.sha256((folder/'source_snapshot.zip').read_bytes()).hexdigest(),arms=6))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
