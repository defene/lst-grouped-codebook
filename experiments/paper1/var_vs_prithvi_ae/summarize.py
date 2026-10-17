"""Reproducible experiment tables and scientific reconstruction figures from saved evaluations."""
from pathlib import Path
import argparse
import json
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eo_data import open_dataset
from eo_data.core import json_write,sha256_file

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def summarize(output,phase,language='en'):
    if read(output/'status.json')['status']!='training_and_evaluation_complete':
        raise ValueError('All eight runs and both evaluations must finish before summarizing')
    report=output/'report';report.mkdir(exist_ok=True)
    records=[];sources={};coverage={};curves={}
    for m in ('hls','lst'):
        for b in ('conv','prithvi'):
            for n in ('clean','noisy'):
                name=f'{phase}_{m}_{b}_{n}_seed17';run=output/name
                logs=[json.loads(s) for s in (run/'train.jsonl').read_text().splitlines()]
                curves[name]=[(x['step'],x['reconstruction']) for x in logs]
                expected=set(np.load(run/'train_record_ids.npy').tolist())
                seen={int(s.rsplit(':',1)[1]) for x in logs for s in x['sample_ids']}
                coverage[name]=dict(available=len(expected),unique_seen=len(seen),
                    all_records_seen=seen==expected,sample_presentations=sum(x['batch_samples'] for x in logs))
                if seen!=expected:raise ValueError(name+': full configured training membership not covered')
                for condition in ('clean','noisy10'):
                    path=run/('eval_'+condition)/'metrics.json';d=read(path)
                    records.append(dict(name=name,modality=m,backbone=b,training=n,evaluation=condition,
                        evaluated_pairs=d['evaluated_pairs'],
                        corrupt_rmse=d['regions'].get(m+'/corrupt',{}).get('rmse'),
                        retained_rmse=d['regions'][m+'/retained']['rmse'],
                        **d['regions'][m+'/all']))
                    sources[str(path.relative_to(output))]=sha256_file(path)
    json_write(report/'results.json',dict(phase=phase,rows=records,training_coverage=coverage,sources=sources,
        caveats=['one seed','validation only, no independent test','different continuous/discrete representations; not rate matched',
                 'small gate only, not convergence' if phase=='pilot' else 'first full-data budget; convergence not established']))
    text=['# A02: Convolutional VQ-VAE versus continuous Prithvi AE','',
        'The convolutional VQ-VAE comes from the FoundationVision/VAR repository. VAR means Visual Autoregressive Modeling; its autoregressive generation model is not trained in A02. Model labels below use CNN VQ-VAE to avoid confusion.', '',
        'Prithvi uses continuous native ViT features, with no VQ or KL objective. All eight models start from random weights.',
        '',f'Phase: **{phase}**. '+('64 train / 64 val records; 128 steps per run. These are implementation checks, not final rankings.' if phase=='pilot' else
        '230,659 train / 11,757 val records; 10,813 steps per run. Every training record was consumed; three complete passes plus 40 samples.'),
        '', '| Modality | Model | Train input | Eval input | RMSE | 95% tile CI | MAE | Val samples |',
        '|---|---|---|---|---:|---|---:|---:|']
    for row in records:
        ci=row['rmse_tile_ci95'];label='CNN VQ-VAE' if row['backbone']=='conv' else 'Prithvi AE'
        ci_text=f'[{ci[0]:.4f}, {ci[1]:.4f}]' if ci else 'unavailable'
        text.append(f"| {row['modality'].upper()} | {label} | {row['training']} | {row['evaluation']} | {row['rmse']:.4f} | {ci_text} | {row['mae']:.4f} | {row['samples']} |")
    text+=['','HLS units: reflectance. LST units: °C. Metrics average the two realizations within each sample and weight samples equally; confidence intervals resample tiles.',
        '','Noise means 10% synthetic missing-region input with a known mask, not additive Gaussian noise. No clean examples are mixed into noisy training.',
        '','The models differ in backbone and representation type. This comparison does not isolate VQ alone and does not match compression rate. There is one training seed and no independent test split. VAR uses its official codec architecture with EO channel/mask adaptations; the training objective is the frozen MSE + gradient + VQ objective, not a reproduction of the full original VAR tokenizer training recipe.']
    if phase=='full':
        text+=['','## Missing and retained regions','',
            'All rows below evaluate the same 10% missing-input condition. Overall error can be dominated by the 90% retained region.',
            '', '| Modality | Model | Train input | Missing-region RMSE | Retained-region RMSE |',
            '|---|---|---|---:|---:|']
        for row in records:
            if row['evaluation']!='noisy10':continue
            label='CNN VQ-VAE' if row['backbone']=='conv' else 'Prithvi AE'
            text.append(f"| {row['modality'].upper()} | {label} | {row['training']} | {row['corrupt_rmse']:.5f} | {row['retained_rmse']:.5f} |")
        lookup={(r['modality'],r['backbone'],r['training'],r['evaluation']):r for r in records}
        text+=['','## Interpretation','',
            'Noise training improves Prithvi overall missing-input RMSE by '+
            ' / '.join(f"{m.upper()} {(1-lookup[(m,'prithvi','noisy','noisy10')]['rmse']/lookup[(m,'prithvi','clean','noisy10')]['rmse'])*100:.1f}%" for m in ('hls','lst'))+
            '. Clean-input reconstruction becomes somewhat worse. These are observed single-seed changes.',
            '', 'After noise training, VAR and Prithvi have similar missing-region RMSE: HLS '+
            ' versus '.join(f"{lookup[('hls',b,'noisy','noisy10')]['corrupt_rmse']:.5f}" for b in ('conv','prithvi'))+
            '; LST '+ ' versus '.join(f"{lookup[('lst',b,'noisy','noisy10')]['corrupt_rmse']:.3f} °C" for b in ('conv','prithvi'))+
            ' (VAR versus Prithvi). Prithvi has lower retained-region error. No significance claim is made for these small missing-region differences.',
            '', f"LST VAR trained only with missing inputs gives {lookup[('lst','conv','noisy','clean')]['rmse']:.3f} °C RMSE on clean input, versus {lookup[('lst','conv','clean','clean')]['rmse']:.3f} °C for the clean-trained VAR. Its missing-input RMSE is {lookup[('lst','conv','noisy','noisy10')]['rmse']:.3f} °C. This is a material failure to generalize across input conditions; the specific mechanism has not been isolated. Raw per-sample metrics, data identities, and common preview inputs were checked in completion_audit.json.",
            '', '## Convergence and reconstruction limitations','',
            'The three-pass budget does not establish full convergence. The fixed example still shows strong smoothing in VAR and artificial fine texture in Prithvi. This is evidence of inadequate detail reconstruction; it does not distinguish an information bottleneck from insufficient optimization or establish that more epochs alone would fix it.',
            '', '| Model | Mean reconstruction MSE, steps 8814–9813 | Mean, steps 9814–10813 | Reduction |',
            '|---|---:|---:|---:|']
        for name,curve in curves.items():
            values=np.asarray(curve)[:,1]
            before=float(values[-2000:-1000].mean());after=float(values[-1000:].mean())
            text.append(f'| {name.removeprefix("full_").removesuffix("_seed17")} | {before:.6f} | {after:.6f} | {(1-after/before)*100:.2f}% |')
        text+=['', 'These training windows contain different shuffled samples, so their changes are descriptive. Prithvi continues improving on the fixed 256-record diagnostic validation subset. LST VAR is closer to a plateau, and its late missing-input diagnostic error rises slightly; a blanket claim that every branch only needs more training is unsupported. The fixed final checkpoint is retained for every model, regardless of the best diagnostic checkpoint.',
            '', 'Audit: [completion_audit.json](completion_audit.json) verifies all 8 runs, 692,017 training presentations per run, identical training order, all 11,757 validation records with two views, frozen source/data/config identities, checkpoint hashes, and every region RMSE recomputed from saved sample rows.']
    text+=[
        '','Figures use the first fixed validation example; display clipping does not affect any metric. Raw arrays remain in each evaluation’s `preview.npz`.',
        '','![HLS reconstruction](hls_reconstruction.png)','', '![LST reconstruction](lst_reconstruction.png)','',
        '![Training reconstruction loss](reconstruction_loss.png)','']
    # VAR is a source repository name, not the model under evaluation.
    text=[line.replace('VAR and Prithvi','the convolutional VQ-VAE and Prithvi')
          .replace('(VAR versus Prithvi)', '(convolutional VQ-VAE versus Prithvi)')
          .replace('LST VAR', 'LST convolutional VQ-VAE')
          .replace('clean-trained VAR', 'clean-trained convolutional VQ-VAE')
          .replace('smoothing in VAR', 'smoothing in the convolutional VQ-VAE') for line in text]
    zh=language=='zh';suffix='_zh' if zh else ''
    if zh:
        from matplotlib import font_manager
        font=Path('C:/Windows/Fonts/msyh.ttc')
        if not font.exists():raise FileNotFoundError('Chinese report requires Microsoft YaHei font')
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
        from experiments.paper1.var_vs_prithvi_ae.report_zh import write_report
        write_report(report,phase,records,coverage,curves)
    else:(report/'RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    labels=(['干净参考图','模型输入','卷积 VQ-VAE\n干净训练','卷积 VQ-VAE\n加噪训练','Prithvi 连续 AE\n干净训练','Prithvi 连续 AE\n加噪训练'] if zh else
            ['Clean reference','Observed input','VQ-VAE / clean train','VQ-VAE / noisy train','Prithvi / clean train','Prithvi / noisy train'])
    for m in ('hls','lst'):
        ds=open_dataset('prithvi',m,'val',root=ROOT)
        std=np.asarray(ds.stats['std'])[:,None,None];mean=np.asarray(ds.stats['mean'])[:,None,None];ds.close()
        fig,axes=plt.subplots(2,6,figsize=(16,6.4),layout='constrained')
        first_id=None;display_limits=None;im=None
        for row,condition in enumerate(('clean','noisy10')):
            previews=[]
            for b,n in [('conv','clean'),('conv','noisy'),('prithvi','clean'),('prithvi','noisy')]:
                folder=output/f'{phase}_{m}_{b}_{n}_seed17'/('eval_'+condition)
                with np.load(folder/'preview.npz') as z:previews.append({k:z[k] for k in z.files})
                if first_id is None:
                    with (folder/'samples.jsonl').open() as f:first_id=json.loads(next(f))['sample_id']
            base=previews[0];reference=base[m+'_reference_mask'][0].astype(bool)
            arrays=[base[m+'_target'],base[m+'_image']]+[p[m+'_prediction'] for p in previews]
            masks=[reference,base[m+'_input_mask'][0].astype(bool)]+[reference]*4
            for column,(array,mask) in enumerate(zip(arrays,masks)):
                physical=array*std+mean
                ax=axes[row,column];ax.set_xticks([]);ax.set_yticks([])
                if m=='hls':
                    rgb=np.clip(physical[[2,1,0]].transpose(1,2,0)/.3,0,1)**(1/2.2)
                    rgb[~mask]=.35;ax.imshow(rgb,interpolation='nearest')
                else:
                    value=physical[0]/100
                    if display_limits is None:display_limits=np.quantile(value[reference],[.01,.99])
                    cmap=plt.get_cmap('inferno').copy();cmap.set_bad('#595959')
                    im=ax.imshow(np.ma.masked_where(~mask,value),vmin=display_limits[0],vmax=display_limits[1],cmap=cmap,interpolation='nearest')
                if row==0:ax.set_title(labels[column],fontsize=10)
                if column==0:ax.set_ylabel(('干净输入验证' if row==0 else '10% 缺失输入验证') if zh else ('Clean evaluation' if row==0 else '10% missing evaluation'),fontsize=10)
        if im is not None:fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,extend='both',label='地表温度（°C）；统一使用参考图 1%–99% 分位数色标' if zh else 'Surface temperature (°C); shared reference 1–99% limits')
        if zh:
            display='RGB：反射率 ÷ 0.3，伽马 1/2.2；灰色表示无效或缺失像素' if m=='hls' else '色标范围统一；灰色表示无效或缺失像素；超出色标的值仅在显示时截断'
        else:display='RGB: reflectance / 0.3, gamma 1/2.2; gray = invalid/missing' if m=='hls' else 'Shared color limits; gray = invalid/missing; out-of-range values clipped for display'
        phase_label=('全量实验' if phase=='full' else '小范围测试') if zh else phase
        fig.suptitle(f'{m.upper()} · {phase_label} · {first_id}\n{display}',fontsize=12)
        fig.savefig(report/f'{m}_reconstruction{suffix}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,m in zip(axes,('hls','lst')):
        for b,color in [('conv','#16658B'),('prithvi','#B66A28')]:
            for n,style in [('clean','-'),('noisy','--')]:
                values=np.asarray(curves[f'{phase}_{m}_{b}_{n}_seed17'])
                window=8 if phase=='pilot' else 100
                y=np.convolve(values[:,1],np.ones(window)/window,mode='valid')
                label=(('卷积 VQ-VAE' if b=='conv' else 'Prithvi 连续 AE')+'／'+('干净训练' if n=='clean' else '加噪训练')) if zh else ('CNN VQ-VAE' if b=='conv' else 'Prithvi AE')+' / '+n
                ax.plot(values[window-1:,0],y,color=color,linestyle=style,label=label)
        ax.set_title(m.upper());ax.set_xlabel('优化器更新步数' if zh else 'Optimizer step');ax.set_ylabel('归一化重建均方误差（MSE）' if zh else 'Normalized reconstruction MSE')
        if phase=='full':ax.set_yscale('log')
        else:ax.set_ylim(bottom=0)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(('全量实验：仅绘制重建损失 · 最近 100 步滑动平均 · 纵轴为对数刻度' if phase=='full' else '小范围测试：仅绘制重建损失 · 最近 8 步滑动平均') if zh else f'{phase}: reconstruction objective only · trailing mean, '+('8' if phase=='pilot' else '100')+' steps'+(' · log y-axis' if phase=='full' else ''))
    fig.savefig(report/f'reconstruction_loss{suffix}.png',dpi=150);plt.close(fig)
    print(str(report/('RESULTS_ZH.md' if zh else 'RESULTS.md')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['pilot','full'],required=True)
    p.add_argument('--language',choices=['en','zh'],default='en')
    args=p.parse_args();summarize(args.output.resolve(),args.phase,args.language)
