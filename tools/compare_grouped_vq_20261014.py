"""Compare frozen ordinary and grouped VQ runs; no training or inference."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/grouped_vs_vq_20261014'
OUT.mkdir(parents=True,exist_ok=True)
RESULTS=ROOT/'outputs/torch_experiments_20261013'
all_models=json.loads((ROOT/'outputs/loss_diagnostics_20261014/diagnostics.json').read_text())['models']
models={m['model']:m for m in all_models}
reps=['R','G','G_power','G_asinh','C','C_affine','C_power','C_asinh']
rows=[]; usage=[]; curves=[]
for rep in reps:
    a,b=models['vq_'+rep],models['grouped_g8k256_'+rep]
    am=json.loads((RESULTS/a['validation']['pixel10']['source']).read_text())
    bm=json.loads((RESULTS/b['validation']['pixel10']['source']).read_text())
    for key in ('physical_input_sha256','mean_sha256','encoded_input_sha256'):
        assert am[key]==bm[key],(rep,key)
    assert am['data_fingerprint']==bm['data_fingerprint']
    assert am['unique_records']==bm['unique_records']==11684
    assert am['views']==bm['views']==2
    raw_a,raw_b=am['raw']['pixel_weighted_rmse'],bm['raw']['pixel_weighted_rmse']
    row=dict(representation=rep,vq_pixel10_rmse_C=raw_a,grouped_pixel10_rmse_C=raw_b,
        grouped_rmse_reduction_pct=100*(1-raw_b/raw_a),
        vq_clean_rmse_C=a['validation']['clean']['raw']['pixel_weighted_rmse'],
        grouped_clean_rmse_C=b['validation']['clean']['raw']['pixel_weighted_rmse'],
        vq_epoch5_pixel=a['epochs'][-1]['pixel']['mean'],grouped_epoch5_pixel=b['epochs'][-1]['pixel']['mean'],
        vq_epoch5_bottleneck=a['epochs'][-1]['bottleneck']['mean'],grouped_epoch5_bottleneck=b['epochs'][-1]['bottleneck']['mean'],
        vq_used_codes=am['raw']['token_statistics']['used_codes'],
        vq_perplexity=am['raw']['token_statistics']['pooled_perplexity'])
    groups=bm['grouped_token_statistics']['per_group']
    row.update(grouped_used_min=min(g['used_codes'] for g in groups),
        grouped_used_max=max(g['used_codes'] for g in groups),
        grouped_perplexity_min=min(g['perplexity'] for g in groups),
        grouped_perplexity_max=max(g['perplexity'] for g in groups),
        grouped_top1_min=min(g['top1_fraction'] for g in groups),
        grouped_top1_max=max(g['top1_fraction'] for g in groups))
    rows.append(row)
    for g in groups:
        # Recompute reported concentration and entropy from exact counts.
        counts=np.array(g['index_counts']); p=counts/counts.sum(); positive=p>0
        assert counts.sum()==g['token_count']
        assert np.isclose(p.max(),g['top1_fraction'])
        assert np.isclose(2**(-(p[positive]*np.log2(p[positive])).sum()),g['perplexity'])
        usage.append(dict(representation=rep,group=g['group'],used_codes=g['used_codes'],
            perplexity=g['perplexity'],top1_fraction=g['top1_fraction'],top10_fraction=g['top10_fraction'],
            scales=[dict(scale=s['scale'],used_codes=s['used_codes'],perplexity=s['perplexity'],
                top1_fraction=max(s['index_counts'])/s['token_count']) for s in g['scales']]))
    curves.append(dict(representation=rep,vq=[dict(epoch=i+1,pixel=e['pixel']['mean'],bottleneck=e['bottleneck']['mean']) for i,e in enumerate(a['epochs'])],
        grouped=[dict(epoch=i+1,pixel=e['pixel']['mean'],bottleneck=e['bottleneck']['mean']) for i,e in enumerate(b['epochs'])]))

result=dict(scope='Frozen epoch-5 models; full HLP v2 val 11684 records, 2 fixed pixel10 views. No test selection.',
    pairing='Physical input, mean, encoded input and full data fingerprint identical within all eight representation pairs.',
    rows=rows,group_usage=usage,epoch_curves=curves,
    mechanism=dict(vq=dict(groups=1,dim_per_group=32,codes_per_group=1024,scales=[16],index_bits=2560,
                          parameters=87190776,initialization='uniform +/-1/1024',unit_rms=False,warmup_updates=0),
                   grouped=dict(groups=8,dim_per_group=4,codes_per_group=256,scales=[1,2,3,4,5,6,8,10,13,16],
                          index_bits=43520,parameters=108828056,initialization='nn.Embedding default normal(0,1)',
                          unit_rms=False,warmup_updates=500)),
    caveats=['Not equal-rate or equal-architecture comparisons; no isolated causal claim about grouping.',
             'Bottleneck objectives differ: single reconstruction vs mean over ten cumulative reconstructions.',
             'Group usage pools scales within each independent group; inspect per-scale data too.',
             'Token statistics include reference-invalid positions; observed usage is not measured compressed bitrate.',
             'Grouped statistics count more token positions than the single-scale baseline.',
             'All scores are final-epoch, not validation-selected best.'])
(OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
for term,title in [('pixel','Physical reconstruction loss'),('bottleneck','Native bottleneck loss (different objectives)')]:
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey=True)
    for ax,rep in zip(axes.flat,reps):
        for prefix,label,color in [('vq_','Ordinary VQ','#1769aa'),('grouped_g8k256_','Grouped + multiscale','#c04b2b')]:
            blocks=models[prefix+rep]['blocks100']
            ax.plot([b['epoch'] for b in blocks],[b[term] for b in blocks],label=label,color=color,lw=1.2)
        ax.set_title(rep);ax.set_yscale('log');ax.set_xlim(0,5)
        ax.set_xlabel('Epoch');ax.set_ylabel('Loss: weighted 100-update mean');ax.grid(alpha=.2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.96),ncol=2,frameon=False)
    fig.suptitle(title+' | Matched data, unequal bitrate / backbone',y=1.)
    fig.tight_layout(rect=(0,0,1,.90));fig.savefig(OUT/(term+'_curves.png'),dpi=150);plt.close(fig)

gs=[u for u in usage if u['representation']=='G']
fig,ax=plt.subplots(figsize=(11,5))
array=np.array([[s['perplexity'] for s in g['scales']] for g in gs])
im=ax.imshow(array,cmap='viridis',aspect='auto',vmin=1,vmax=max(2,array.max()))
for i in range(8):
    for j in range(10):
        ax.text(j,i,f'{array[i,j]:.1f}',ha='center',va='center',fontsize=9,color='white' if array[i,j] < array.max()*.55 else 'black')
ax.set_xticks(range(10),[str(s) for s in result['mechanism']['grouped']['scales']]);ax.set_yticks(range(8))
ax.set_xlabel('Spatial scale (side length)');ax.set_ylabel('Independent codebook group')
ax.set_title('Grouped G: per-scale perplexity (256 available codes per group)')
fig.colorbar(im,ax=ax,label='Effective number of codes (perplexity)')
fig.tight_layout();fig.savefig(OUT/'G_per_scale_usage.png',dpi=150);plt.close(fig)

lines=['Grouped 与普通 VQ：冻结历史模型对照，未运行新训练。',
       '同一 HLP v2 完整验证集、同一 pixel10 视图及输入/均温/编码哈希；所有分数是第5轮，不是验证最优。',
       'Grouped 名义索引位数为普通版的17倍，骨干参数也不同，不能把提升只归因于分组。',
       '两者均未采用单位RMS归一化或训练数据初始化。普通码本均匀初始化±1/1024，grouped为Embedding默认正态初始化。',
       'grouped逐尺度累计量化loss取平均；与普通单尺度loss数值不直接等价。','',
       '表示\t普通VQ验证RMSE_C\tGrouped验证RMSE_C\tRMSE下降%\t普通VQ末轮pixel\tGrouped末轮pixel\t普通VQ末轮bottleneck\tGrouped末轮bottleneck']
for r in rows:
    lines.append('\t'.join([r['representation']]+[f'{r[k]:.6f}' for k in ('vq_pixel10_rmse_C','grouped_pixel10_rmse_C','grouped_rmse_reduction_pct','vq_epoch5_pixel','grouped_epoch5_pixel','vq_epoch5_bottleneck','grouped_epoch5_bottleneck')]))
lines+=['','Grouped G 各组统计（跨尺度，不能把不同组ID混为一个码本）：','组\t使用码数/256\tperplexity\t最常用码占比']
for g in gs:lines.append(f"{g['group']}\t{g['used_codes']}\t{g['perplexity']:.4f}\t{100*g['top1_fraction']:.2f}%")
lines+=['','结论：8种表示中7种显著降低最终RMSE，C_power没有改善。',
        'G等表示中，grouped重建loss持续下降，但bottleneck缓慢上升，说明量化项最低不能代替重建/验证选模。',
        '较多码曾经被使用，不等于分布均匀；低尺度近常数编码也可能与均温已提供及残差尺度有关，不能直接认定为失败。',
        '建议优先将骨干、初始化、warmup、量化尺度与分组因素逐一对齐；单位RMS暂作为可选候选。',
        '若单独比较分组收益，应固定单尺度和名义码率，例如1×1024、2×32、5×4均为每空间位置10bit；这要求各组维数可整除，最后一项不能直接套在D=32上。',
        '保持D=32时可用1×1024 vs 2×32作单尺度等名义码率对照；8×256属于更高容量方案。']
(OUT/'COMPARISON.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'rows':rows,'G_group_usage':gs},ensure_ascii=False,indent=2))
