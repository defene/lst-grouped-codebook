"""Audit full census coverage and produce transparent task difficulty summaries."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from eo_data.core import json_write
OUT=ROOT/'data_inspection/lst_sr_census_20260913'
METRICS=['mean_c','std_c','iqr_c','entropy_bits','gradient_c','gradient4_c','gradient16_c',
         'valid_fraction','synthetic_coverage','synthetic_mae_c','synthetic_rmse_c',
         'synthetic_nearest_rmse_c','synthetic_nrmse','common_coverage','native_mae_c',
         'native_rmse_c','native_bias_c','native_bias_removed_rmse_c','synthetic_common_rmse_c',
         'coarse_bias_c','coarse_rmse_c','native_lr_valid_fraction','native_lr_odd_fraction']

def distribution(values):
    a=np.asarray(values,dtype=float); a=a[np.isfinite(a)]
    if not len(a):return {'n':0}
    p=np.quantile(a,[.01,.1,.25,.5,.75,.9,.99])
    return dict(n=len(a),mean=float(a.mean()),std=float(a.std()),min=float(a.min()),max=float(a.max()),
                **dict(zip(['p01','p10','p25','p50','p75','p90','p99'],map(float,p))))

def pooled_rmse(df,metric,count):
    good=df[metric].notna() & (df[count]>0)
    d=df.loc[good]
    return float(np.sqrt((d[metric]**2*d[count]).sum()/d[count].sum())) if len(d) else None

def audit(df,benchmark):
    counts={}; expected=[]
    assert not df.duplicated(['dataset','record_id','slot']).any()
    assert np.isfinite(df['valid_pixels']).all() and (df['valid_pixels']>0).all()
    assert (df.synthetic_pixels<=df.valid_pixels).all()
    native=df.dataset=='lstsr_tb'
    assert (df.loc[native,'common_pixels']<=df.loc[native,'synthetic_pixels']).all()
    if not benchmark:
        for dataset in ('prithvi','lstsr_tb'):
            prepared=ROOT/'processed_data/v1'/dataset
            bundle=json.loads((prepared/'bundle.json').read_text())
            for key,entry in bundle['samples'].items():
                split,mod=key.split('/')
                if mod!='lst_hr':continue
                actual=df[(df.dataset==dataset)&(df.source_split==split)].sort_values(['record_id','slot'])
                ix=np.load(prepared/entry['index'])
                ix=ix[np.lexsort((ix[:,1],ix[:,0]))]
                slot=ix[:,1]+1 if dataset=='lstsr_tb' else np.zeros(len(ix),int)
                assert len(actual)==len(ix),(dataset,split,len(actual),len(ix))
                assert np.array_equal(actual.record_id.to_numpy(),ix[:,0])
                assert np.array_equal(actual.slot.to_numpy(),slot)
                assert np.array_equal(actual.valid_pixels.to_numpy(),ix[:,2])
                expected.append({'dataset':dataset,'source_split':split,'images':len(actual),'sample_keys_and_valid_counts_match':True})
    return {'scope':'benchmark_only' if benchmark else 'full','duplicate_sample_keys':0,
            'frozen_indices':expected,'all_valid_count_and_support_checks_passed':True}

def aggregate(df):
    results={}; tab=[]
    for dataset in ('prithvi','lstsr_tb'):
        source=df[df.dataset==dataset]
        results[dataset]={}
        for split in ('all','train','val','test'):
            d=source if split=='all' else source[source.split==split]
            if not len(d):continue
            measures={m:distribution(d[m]) for m in METRICS}
            seq=d.groupby('record_id',sort=False).synthetic_rmse_c.mean()
            tile=d.groupby('tile',sort=False,observed=True).synthetic_rmse_c.mean()
            entry={'images':len(d),'sequences':int(d.record_id.nunique()),'tiles':int(d.tile.nunique()),
                   'synthetic_scored_images':int(d.synthetic_rmse_c.notna().sum()),
                   'synthetic_coverage_ge90_images':int((d.synthetic_coverage>=.9).sum()),
                   'synthetic_pixels_pooled_rmse_c':pooled_rmse(d,'synthetic_rmse_c','synthetic_pixels'),
                   'synthetic_sequence_equal_mean_rmse_c':float(seq.mean()),
                   'synthetic_tile_equal_mean_rmse_c':float(tile.mean()),
                   'synthetic_ge90':distribution(d.loc[d.synthetic_coverage>=.9,'synthetic_rmse_c']),
                   'synthetic_fraction_rmse_above_c':{str(t):float((d.loc[d.synthetic_rmse_c.notna(),'synthetic_rmse_c']>t).mean()) for t in (1,2,5)},
                   'metrics':measures}
            if dataset=='lstsr_tb':
                good=d[d.common_pixels>0]
                entry['native_pixels_pooled_rmse_c']=pooled_rmse(good,'native_rmse_c','common_pixels')
                entry['synthetic_common_pixels_pooled_rmse_c']=pooled_rmse(good,'synthetic_common_rmse_c','common_pixels')
                entry['native_sequence_equal_mean_rmse_c']=float(good.groupby('record_id').native_rmse_c.mean().mean())
                entry['native_tile_equal_mean_rmse_c']=float(good.groupby('tile',observed=True).native_rmse_c.mean().mean())
                entry['native_common_coverage_ge90_images']=int((good.common_coverage>=.9).sum())
                entry['native_ge90']=distribution(good.loc[good.common_coverage>=.9,'native_rmse_c'])
                entry['synthetic_common_ge90']=distribution(good.loc[good.common_coverage>=.9,'synthetic_common_rmse_c'])
                entry['native_minus_synthetic_rmse_c']=distribution(good.native_rmse_c-good.synthetic_common_rmse_c)
                entry['fraction_native_rmse_above_synthetic_same_pixels']=float((good.native_rmse_c>good.synthetic_common_rmse_c).mean())
                entry['native_fraction_rmse_above_c']={str(t):float((good.native_rmse_c>t).mean()) for t in (1,2,5)}
            results[dataset][split]=entry
            for metric,values in measures.items():tab.append(dict(dataset=dataset,split=split,metric=metric,**values))
    return results,pd.DataFrame(tab)

def matched_strata(df):
    # Equal weight to shared tile/calendar-month cells in the overlapping period.
    records=[]; outputs={}
    for split in ('all','test'):
        d=df[(df.year>=2019)&(df.year<=2025)&(df.synthetic_coverage>=.9)]
        if split!='all':d=d[d.split==split]
        grouped=d.groupby(['dataset','tile','month'],observed=True).agg(rmse=('synthetic_rmse_c','mean'),
                     images=('synthetic_rmse_c','count'),iqr=('iqr_c','mean'),gradient=('gradient_c','mean'))
        if not {'prithvi','lstsr_tb'}.issubset(set(grouped.index.get_level_values(0))):continue
        paired=grouped.loc['prithvi'].join(grouped.loc['lstsr_tb'],how='inner',lsuffix='_prithvi',rsuffix='_lstsr').dropna()
        if not len(paired):continue
        delta=paired.rmse_lstsr-paired.rmse_prithvi
        outputs[split]={'shared_tile_month_cells':len(paired),'shared_tiles':int(paired.index.get_level_values('tile').nunique()),
                       'prithvi_images':int(paired.images_prithvi.sum()),'lstsr_images':int(paired.images_lstsr.sum()),
                       'prithvi_cell_equal_mean_rmse_c':float(paired.rmse_prithvi.mean()),
                       'lstsr_cell_equal_mean_rmse_c':float(paired.rmse_lstsr.mean()),
                       'fraction_cells_lstsr_higher':float((delta>0).mean()),'cell_difference_c':distribution(delta)}
        flat=paired.reset_index();flat.insert(0,'split',split);records.append(flat)
    return outputs,pd.concat(records,ignore_index=True) if records else pd.DataFrame()

def plot(results,destination):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    ax=axes[0]
    for i,split in enumerate(('all','test')):
        for j,(ds,label,color) in enumerate((('prithvi','Prithvi','#356999'),('lstsr_tb','LSTSR-TB','#b87526'))):
            m=results[ds][split]['metrics']['synthetic_rmse_c'];x=i+(j-.5)*.25
            ax.errorbar(x,m['p50'],yerr=[[m['p50']-m['p10']],[m['p90']-m['p50']]],fmt='o',capsize=5,color=color,label=label if i==0 else None)
    ax.set_xticks([0,1],['All images','Test split']);ax.set_ylabel('Per-image RMSE (degrees C)')
    ax.set_title('Same synthetic 32x task\nMedian and P10-P90');ax.legend(frameon=False)
    ax=axes[1]
    for i,metric in enumerate(('synthetic_common_rmse_c','native_rmse_c','native_bias_removed_rmse_c')):
        m=results['lstsr_tb']['test']['metrics'][metric]
        ax.errorbar(i,m['p50'],yerr=[[m['p50']-m['p10']],[m['p90']-m['p50']]],fmt='o',capsize=5,color=['#356999','#b87526','#777777'][i])
    ax.set_xticks(range(3),['Synthetic LR','Real MODIS','Real MODIS\noffset removed*'])
    ax.set_title('LSTSR-TB test: same target pixels\nMedian and P10-P90');ax.set_ylabel('Per-image RMSE (degrees C)')
    fig.supxlabel('*Offset uses target truth for diagnosis; it is not a deployable baseline.',fontsize=9)
    for ax in axes:ax.set_ylim(bottom=0);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
    fig.savefig(destination/'difficulty_comparison.png',dpi=170);plt.close(fig)

def report(results,matched,destination,benchmark):
    lines=['# LST 全量统计与超分辨率难度比较','',
      '**基准检查产物，不是全量结果。**' if benchmark else '**范围：全部冻结 LST 样本、全部划分；两套数据相同的 32 倍单图温度超分设置。**','',
      'Prithvi 的源 val 按用户确认报告为 test。Prithvi 无现成 MODIS LR，因此不能直接比较两套真实跨传感器任务。',
      '统一任务：256×256 HR 在 32×32 像元块内取有效像元平均，得到 8×8 LR，再双线性插值回 256×256。统计使用摄氏度；不训练模型，不使用 HLS 或其他月份。',
      '真实任务：LSTSR-TB 用实际 8×8 MODIS 温度插值，和合成 LR 基线在完全相同的有效 HR 像元上对照。','',
      '## 相同合成退化下的结果','',
      '| 数据 | 划分 | 全部图像数 | 可评分图像数 | 每图 RMSE 均值 °C | RMSE 中位数 [P10,P90] °C | 像元合并 RMSE °C |',
      '|---|---|---:|---:|---:|---|---:|']
    for dataset in ('prithvi','lstsr_tb'):
        for split,r in results[dataset].items():
            m=r['metrics']['synthetic_rmse_c']
            lines.append(f"| {dataset} | {split} | {r['images']:,} | {m['n']:,} | {m['mean']:.4f} | {m['p50']:.4f} [{m['p10']:.4f}, {m['p90']:.4f}] | {r['synthetic_pixels_pooled_rmse_c']:.4f} |")
    lines+=['','## 真实 MODIS 输入增加的误差','',
      '| LSTSR 划分 | 同像元合成 LR RMSE 均值 °C | 真实 MODIS RMSE 均值 °C | 去除单图均值偏差后 RMSE 均值 °C | 真实误差较高的图像占比 |',
      '|---|---:|---:|---:|---:|']
    for split,r in results['lstsr_tb'].items():
        m=r['metrics']
        lines.append(f"| {split} | {m['synthetic_common_rmse_c']['mean']:.4f} | {m['native_rmse_c']['mean']:.4f} | {m['native_bias_removed_rmse_c']['mean']:.4f} | {r['fraction_native_rmse_above_synthetic_same_pixels']:.2%} |")
    lines+=['','逐图去偏差利用 HR 真值，仅用于区分整体温度偏移与空间误差，不能作为可部署模型成绩。误差不应全部归因于配准：还可能包含采集时间、响应函数、原始温度反演差异和质量差异。','',
      '## 公平性与覆盖','',
      '- 全部图像均有复杂度记录。SR 只在输入与目标有效支持范围上计算，零支持时保留缺失而非报 0。另报告覆盖率 ≥90% 的敏感性结果。',
      '- 图像等权会使可用月份更多的序列占更大权重；JSON 同时报告序列等权、瓦片等权及像元合并 RMSE。',
      '- 在双方共有的 2019—2025 年范围内，对重叠的瓦片×日历月份分层等权比较；这是地理与季节控制，不是同日同地逐像元配对。',
      '- Prithvi 日期使用已有 metadata.datetime。此前检查发现部分日期字段与光学场景时间不同，因此日期分层依赖其元数据语义。','']
    for split,m in matched.items():
        lines.append(f"- {split} 匹配：{m['shared_tiles']} 个瓦片、{m['shared_tile_month_cells']} 个瓦片月份单元；单元等权 RMSE Prithvi={m['prithvi_cell_equal_mean_rmse_c']:.4f}°C，LSTSR={m['lstsr_cell_equal_mean_rmse_c']:.4f}°C；LSTSR 较高单元占 {m['fraction_cells_lstsr_higher']:.2%}。")
    lines+=['','## 如何解释“更难”','',
      '统一退化下的插值误差衡量一个明确、可复现基线未恢复的空间细节，不是学习算法的不可约误差。真实 MODIS 的额外误差反映跨传感器任务负担。',
      '这份统计不能证明所有超分模型都会得到相同难度排序。若目标是固定模型达到某精度所需的数据量或训练时间，需要在相同模型、训练预算、损失和输入模态下另做实验。','',
      '## 复现与核查','',
      '- `coverage_audit.json`：逐划分和冻结样本索引核对记录 ID、月份槽位及有效像元计数；重复样本键检查。',
      '- `distributions.csv`：图内均值、标准差、IQR、熵、梯度及 SR 基线的均值和精确分位数。',
      '- `summary.json`：按图像、序列、瓦片及像元权重的汇总、支持范围、真实/合成对照。',
      '- `matched_tile_months.csv`：地理与季节匹配的统计单元。',
      '- `chunks/*.parquet`：每张温度图的统计、来源 ID 和日期。',
      '- `protocol.json`：固定指标定义、输入和代码指纹；`analysis.ipynb` 为读取与审计入口。',
      '- 熵按 1°C 固定分箱；梯度按有效相邻边平均；4/16 倍粗网格仅采用完全有效的块。','']
    (destination/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')

def notebook(destination):
    import nbformat as nbf
    nb=nbf.v4.new_notebook()
    nb.cells=[nbf.v4.new_markdown_cell('# LST 全量统计与超分任务比较\n\n读取本次冻结的逐图统计、核对完整性并复查汇总。插值误差是基线难度指标，不是学习模型的不可约误差。'),
              nbf.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nroot = Path.cwd()\nif not (root / 'summary.json').exists():\n    root = root / 'data_inspection/lst_sr_census_20260913'\nsummary = json.loads((root/'summary.json').read_text(encoding='utf-8'))\naudit = json.loads((root/'coverage_audit.json').read_text(encoding='utf-8'))\naudit"),
              nbf.v4.new_code_cell("stats = pd.read_csv(root/'distributions.csv')\nstats[(stats.metric=='synthetic_rmse_c') & stats.split.isin(['all','test'])]"),
              nbf.v4.new_code_cell("summary['matched_tile_months']"),
              nbf.v4.new_code_cell("from IPython.display import Image, display\ndisplay(Image(filename=str(root/'difficulty_comparison.png')))"),
              nbf.v4.new_markdown_cell('全量重跑：在项目根目录运行 `python experiments/data/lst_sr_census.py --workers 4`，完成后运行 `python experiments/data/summarize_lst_sr_census.py`。不会修改输入数据。')]
    nbf.write(nb,destination/'analysis.ipynb')

def sources_receipt(results,checked,destination):
    rows=[];native=[]
    for ds in ('prithvi','lstsr_tb'):
        for split in ('all','test'):
            r=results[ds][split];m=r['metrics']['synthetic_rmse_c']
            rows.append({'dataset':ds,'split':split,'images':r['images'],'scored_images':m['n'],
                         'mean_rmse_c':m['mean'],'median_rmse_c':m['p50'],'p90_rmse_c':m['p90']})
    for split in ('all','test'):
        r=results['lstsr_tb'][split]['metrics']
        native.append({'split':split,'scored_images':r['native_rmse_c']['n'],
                       'synthetic_common_mean_rmse_c':r['synthetic_common_rmse_c']['mean'],
                       'native_mean_rmse_c':r['native_rmse_c']['mean'],
                       'offset_removed_mean_rmse_c':r['native_bias_removed_rmse_c']['mean']})
    source={'label':'本地冻结 LST 数据全集','files':[{'label':'Prithvi 冻结索引与无损数据包'},{'label':'LSTSR-TB 冻结索引与原始 NPZ'}],
            'metricDefinitions':[{'label':'每图 RMSE','definition':'对有效目标像元计算温度误差平方的均值后开方，单位为摄氏度；表中均值对图像等权。'}],
            'filters':['所有冻结 LST 样本；分别汇总全部划分与测试集','Prithvi 源 val 按用户确认报告为测试集','统一 32 倍面积平均退化和双线性插值'],
            'caveats':['插值误差是固定基线的难度指标，不是任意学习模型的不可约误差。','没有评分支持的图像保留记录但不计入误差均值。'],
            'evidenceFlow':[{'kind':'calculation','title':'统一比较方法','detail':'256×256 HR 在32×32块内作有效像元平均得到8×8 LR，再使用半像元中心双线性插值重建。仅评分完整有效插值支持上的有效HR像元。'},
                            {'kind':'validation','title':'完整性核对','detail':'逐划分与冻结索引核对记录ID、月份槽位及有效像元计数全部通过；重复样本键为0。'}]}
    second={'label':'LSTSR-TB 原始 HR/LR 温度对','files':[{'label':'LSTSR-TB 原始 NPZ'},{'label':'全量逐图 SR 指标'}],
            'metricDefinitions':source['metricDefinitions'],
            'filters':['真实MODIS与合成LR在完全相同的有效HR像元上比较'],
            'caveats':['原始MODIS LR沿用已有摄氏度×100存储解释。','去除单图平均偏差使用HR真值，只用于诊断，不是可部署基线。'],
            'evidenceFlow':[{'kind':'calculation','title':'区分输入差异','detail':'对真实MODIS输入和Landsat面积平均得到的合成输入分别作相同插值。逐图去偏差RMSE等于 sqrt(MSE−bias²)。'}]}
    payload={'schemaVersion':1,'items':[
       {'id':'controlled-sr','title':'统一32倍LST超分基线的全量和测试结果','queries':[{'id':'controlled-census','source':source,'columns':list(rows[0]),'rows':rows,
        'methods':[{'language':'calculation','code':'每图 RMSE = sqrt(有效像元误差平方和 / 有效像元数)\n图像等权均值 = 有评分支持的每图 RMSE 之和 / 可评分图像数'}]}]},
       {'id':'native-cross-sensor','title':'真实MODIS输入引入的额外重建误差','queries':[{'id':'native-census','source':second,'columns':list(native[0]),'rows':native}]}]}
    json_write(destination/'sources_receipt.json',payload)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',action='store_true');args=ap.parse_args()
    destination=OUT/'benchmark_analysis' if args.benchmark else OUT;destination.mkdir(exist_ok=True)
    chunks=OUT/('benchmark_chunks' if args.benchmark else 'chunks')
    if not args.benchmark:
        status=json.loads((OUT/'status.json').read_text())
        if status['status']!='scan_complete':raise RuntimeError('Full scan has not completed')
    print('Loading chunks',flush=True)
    df=pads.dataset(chunks,format='parquet').to_table().to_pandas()
    for name in ('dataset','split','source_split','tile'):df[name]=df[name].astype('category')
    df['valid_fraction']=df.valid_pixels/65536
    df['synthetic_coverage']=df.synthetic_pixels/65536
    df['common_coverage']=df.common_pixels/65536
    checked=audit(df,args.benchmark);json_write(destination/'coverage_audit.json',checked)
    print('Coverage audit passed',flush=True)
    results,table=aggregate(df);matched,mt=matched_strata(df)
    summary={'scope':'benchmark' if args.benchmark else 'full','results':results,'matched_tile_months':matched}
    json_write(destination/'summary.json',summary);table.to_csv(destination/'distributions.csv',index=False)
    mt.to_csv(destination/'matched_tile_months.csv',index=False)
    if not args.benchmark:
        plot(results,destination)
        sources_receipt(results,checked,destination)
    report(results,matched,destination,args.benchmark);notebook(destination)
    print(json.dumps({'scope':summary['scope'],'summary':str(destination/'summary.json'),'images':len(df)}),flush=True)

if __name__=='__main__':main()
