"""Full task comparison after both census passes; validates baseline parity."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import summarize_lst_sr_census as plain
from eo_data.core import json_write
ROOT=plain.ROOT;OUT=ROOT/'data_inspection/lst_hls_sr_census_20260913'
METRICS=['hls_date_gap_days','hls_valid_fraction','hls_coarse_valid_fraction','synthetic_coverage','synthetic_baseline_rmse_c',
 'synthetic_guided_rmse_c','synthetic_guided_mae_c','synthetic_gain_c','synthetic_guided_fraction',
 'synthetic_guided_nrmse',
 'native_baseline_rmse_c','native_guided_rmse_c','native_guided_mae_c','native_gain_c','native_guided_fraction',
 'native_guided_train_calibrated_rmse_c','native_guided_oracle_debiased_rmse_c',
 'synthetic_common_guided_rmse_c','common_coverage','max_abs_hls_lst_correlation']

def aggregate(df):
    results={};rows=[]
    for ds in ('prithvi','lstsr_tb'):
        source=df[df.dataset==ds];results[ds]={}
        for split in ('all','train','val','test'):
            d=source if split=='all' else source[source.split==split]
            if not len(d):continue
            metrics={name:plain.distribution(d[name]) for name in METRICS}
            r={'images':len(d),'sequences':int(d.record_id.nunique()),'tiles':int(d.tile.nunique()),'metrics':metrics,
               'synthetic_guided_pixel_pooled_rmse_c':plain.pooled_rmse(d,'synthetic_guided_rmse_c','synthetic_pixels'),
               'synthetic_sequence_equal_mean_rmse_c':float(d.groupby('record_id').synthetic_guided_rmse_c.mean().mean()),
               'synthetic_tile_equal_mean_rmse_c':float(d.groupby('tile',observed=True).synthetic_guided_rmse_c.mean().mean()),
               'synthetic_fraction_improved':float((d.loc[d.synthetic_guided_rmse_c.notna(),'synthetic_gain_c']>0).mean()),
               'synthetic_fit_available_fraction':float((d.synthetic_fit_cells>=16).mean()),
               'synthetic_high_support':plain.distribution(d.loc[d.synthetic_coverage>=.9,'synthetic_guided_rmse_c']),
               'synthetic_near_date_le30_high_support':plain.distribution(d.loc[(d.synthetic_coverage>=.9)&(d.hls_date_gap_days<=30),'synthetic_guided_rmse_c']),
               'date_gap_fractions':{str(t):float((d.hls_date_gap_days<=t).mean()) for t in (0,1,7,30,365)},
               'date_gap_over_365_fraction':float((d.hls_date_gap_days>365).mean())}
            if ds=='lstsr_tb':
                r.update(native_guided_pixel_pooled_rmse_c=plain.pooled_rmse(d,'native_guided_rmse_c','common_pixels'),
                  native_sequence_equal_mean_rmse_c=float(d.groupby('record_id').native_guided_rmse_c.mean().mean()),
                  native_tile_equal_mean_rmse_c=float(d.groupby('tile',observed=True).native_guided_rmse_c.mean().mean()),
                  native_fraction_improved=float((d.loc[d.native_guided_rmse_c.notna(),'native_gain_c']>0).mean()),
                  native_fit_available_fraction=float((d.native_fit_cells>=16).mean()),
                  native_high_support=plain.distribution(d.loc[d.common_coverage>=.9,'native_guided_rmse_c']),
                  synthetic_common_high_support=plain.distribution(d.loc[d.common_coverage>=.9,'synthetic_common_guided_rmse_c']))
            results[ds][split]=r
            for name,m in metrics.items():rows.append(dict(dataset=ds,split=split,metric=name,**m))
    return results,pd.DataFrame(rows)

def matched(df):
    outputs={};frames=[]
    for split in ('all','test','all_year_month','test_year_month'):
        d=df[(df.year>=2019)&(df.year<=2025)&(df.synthetic_coverage>=.9)]
        if split.startswith('test'):d=d[d.split=='test']
        grouping=['dataset','tile']+(['year'] if split.endswith('year_month') else [])+['month']
        g=d.groupby(grouping,observed=True).agg(rmse=('synthetic_guided_rmse_c','mean'),
              baseline=('synthetic_baseline_rmse_c','mean'),images=('synthetic_guided_rmse_c','count'),gap=('hls_date_gap_days','mean'))
        if not {'prithvi','lstsr_tb'}.issubset(set(g.index.get_level_values(0))):continue
        p=g.loc['prithvi'].join(g.loc['lstsr_tb'],how='inner',lsuffix='_prithvi',rsuffix='_lstsr').dropna()
        if not len(p):continue
        delta=p.rmse_lstsr-p.rmse_prithvi
        cluster=delta.groupby(level='tile',observed=True).agg(['sum','count'])
        rng=np.random.default_rng(20260913)
        draws=rng.integers(0,len(cluster),size=(2000,len(cluster)))
        bootstrap=cluster['sum'].to_numpy()[draws].sum(axis=1)/cluster['count'].to_numpy()[draws].sum(axis=1)
        outputs[split]={'grouping':grouping[1:],'cells':len(p),'tiles':int(p.index.get_level_values('tile').nunique()),
          'prithvi_images':int(p.images_prithvi.sum()),'lstsr_images':int(p.images_lstsr.sum()),
          'prithvi_guided_mean_rmse_c':float(p.rmse_prithvi.mean()),'lstsr_guided_mean_rmse_c':float(p.rmse_lstsr.mean()),
          'prithvi_baseline_mean_rmse_c':float(p.baseline_prithvi.mean()),'lstsr_baseline_mean_rmse_c':float(p.baseline_lstsr.mean()),
          'fraction_cells_lstsr_higher':float((p.rmse_lstsr>p.rmse_prithvi).mean()),
          'lstsr_minus_prithvi_mean_rmse_c':float(delta.mean()),
          'tile_cluster_bootstrap_95_percentile_interval_c':np.quantile(bootstrap,[.025,.975]).tolist(),
          'bootstrap_replicates':2000,'bootstrap_seed':20260913}
        p=p.reset_index();p.insert(0,'split',split);frames.append(p)
    return outputs,pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def audit(df,benchmark):
    checked=plain.audit(df,benchmark)
    if not benchmark:
        optical_checks=[]
        for ds in ('prithvi','lstsr_tb'):
            prepared=ROOT/'processed_data/v1'/ds;b=json.loads((prepared/'bundle.json').read_text())
            for key,entry in b['samples'].items():
                split,mod=key.split('/')
                if mod!='hls':continue
                d=df[(df.dataset==ds)&(df.source_split==split)]
                counts=d.groupby('record_id',sort=True).hls_valid_pixels.agg(['first','min','max'])
                assert (counts['min']==counts['max']).all()
                ix=np.load(prepared/entry['index']);ix=ix[np.argsort(ix[:,0])]
                assert np.array_equal(counts.index.to_numpy(),ix[:,0])
                assert np.array_equal(counts['first'].to_numpy(),ix[:,2])
                optical_checks.append({'dataset':ds,'source_split':split,'records':len(counts),'hls_valid_pixels_match':True})
        checked['optical_indices']=optical_checks
        date_reference=json.loads((OUT/'full_date_alignment.json').read_text())
        date_checks=[]
        for ds in ('prithvi','lstsr_tb'):
            d=df[df.dataset==ds]
            actual={str(int(k)):int(v) for k,v in d.hls_date_gap_days.value_counts().items()}
            assert actual==date_reference[ds]['gap_day_histogram']
            date_checks.append({'dataset':ds,'images':len(d),'independent_full_metadata_histogram_match':True})
        checked['date_alignment']=date_checks
    cols=['dataset','record_id','slot','synthetic_pixels','common_pixels','synthetic_rmse_c','native_rmse_c','synthetic_target_std_c']
    basepath=plain.OUT/('benchmark_chunks' if benchmark else 'chunks')
    b=pads.dataset(basepath,format='parquet').to_table(columns=cols).to_pandas()
    compare=df[['dataset','record_id','slot','synthetic_pixels','common_pixels','synthetic_baseline_rmse_c','native_baseline_rmse_c']].merge(
        b,on=['dataset','record_id','slot'],how='left',validate='one_to_one',suffixes=('','_plain'))
    for f in ('synthetic_pixels','common_pixels'):
        assert np.allclose(compare[f],compare[f+'_plain'],equal_nan=True)
    deltas={}
    for new,old in [('synthetic_baseline_rmse_c','synthetic_rmse_c'),('native_baseline_rmse_c','native_rmse_c')]:
        assert np.allclose(compare[new],compare[old],atol=1e-9,rtol=1e-10,equal_nan=True)
        deltas[new]=float((compare[new]-compare[old]).abs().max())
    checked['full_baseline_parity_max_abs_difference']=deltas
    sd=compare.synthetic_target_std_c.to_numpy()
    df['synthetic_guided_nrmse']=np.where(sd>=.1,df.synthetic_guided_rmse_c.to_numpy()/np.maximum(sd,1e-20),np.nan)
    return checked

def temporal_table(df):
    edges=[-1,0,7,30,365,np.inf];labels=['same_day','1-7_days','8-30_days','31-365_days','over_365_days']
    d=df.copy(deep=False);d=d.assign(date_gap_bin=pd.cut(d.hls_date_gap_days,edges,labels=labels))
    return d.groupby(['dataset','split','date_gap_bin'],observed=True).agg(
        images=('record_id','size'),mean_guided_rmse_c=('synthetic_guided_rmse_c','mean'),
        mean_baseline_rmse_c=('synthetic_baseline_rmse_c','mean'),mean_gain_c=('synthetic_gain_c','mean'),
        mean_native_guided_rmse_c=('native_guided_rmse_c','mean'),mean_correlation=('max_abs_hls_lst_correlation','mean')).reset_index()

def plot(results,destination):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,split in zip(axes,('all','test')):
        names=['Prithvi\nsynthetic LR','LSTSR-TB\nsynthetic LR','LSTSR-TB\nreal MODIS']
        configs=[('prithvi','synthetic'),('lstsr_tb','synthetic'),('lstsr_tb','native')]
        for i,(ds,prefix) in enumerate(configs):
            m=results[ds][split]['metrics']
            vals=[m[prefix+'_baseline_rmse_c']['mean'],m[prefix+'_guided_rmse_c']['mean']]
            if prefix=='native':vals.append(m['native_guided_train_calibrated_rmse_c']['mean'])
            for j,value in enumerate(vals):
                x=i+(j-(len(vals)-1)/2)*.27
                label=['LR LST only','LR LST + HLS','HLS + train offset calibration'][j] if i==0 or j==2 else None
                ax.bar(x,value,.24,color=['#8c99a7','#326f99','#bf8b3d'][j],label=label)
                ax.text(x,value+.025,f'{value:.2f}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks(range(3),names);ax.set_ylabel('Mean per-image RMSE (degrees C)');ax.set_title('All images' if split=='all' else 'Test split')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.2)
    axes[0].legend(frameon=False,loc='upper left',fontsize=9)
    fig.suptitle('32x LST super-resolution: effect of six-band HLS guidance',fontsize=14)
    fig.savefig(destination/'guided_difficulty_comparison.png',dpi=170);plt.close(fig)

def report(results,match,destination,benchmark):
    lines=['# 全量：低分辨率 LST＋HLS → 高分辨率 LST','',
      '**仅基准检查数据，不是全量结果。**' if benchmark else '**覆盖两套数据的全部冻结温度样本与对应 HLS。**','',
      '## 比较设置','',
      '相同合成任务：256×256 HR 温度按32×32块有效像元平均得到8×8 LR；与对应六波段HLS一起输入。真实任务另使用LSTSR的实际8×8 MODIS输入。',
      '引导基线：在每张图的低分辨率网格上，用LR温度与块平均HLS拟合六维岭回归（至少16个有效格点，特征标准化，固定λ=1，无参数搜索）；预测为双线性LR温度加上线性HLS细节。拟合不使用目标HR温度。',
      '光学缺失、插值支持不足或拟合格点不足时回退到纯温度双线性插值。两条基线在相同目标像元评分，因此不会因只保留清晰光学区域而虚增改善。','',
      '## 全量和测试结果','',
      '| 数据 | 划分 | 图像数 | 仅LR温度 RMSE均值 °C | LR温度＋HLS RMSE均值 °C | 引导后RMSE中位数 [P10,P90] °C | 改善图像占比 |',
      '|---|---|---:|---:|---:|---|---:|']
    for ds in ('prithvi','lstsr_tb'):
        for split,r in results[ds].items():
            m=r['metrics'];g=m['synthetic_guided_rmse_c'];b=m['synthetic_baseline_rmse_c']
            lines.append(f"| {ds} 合成32× | {split} | {r['images']:,} | {b['mean']:.4f} | {g['mean']:.4f} | {g['p50']:.4f} [{g['p10']:.4f},{g['p90']:.4f}] | {r['synthetic_fraction_improved']:.2%} |")
    for split,r in results['lstsr_tb'].items():
        m=r['metrics'];g=m['native_guided_rmse_c'];b=m['native_baseline_rmse_c']
        lines.append(f"| LSTSR 真实MODIS | {split} | {r['images']:,} | {b['mean']:.4f} | {g['mean']:.4f} | {g['p50']:.4f} [{g['p10']:.4f},{g['p90']:.4f}] | {r['native_fraction_improved']:.2%} |")
    lines+=['','真实与合成LSTSR评分面积可能不同，严格同像元比较见summary.json中的synthetic_common_guided_rmse_c与native_guided_rmse_c。','',
      '## 真实输入的温度偏差诊断','',
      '| 划分 | 原始引导RMSE均值 °C | 仅用训练集拟合全局温差校准后 °C | 利用逐图真值去除偏差后 °C（诊断） |',
      '|---|---:|---:|---:|']
    for split,r in results['lstsr_tb'].items():
        m=r['metrics']
        lines.append(f"| {split} | {m['native_guided_rmse_c']['mean']:.4f} | {m['native_guided_train_calibrated_rmse_c']['mean']:.4f} | {m['native_guided_oracle_debiased_rmse_c']['mean']:.4f} |")
    lines+=['','全局校准只拟合train的平均温差，再用于所有划分；逐图去偏差使用目标真值，不是可部署的基线。两者用来区分可学习的整体偏移和仍然存在的空间误差。','',
      '## HLS与目标温度日期差','',
      '| 数据 | HLS日期差中位数（天） | 同日比例 | 30天以内比例 | 超过1年比例 |','|---|---:|---:|---:|---:|']
    for ds in ('prithvi','lstsr_tb'):
        r=results[ds]['all'];d=r['date_gap_fractions'];m=r['metrics']['hls_date_gap_days']
        lines.append(f"| {ds} | {m['p50']:.0f} | {d['0']:.2%} | {d['30']:.2%} | {r['date_gap_over_365_fraction']:.2%} |")
    lines+=['','日期差为日历日差；Prithvi温度侧沿用metadata.datetime，未独立确认其为LST采集时间。LSTSR每条序列的一份HLS会用于多个跨年份月份；日期差不自动意味着几何错位，也不能单独证明误差原因。','',
      '## 地理、季节与权重敏感性','']
    for split,m in match.items():
        group_label='瓦片×年份×月份' if 'year' in m['grouping'] else '瓦片×月份'
        lines.append(f"- {split}：共有{m['tiles']}个瓦片、{m['cells']}个{group_label}单元，在2019—2025年且评分覆盖≥90%的图像上单元等权比较。引导后RMSE为Prithvi {m['prithvi_guided_mean_rmse_c']:.4f}°C、LSTSR {m['lstsr_guided_mean_rmse_c']:.4f}°C；LSTSR误差较高单元占{m['fraction_cells_lstsr_higher']:.2%}。")
        ci=m['tile_cluster_bootstrap_95_percentile_interval_c']
        lines.append(f"  瓦片整体重采样2000次的差值95%百分位区间为[{ci[0]:.4f}, {ci[1]:.4f}]°C，仅反映现有重叠瓦片的重采样敏感性。匹配瓦片不等于同一裁剪区域或同一天；测试集共有瓦片很少时，不据此推广难度排序。")
    lines+=['','summary.json另含图像等权、序列等权、瓦片等权、像元合并及高覆盖率子集结果。date_gap_strata.csv按日期差分组展示误差与光学改善。','',
      '## 结论的适用范围','',
      '该实验直接评估了指定输入条件下的一条完整、可复现的光学引导基线。它不是训练后的深度网络性能比较；不能把基线差异直接当作所有模型的难度排序或不可约误差。HLS与温度关系可能非线性，使用其他模型、多时相条件、退化核或倍率时需要重新验证。',
      'Prithvi没有对应的真实MODIS输入，因此“两个真实MODIS跨传感器任务谁更难”在现有数据下无法直接回答。可公平比较的是共同的合成32倍任务，再单独量化LSTSR真实MODIS输入的额外负担。','',
      'LSTSR样本数量更大，可能为训练带来优势；本统计没有控制训练样本量、训练计算量和模型结构，因此不能据此宣称它更难训练。','',
      '## 文件','',
      '- coverage_audit.json：全量温度和光学索引逐条核对，以及与第一阶段全部双线性基线结果一致性检查。',
      '- distributions.csv、summary.json：精确统计及分位数。',
      '- matched_tile_months.csv、date_gap_strata.csv：分层与敏感性结果。',
      '- chunks/*.parquet：每个样本的指标、来源ID、月份和日期差。',
      '- protocol.json：输入与代码指纹、固定模型与掩码规则。',
      '- 原温度复杂度全集保存在相邻lst_sr_census_20260913目录。','']
    if not benchmark:
        p=results['prithvi']['all']['metrics'];l=results['lstsr_tb']['all']['metrics']
        pm=p['synthetic_guided_rmse_c']['mean'];lm=l['synthetic_guided_rmse_c']['mean']
        verdict=['## 本次统计支持的判断','',
          f"在本次固定32×、低分辨率LST＋单幅六波段HLS的基线上，LSTSR-TB误差更高：全量图像等权RMSE为{lm:.4f}°C，Prithvi为{pm:.4f}°C，相差{lm/pm-1:.1%}。这支持本设置下LSTSR-TB更具挑战性，但不是对所有训练模型的普遍排序。",'',
          f"HLS引导使Prithvi的RMSE下降{1-pm/p['synthetic_baseline_rmse_c']['mean']:.1%}；LSTSR-TB从{l['synthetic_baseline_rmse_c']['mean']:.4f}变为{lm:.4f}°C，整体没有改善。这里反映的是固定线性引导方法的收益，不代表HLS对非线性模型无用。",'']
        if 'all_year_month' in match:
            matched=match['all_year_month']
            verdict +=[f"在{matched['tiles']}个共同瓦片、{matched['cells']}个瓦片×年份×月份单元中，Prithvi与LSTSR-TB分别为{matched['prithvi_guided_mean_rmse_c']:.4f}、{matched['lstsr_guided_mean_rmse_c']:.4f}°C，后者仍高{matched['lstsr_guided_mean_rmse_c']/matched['prithvi_guided_mean_rmse_c']-1:.1%}。此为全部划分的分层比较，相同单元不保证完全相同的裁剪区域或采集日。",'']
        verdict +=[f"LSTSR真实MODIS＋HLS的全量RMSE为{l['native_guided_rmse_c']['mean']:.4f}°C，仅用训练集拟合一个全局温差后为{l['native_guided_train_calibrated_rmse_c']['mean']:.4f}°C；同像元的合成LR＋HLS为{l['synthetic_common_guided_rmse_c']['mean']:.4f}°C。真实输入的额外误差没有被一个全局偏移消除，但这仍然不是不可约误差估计。",'']
        if 'test_year_month' not in match:
            verdict +=['两套测试集在瓦片×年份×月份上没有共同单元，因此严格时空分层的测试集难度差异仍无法估计。','']
        lines[4:4]=verdict
        complexity=json.loads((plain.OUT/'summary.json').read_text())['results']
        extra=['## 全数据集的目标图像复杂度','',
          '以下均为先逐张计算、再图像等权求均值；目标温度有效性掩码与超分评估一致，梯度只计算有效相邻像元。超分误差还要求低分辨率插值支持有效。熵固定使用1°C宽度的温度直方图，避免逐图归一化带来的不可比性。','',
          '| 数据 | 温度样本数 | 单图标准差 °C | 单图IQR °C | 熵 bit | 相邻像元绝对温差 °C |',
          '|---|---:|---:|---:|---:|---:|']
        for ds in ('prithvi','lstsr_tb'):
            r=complexity[ds]['all'];m=r['metrics']
            extra.append(f"| {ds} | {r['images']:,} | {m['std_c']['mean']:.4f} | {m['iqr_c']['mean']:.4f} | {m['entropy_bits']['mean']:.4f} | {m['gradient_c']['mean']:.4f} |")
        extra+=['','LSTSR-TB共有277,808条序列，对应2,892,007个可用温度月份样本；一份HLS对应序列内多个月份，各月份可能来自不同年份。Prithvi为242,416对256×256的HLS/LST样本。','',
          'Prithvi：训练230,659、测试11,757，无独立验证集（源索引名称val按用户确认解释为test）。LSTSR-TB：训练250,449条序列/2,608,772个温度样本，验证13,694条/141,711个，测试13,665条/141,524个。','',
          '复杂度不等于给定输入条件下的预测难度；熵和梯度升高也可能包含噪声，需结合以上超分基线解读。','']
        pos=lines.index('## 文件');lines[pos:pos]=extra
    lines += ['','## 2026-09-13：HLP下载代码补充后的日期解释','',
      '用户提供的第一套下载代码现已确认：Prithvi/HLP的metadata.datetime来自HLS候选观测清单，是检索锚点，并非LST采集日期。上文HLP日期差的0天中位数及同日比例实际衡量所选HLS与检索锚点，不能与LSTSR/HLC真实月份温度日期差作同义比较，也不能据此证明HLP光学与温度同步。',
      '数值误差统计、分层计算与冻结样本未改变；本报告按HLP记录锚点年月进行的分层不能升级为实际LST采集年月的精确匹配。相关来源与窗口、mosaic、网格、上游反射率裁剪说明见[下载来源核对](../HLP_SOURCE_PROVENANCE.md)。','']
    (destination/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')

def receipt(results,match,destination):
    rows=[]
    for ds in ('prithvi','lstsr_tb'):
        for split in ('all','test'):
            r=results[ds][split];m=r['metrics']
            rows.append({'dataset':ds,'input':'synthetic32x+HLS','split':split,'images':r['images'],
             'lr_only_mean_rmse_c':m['synthetic_baseline_rmse_c']['mean'],'guided_mean_rmse_c':m['synthetic_guided_rmse_c']['mean'],
             'train_calibrated_mean_rmse_c':None})
    for split in ('all','test'):
        r=results['lstsr_tb'][split];m=r['metrics']
        rows.append({'dataset':'lstsr_tb','input':'real_MODIS+HLS','split':split,'images':r['images'],
         'lr_only_mean_rmse_c':m['native_baseline_rmse_c']['mean'],'guided_mean_rmse_c':m['native_guided_rmse_c']['mean'],
         'train_calibrated_mean_rmse_c':m['native_guided_train_calibrated_rmse_c']['mean']})
    src={'label':'全量 LST 与 HLS 冻结数据','files':[{'label':'Prithvi 冻结索引与数据包'},{'label':'LSTSR-TB 冻结索引与原始温度、HLS NPZ'}],
     'metricDefinitions':[{'label':'图像等权RMSE','definition':'先计算每张图有效目标像元的温度RMSE，再对有评分支持的图像等权平均，单位°C。'}],
     'filters':['全部冻结可用LST样本及对应HLS；分别汇总全量与测试集','Prithvi源val按用户确认报告为测试','32倍空间尺寸差'],
     'caveats':['岭回归只在输入LR温度与粗网格HLS之间拟合，没有使用目标HR拟合或选择参数。','这是一条固定光学引导基线，不是所有深度超分模型的难度上限。','Prithvi没有真实MODIS低分辨率配对。'],
     'evidenceFlow':[{'kind':'calculation','title':'光学引导','detail':'固定λ=1，标准化六波段HLS，在至少16个低分辨率有效格点拟合。输出为插值LR温度加上HLS线性高频细节；光学不可用位置回退到原插值。'},
      {'kind':'validation','title':'全量覆盖和一致性','detail':'全部温度样本键、月份、有效像元数以及光学索引均与冻结版本一致；纯温度基线与第一阶段逐图一致。'},
      {'kind':'calculation','title':'训练集温差校准','detail':'仅用LSTSR训练集拟合平均温差，并固定用于测试集；没有使用测试集目标来拟合校准量。'}]}
    dates=[]
    for ds in ('prithvi','lstsr_tb'):
        r=results[ds]['all']
        dates.append({'dataset':ds,'images':r['images'],'median_gap_days':r['metrics']['hls_date_gap_days']['p50'],
          'same_day_fraction':r['date_gap_fractions']['0'],'over_365_days_fraction':r['date_gap_over_365_fraction']})
    ds_source={'label':'全部样本的温度与HLS日期元数据','files':[{'label':'Prithvi 与 LSTSR 冻结源索引'}],
      'metricDefinitions':[{'label':'日期差','definition':'HLS日期与温度侧元数据日期之差的绝对值，单位为日历天。'}],
      'caveats':['Prithvi温度侧日期来自metadata.datetime，未独立确认其为LST采集时间。','日期相隔较长不自动意味着几何错位，也不能单独证明模型误差的原因。']}
    payload={'schemaVersion':1,'items':[
      {'id':'guided-sr-census','title':'低分辨率LST与HLS输入下的全量超分基线比较','queries':[{'id':'full-guided','source':src,'columns':list(rows[0]),'rows':rows}]},
      {'id':'optical-date-alignment','title':'辅助光学影像与目标温度的日期差','queries':[{'id':'full-dates','source':ds_source,'columns':list(dates[0]),'rows':dates}]}]}
    matched_rows=[]
    for split,m in match.items():
        matched_rows.append({'scope':split,'tiles':m['tiles'],'cells':m['cells'],
          'prithvi_guided_mean_rmse_c':m['prithvi_guided_mean_rmse_c'],'lstsr_guided_mean_rmse_c':m['lstsr_guided_mean_rmse_c'],
          'difference_c':m['lstsr_minus_prithvi_mean_rmse_c'],
          'cluster_interval_low_c':m['tile_cluster_bootstrap_95_percentile_interval_c'][0],
          'cluster_interval_high_c':m['tile_cluster_bootstrap_95_percentile_interval_c'][1]})
    if matched_rows:
        match_source={'label':'逐样本全量引导误差的地理和季节分层',
          'files':[{'label':'全量引导指标 Parquet 与 matched_tile_months.csv'}],
          'filters':['2019—2025年，目标评分覆盖率至少90%','共同瓦片×月份单元等权；year_month行进一步要求相同年份'],
          'metricDefinitions':[{'label':'差值','definition':'LSTSR-TB减去Prithvi的单元等权平均RMSE，单位°C。'}],
          'caveats':['相同瓦片和月份不保证是同一裁剪区域或同一天。','区间来自2000次瓦片整体重采样；共享测试瓦片很少时不能推广。']}
        payload['items'].append({'id':'matched-comparison','title':'控制地区和季节后的比较','queries':[{
          'id':'matched-guided','source':match_source,'columns':list(matched_rows[0]),'rows':matched_rows}]})
    complexity=json.loads((plain.OUT/'summary.json').read_text())['results']
    complexity_rows=[]
    for ds in ('prithvi','lstsr_tb'):
        r=complexity[ds]['all'];m=r['metrics']
        complexity_rows.append({'dataset':ds,'images':r['images'],'mean_image_std_c':m['std_c']['mean'],
          'mean_image_iqr_c':m['iqr_c']['mean'],'mean_image_entropy_bits':m['entropy_bits']['mean'],
          'mean_adjacent_gradient_c':m['gradient_c']['mean']})
    complexity_source={'label':'全部目标温度图像的复杂度统计',
      'files':[{'label':'全量温度统计 summary.json 与逐样本 Parquet'}],
      'metricDefinitions':[{'label':'复杂度均值','definition':'标准差、IQR、熵及相邻像元绝对温差均先逐张计算，再对全部温度样本等权平均。'}],
      'filters':['仅使用有效目标温度像元','温度熵采用固定1°C直方图分箱'],
      'caveats':['图像复杂度不是给定低分辨率温度与光学输入后任务难度的充分指标。']}
    payload['items'].append({'id':'target-complexity','title':'全数据集的单张温度图像复杂度','queries':[{
      'id':'complexity-census','source':complexity_source,'columns':list(complexity_rows[0]),'rows':complexity_rows}]})
    labels={'dataset':'数据集','input':'输入设置','split':'划分','scope':'分层范围','images':'温度样本数',
      'lr_only_mean_rmse_c':'仅LST RMSE均值(°C)','guided_mean_rmse_c':'LST＋HLS RMSE均值(°C)',
      'train_calibrated_mean_rmse_c':'训练温差校准后RMSE(°C)','median_gap_days':'日期差中位数(天)',
      'same_day_fraction':'同日比例','over_365_days_fraction':'超过365天比例','tiles':'共同瓦片数','cells':'共同单元数',
      'prithvi_guided_mean_rmse_c':'Prithvi RMSE(°C)','lstsr_guided_mean_rmse_c':'LSTSR-TB RMSE(°C)',
      'difference_c':'差值(°C)','cluster_interval_low_c':'重采样区间下界(°C)','cluster_interval_high_c':'重采样区间上界(°C)',
      'mean_image_std_c':'单图标准差均值(°C)','mean_image_iqr_c':'单图IQR均值(°C)',
      'mean_image_entropy_bits':'单图熵均值(bit)','mean_adjacent_gradient_c':'相邻温差均值(°C)'}
    for item in payload['items']:
        for query in item['queries']:
            query['columns']=[{'field':field,'label':labels.get(field,field)} for field in query['columns']]
    json_write(destination/'sources_receipt.json',payload)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',action='store_true');args=ap.parse_args()
    destination=OUT/'benchmark_analysis' if args.benchmark else OUT;destination.mkdir(exist_ok=True)
    if not args.benchmark:
        for p in (OUT,plain.OUT):assert json.loads((p/'status.json').read_text())['status']=='scan_complete'
    df=pads.dataset(OUT/('benchmark_chunks' if args.benchmark else 'chunks'),format='parquet').to_table().to_pandas()
    for c in ('dataset','split','source_split','tile'):df[c]=df[c].astype('category')
    df['synthetic_coverage']=df.synthetic_pixels/65536;df['common_coverage']=df.common_pixels/65536
    df['hls_valid_fraction']=df.hls_valid_pixels/65536;df['hls_coarse_valid_fraction']=df.hls_coarse_valid_cells/64
    df['synthetic_guided_fraction']=df.synthetic_guided_pixels/df.synthetic_pixels.replace(0,np.nan)
    df['native_guided_fraction']=df.native_guided_pixels/df.common_pixels.replace(0,np.nan)
    df['synthetic_gain_c']=df.synthetic_baseline_rmse_c-df.synthetic_guided_rmse_c
    df['native_gain_c']=df.native_baseline_rmse_c-df.native_guided_rmse_c
    calibration=float(df.loc[(df.dataset=='lstsr_tb')&(df.split=='train'),'native_guided_bias_c'].mean())
    df['native_guided_train_calibrated_rmse_c']=np.sqrt(np.maximum(0,df.native_guided_rmse_c**2-2*calibration*df.native_guided_bias_c+calibration**2))
    df['native_guided_oracle_debiased_rmse_c']=np.sqrt(np.maximum(0,df.native_guided_rmse_c**2-df.native_guided_bias_c**2))
    print('Auditing full guidance and baseline parity',flush=True)
    checked=audit(df,args.benchmark);json_write(destination/'coverage_audit.json',checked)
    results,table=aggregate(df);match,mt=matched(df)
    summary={'scope':'benchmark' if args.benchmark else 'full','results':results,'matched_tile_months':match,
      'aggregation_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'native_global_calibration':{'fit_split':'train','weighting':'equal scored training images','subtract_c':calibration,
       'method':'For each image corrected MSE = original MSE - 2*train_bias*image_bias + train_bias^2. Test targets are not used to fit the offset.'}}
    json_write(destination/'summary.json',summary);table.to_csv(destination/'distributions.csv',index=False)
    mt.to_csv(destination/'matched_tile_months.csv',index=False);temporal_table(df).to_csv(destination/'date_gap_strata.csv',index=False)
    report(results,match,destination,args.benchmark)
    if not args.benchmark:
        plot(results,destination);receipt(results,match,destination)
        plain.notebook(destination)
        import nbformat
        path=destination/'analysis.ipynb';nb=nbformat.read(path,as_version=4)
        for cell in nb.cells:
            cell.source=cell.source.replace('lst_sr_census_20260913','lst_hls_sr_census_20260913').replace('synthetic_rmse_c','synthetic_guided_rmse_c').replace('difficulty_comparison.png','guided_difficulty_comparison.png')
            cell.source=cell.source.replace('experiments/data/lst_sr_census.py','experiments/data/lst_hls_sr_census.py').replace('experiments/data/summarize_lst_sr_census.py','experiments/data/summarize_lst_hls_sr_census.py')
        nb.cells[0].source='# 全量：低分辨率LST＋HLS → 高分辨率LST\n\n完整性验收、光学引导基线与分层比较。固定基线结果不代表所有深度模型的难度排序。'
        nbformat.write(nb,path)
    print(json.dumps({'images':len(df),'scope':summary['scope'],'summary':str(destination/'summary.json')}),flush=True)

if __name__=='__main__':main()
