"""Chinese code usage diagnosis from saved final metrics and read-only subset counts."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[3]


def main():
    report=ROOT/'runs/paper1/var_vs_prithvi_ae_full/report'
    folder=report/'codebook_audit'
    data=json.loads((folder/'results.json').read_text(encoding='utf-8'))
    assert data['status']=='complete' and len(data['runs'])==4
    lines=['# 卷积 VQ-VAE 码本使用情况核查','',
        '**结论：码字使用严重集中，存在明显的码本利用不足和部分索引坍缩迹象，LST 尤其明显；没有出现所有位置都只使用同一个码字的完全坍缩。**', '',
        '分析对象为 A02 四个已完成的卷积 VQ-VAE。Prithvi 连续 AE 没有码本，不涉及这一问题。此次只读取最终检查点并补做编码统计，没有更新任何模型参数。', '',
        '## 如何理解这些指标','',
        '- **用过的码字数**：在指定验证数据中至少出现过一次的不同编号数量。它无法说明使用是否均匀，未出现的码字也不能据此断定在整个训练过程中从未被使用。',
        '- **有效码字数（perplexity）**：由使用频率的熵折算，公式为 `2^(-Σ pᵢ log₂ pᵢ)`。均匀使用 4096 个码字时为 4096；只使用一个码字时为 1。数值约为 7，表示分布熵相当于均匀使用约 7 个码字，并不表示只出现过 7 个不同编号。',
        '- **前 10 个码字占比**：将码字按出现频率排序，查看最常用的 10 个编号占全部索引的比例。', '',
        '本报告把低有效码字数与高度集中的索引使用称为“部分索引坍缩迹象”。严格的训练过程中码字集合持续收缩，还需要同口径的历时统计；当前结果不能证明这种时间上的收缩过程。有关索引坍缩的优化问题，可参考 [Huh 等，ICML 2023](https://proceedings.mlr.press/v202/huh23a.html) 及其[作者项目说明](https://minyoungg.github.io/vqtorch/)。', '',
        '## 全量验证：11,757 条记录','',
        '下表直接读取已完成的全量验证统计，包含十个尺度的全部索引位置，每幅图像 680 个索引，两个验证视图。', '',
        '| 数据 | 训练输入 | 验证输入 | 用过的码字数 / 4096 | 比例 | 有效码字数 |',
        '|---|---|---|---:|---:|---:|']
    for name,run in data['runs'].items():
        m=name.split('_')[1];n='干净' if '_clean_' in name else '加噪'
        for condition,v in run['full_validation_metrics'].items():
            e='干净' if condition=='clean' else '10% 缺失'
            lines.append(f"| {m.upper()} | {n} | {e} | {v['used_codes']} | {v['usage_fraction']*100:.1f}% | {v['perplexity']:.2f} |")
    lines+=['',
        'HLS 有超过 3000 个编号出现过，但整体有效码字数约为 13–15；LST 在训练对应的验证条件下约为 7。由此可见，“用过很多码字”并不等于码本利用充分。LST 加噪训练模型在干净输入上虽然使用了更多不同编号，但重建误差反而更高，因此使用率本身也不是重建质量指标。', '',
        '## 补充统计：固定 256 条验证记录','',
        '补充统计使用与训练期间阶段性诊断一致的 256 条验证记录，各模型的记录集合完全相同。每条记录两个视图；干净输入的两个视图相同，重复计数不改变使用频率。下表只比较与训练条件对应的验证输入。请勿把这里的前 10 占比当成已重算的全量占比。', '',
        '| 数据 | 训练→验证 | 前 10 个码字占比 | 有效参考位置的前 10 占比 | 覆盖 99% 索引所需码字数 |',
        '|---|---|---:|---:|---:|']
    for name,r in data['runs'].items():
        condition='clean' if '_clean_' in name else 'noisy10'
        v=r['evaluations'][condition];p=v['pooled'];valid=v['pooled_reference_valid']
        label='干净→干净' if condition=='clean' else '加噪→10% 缺失'
        lines.append(f"| {name.split('_')[1].upper()} | {label} | {p['top10_fraction']*100:.2f}% | {valid['top10_fraction']*100:.2f}% | {p['codes_covering_99pct']} |")
    lines+=['',
        '“有效参考位置”要求该尺度位置对应区域的参考有效像素比例至少为 75%。过滤后仍有相近的集中程度，因而不能把现象仅归结为无效像素的统一填充值。', '',
        '## 分尺度有效码字数（256 条记录）','',
        '| 尺度 | HLS 干净训练 | HLS 加噪训练 | LST 干净训练 | LST 加噪训练 |',
        '|---|---:|---:|---:|---:|']
    names=list(data['runs'])
    for i in range(10):
        scale=data['runs'][names[0]]['evaluations']['clean']['per_scale'][i]['scale']
        values=[]
        for name in names:
            condition='clean' if '_clean_' in name else 'noisy10'
            values.append(data['runs'][name]['evaluations'][condition]['per_scale'][i]['all_positions']['perplexity'])
        lines.append(f'| {scale}×{scale} | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines+=['',
        '各尺度都存在较低的有效码字数，现象并非只集中在最后一个尺度。粗尺度可用位置较少，因此“出现过多少码字”受样本量限制，不能直接横向比较原始数量。残差量化也不要求每个尺度均匀使用全部码字；但本次跨尺度的集中程度，结合重建不足，值得优先排查。', '',
        '![码字频率与各尺度有效码字数](codebook_usage_zh.png)', '',
        '## 额外核查与解释边界','',
        '- 四个码本均包含 4096 个数值上互不完全相同的 32 维向量。因此，目前看到的是索引选择集中，不是所有码字向量完全变成同一个值。',
        '- 原生 `ema_vocab_hit_SV` 仅在分布式初始化后更新，这些单进程训练检查点中的该缓冲区全为 0。这里的 0 表示未记录，不能用于判断坍缩。全量指标和本次补查均来自实际编码索引的独立计数。',
        '- 补查的 10% 缺失输入统计与已保存的第 10,813 步、256 条记录诊断结果一致。',
        '- 前 10 集中占比、分尺度结果和参考有效位置过滤均来自固定 256 条记录；全量有效码字数和用过的码字数来自 11,757 条记录。',
        '- 这支持码本利用不足的判断，但尚不能证明它单独导致所有重建误差，也不能仅据此选择某种修复方案。没有修改冻结的模型、数据预处理或训练配置。', '',
        '数据：[机器可读核查结果](codebook_audit/results.json)。每个条件的原始分尺度计数与记录编号保存在同目录的 `_counts.npz` 文件中。', '']
    (report/'CODEBOOK_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    font=Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,m in enumerate(('hls','lst')):
        for n,color,style in [('clean','#16658B','-'),('noisy','#B66A28','--')]:
            name=f'full_{m}_conv_{n}_seed17';condition='clean' if n=='clean' else 'noisy10'
            with np.load(folder/f'{name}_{condition}_counts.npz') as z:counts=z['counts'];scales=z['scales']
            v=data['runs'][name]['evaluations'][condition]
            label='干净训练→干净输入' if n=='clean' else '加噪训练→10% 缺失输入'
            pooled=counts.sum(0);cdf=np.cumsum(np.sort(pooled)[::-1])/pooled.sum()*100
            axes[0,col].plot(np.arange(1,4097),cdf,color=color,linestyle=style,label=label)
            axes[1,col].plot(np.arange(10),[r['all_positions']['perplexity'] for r in v['per_scale']],
                color=color,linestyle=style,marker='o' if n=='clean' else 's',markersize=4,label=label)
        axes[0,col].set(title=m.upper()+'：高频码字累计占比',xlabel='按频率排序的前 N 个码字（对数刻度）',ylabel='累计索引占比（%）',xscale='log',xlim=(1,4096),ylim=(0,101))
        axes[0,col].axvline(10,color='#888888',linestyle=':',linewidth=1)
        axes[0,col].legend(fontsize=9,loc='lower right')
        axes[1,col].set(title=m.upper()+'：各尺度有效码字数',xlabel='量化尺度',ylabel='有效码字数（perplexity）',ylim=(0,22))
        axes[1,col].set_xticks(np.arange(10),[f'{s}×{s}' for s in scales],rotation=35,ha='right')
        for row in (0,1):axes[row,col].grid(alpha=.2)
    fig.suptitle('卷积 VQ-VAE 码本核查｜相同 256 条验证记录，每条 2 个视图\n上图：累计使用频率；下图：由分布熵折算的有效码字数',fontsize=14)
    fig.savefig(report/'codebook_usage_zh.png',dpi=150);plt.close(fig)
    print(report/'CODEBOOK_ZH.md')


if __name__=='__main__':main()
