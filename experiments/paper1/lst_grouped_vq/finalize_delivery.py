"""A06 report-only delivery: inspect existing results; never train or evaluate models."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'runs/paper1/lst_grouped_vq_20260924'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    from .report import verify_evaluations
    for name in ('REPORT_ZH.md', 'completion_review.json'):
        backup = OUT / ('automatic_' + name)
        if not backup.exists():
            shutil.copy2(OUT / name, backup)
    rows = verify_evaluations('full')
    gate = read(OUT / 'full/gate.json')
    assert gate['status'] == 'passed'
    detail = ['# A06 完整码字统计', '',
              '覆盖 val/test、clean/noisy10、全部组和全部10个尺度。组编号从0开始；不同组的相同编号不是同一码字。', '',
              '频数统计全部传输位置（包括参考有效区之外）。逐尺度总数=N×2×尺度²；跨尺度合计=N×2×680。图内指标先在单张图、单个视图内计算。尺度1没有相邻边，变化率记为—。', '',
              '每个尺度后的完整非零频数采用“码字编号:次数”；未列出的合法编号频数为0。[全部码字（含零）机器可读频数](token_frequencies.json)。熵和困惑度不是实际压缩文件码率。', '']
    frequencies = []
    regional_checks = []
    for r in rows:
        dest = OUT / 'full' / f"full_lst_grouped_{r['model']}_seed17" / f"final_{r['split']}_{r['condition']}"
        data = [json.loads(line) for line in (dest / 'samples.jsonl').read_text('utf-8').splitlines()]
        for region, metric in r['regions'].items():
            selected = [v for v in data if v['region'] == region.split('/')[1]]
            for field, result in [('mse_physical', 'rmse'), ('gradient_mse_physical', 'gradient_rmse')]:
                per_sample = defaultdict(list)
                for v in selected:
                    if v.get(field) is not None:
                        per_sample[v['sample_id']].append(v[field])
                if per_sample:
                    actual = float(np.sqrt(np.mean([np.mean(v) for v in per_sample.values()])))
                    assert np.isclose(actual, metric[result], rtol=1e-10), (r['model'], region, field)
            regional_checks.append([r['model'], r['split'], r['condition'], region])
        counts = np.load(dest / 'token_counts.npz')['counts']
        detail += [f"## {r['split']} / {r['model']} / {r['condition']}", '',
                   f"原始计数：[token_counts.npz]({(dest/'token_counts.npz').relative_to(OUT).as_posix()})；SHA256 `{r['token_counts_sha256']}`。", '']
        for g in r['tokens']['lst']['per_group']:
            gi = g['group']
            detail += [f"### 组 {gi}（K={g['vocabulary']}）", '',
                       f"跨尺度共 {g['count']} 个码字；使用 {g['used_codes']}/{g['vocabulary']}；熵 {g['entropy_bits']:.6f} bit；困惑度 {g['perplexity']:.6f}；top1 {g['top1_fraction']:.4%}；top10 {g['top10_fraction']:.4%}。", '',
                       '| 尺度 | 总频数 | 使用/K | 熵bit | 困惑度 | top1 | top10 | 图内独有码字均值 | 图内熵均值 | 相邻变化率 |',
                       '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for si, s in enumerate(g['scales']):
                c = counts[gi, si]
                p = c[c > 0] / c.sum()
                assert int(c.sum()) == s['count'] and int((c > 0).sum()) == s['used_codes']
                assert np.isclose(float(-(p*np.log2(p)).sum()), s['entropy_bits'])
                assert np.isclose(float(c.max()/c.sum()), s['top1_fraction'])
                assert np.isclose(float(np.sort(c)[-10:].sum()/c.sum()), s['top10_fraction'])
                adjacent = s['within_image_adjacent_change_fraction']
                adj = '—' if adjacent is None else f'{adjacent:.4%}'
                detail.append(f"| {s['scale']} | {s['count']} | {s['used_codes']}/{s['vocabulary']} | {s['entropy_bits']:.4f} | {s['perplexity']:.4f} | {s['top1_fraction']:.4%} | {s['top10_fraction']:.4%} | {s['within_image_mean_unique_codes']:.4f} | {s['within_image_mean_entropy_bits']:.4f} | {adj} |")
                frequencies.append(dict(model=r['model'], split=r['split'], condition=r['condition'], group=gi, scale=s['scale'], counts=c.tolist()))
            detail += ['', '完整非零码字频数（编号:次数）：', '']
            for si, s in enumerate(g['scales']):
                c = counts[gi, si]
                detail += [f"- 尺度 {s['scale']}：" + '，'.join(f'{i}:{int(v)}' for i,v in enumerate(c) if v)]
            detail += ['', '| 跨视图比较区 | 比较位置数 | 码字翻转率 | 一致率 |', '|---|---:|---:|---:|']
            for region, items in r['stability'].items():
                x = next(x for x in items if x['group'] == gi)
                detail.append(f"| {region} | {x['compared_positions']} | {x['flip_rate']:.4%} | {1-x['flip_rate']:.4%} |")
            detail += ['', '跨视图翻转率按组汇总所有尺度；现有输出没有逐尺度翻转计数，不能据此声称某个尺度的稳定性。reference为参考有效区，common_visible为两视图共同可见区，按冻结实现的尺度掩膜阈值0.75统计。clean的两视图相同，零翻转不构成抗缺失证据。', '']
    write(OUT / 'token_frequencies.json', {'count_scope':'all transmitted positions', 'rows':frequencies})
    (OUT / 'CODEBOOK_DETAILS_ZH.md').write_text('\n'.join(detail)+'\n', encoding='utf-8')
    times = read(OUT / 'gpu_hours.json')
    spans = [i for i in times['intervals'] if i['phase']=='full' and i['stage']=='train']
    concurrency = {}
    for bound in ('lower', 'upper'):
        edges = sorted({v for i in spans for v in (i['start'], i[bound+'_end'])})
        stats = defaultdict(lambda: {'alone':0., 'concurrent':0.})
        for a,b in zip(edges, edges[1:]):
            active = {i['name'] for i in spans if i['start'] <= a and i[bound+'_end'] >= b}
            for name in active:
                stats[name]['alone' if len(active)==1 else 'concurrent'] += (b-a)/3600
        concurrency[bound] = dict(stats)
    write(OUT / 'training_concurrency.json', concurrency)
    gallery = read(OUT / 'gallery/review.json')
    assert gallery['status']=='passed' and len(gallery['checks'])==48
    for art in gallery['artifacts']:
        assert sha(OUT/'gallery'/art['path'])==art['sha256']
    assert sha(OUT/'gallery/predictions_8_records.npz')==gallery['predictions_sha256']
    for check in gallery['checks']:
        assert np.isclose(check['gallery_mse'], check['full_evaluation_mse'], rtol=.002)
    base = (OUT/'automatic_REPORT_ZH.md').read_text('utf-8')
    report = base.split('\n',1)[0] + '\n\n' + '''## 结论与适用范围

在相同 6832 bit/图的固定索引码率下，g2k32 优于 single1024：test clean 总体RMSE从0.9481降至0.8054°C（降低15.1%），test noisy10从0.9851降至0.8616°C（降低12.5%）。val也呈相同方向。本实验支持“两组独立子码本拼接”在这一固定训练预算下有收益。

收益并不等于缺失区细节已解决。test noisy10的保留区RMSE从0.9406降至0.8038°C（降低14.5%），缺失区仅从1.3235降至1.2730°C（降低3.8%）；空间RMSE降低12.5%，梯度RMSE降低4.7%。主要提升来自空间重建和保留区，均值项不是总体误差的主要来源。clean均值RMSE反而由0.0382升到0.0409°C，因此不能说所有指标都改善。

g4k256取得最低总体误差：test clean 0.6936°C、noisy10 0.7574°C，相比single1024分别降低26.8%和23.1%；但21792 bit/图是基线的3.19倍。它是扩容量结果，不能将收益全部归因于分组，也不能称相同码率比较。固定索引预算包含一个float32可见均温，不含模型、mask及文件头；频数熵不是实际压缩码率。

结论限于seed17、固定第5轮、当前HLP v2划分及约10% QA缺失条件。没有多训练种子结果，也没有测量各组联合码字熵；不作跨数据、跨种子的稳定优越性推断。tile聚类区间只刻画当前测试样本不确定性，不能代替训练随机性。HLP是HLS-LST-Pairs，fine LST是卫星反演参考产品；本次只输入LST，没有HLS或Prithvi模型参与。

## 完整总体误差

''' + base.split('\n',1)[1]
    report += '''
## 统计口径与完整码字证据

上述误差先在每条记录内平均两个视图的MSE，再对记录等权平均并开方；不直接平均RMSE。均值项为每图偏差平方的平均开方，空间项为去均值误差，因此总体MSE=均值MSE+空间MSE。缺失区/保留区各自在有效区域内归一，不能用单个全局缺失比例简单拼回总体RMSE。clean无缺失区，记为—。梯度为相邻有效像素温差的误差；温度先反归一化再除100转摄氏度。test noisy10实际平均缺失比例约9.8878%。

[逐组、逐尺度完整统计及非零码字频数](CODEBOOK_DETAILS_ZH.md)列出全部12个评估单元、28个组单元、280个组尺度单元，含总频数、使用码字、top1/top10、熵、困惑度、图内独有码字/熵、相邻变化和逐组跨视图稳定性。[全部含零码字频数](token_frequencies.json)可机器读取；原始NPZ与哈希也在附录中。

没有任何子码本完全退化为只用一个码字，但有效使用高度集中。test noisy10基线top10占99.06%；g2两组为99.02%/99.90%；g4各组约99.85%–99.93%，其中组2的top1达84.73%。所以“累计用过许多码字”不能说明使用均衡。g2最细尺度图内独有码字约10.46/7.19，反而小于基线14.27；g4约5.21–6.16。每组维数、字典大小和组数不同，不能将这些单组指标直接作为整套表示容量的排名。

test noisy10参考有效区跨视图翻转率：基线49.96%；g2两组45.62%/47.55%；g4四组34.68%/25.63%/8.08%/36.84%。g4组2的低翻转伴随强top1集中，不能仅凭低翻转认定更鲁棒。相邻变化率也只描述token空间变化，不验证纹理、边界是否正确。重建质量须联合空间/梯度/缺失区误差与固定样本图判断。

## 固定8记录预测图：已逐张查看

记录沿用历史选择231773、232129、232414、232565、232620、233073、233356、234070，未按本次模型优劣重选。全部展示view0，同条件三模型输入一致；每行使用同一参考图1%–99%分位色标，尾部颜色裁剪，灰色为无效/缺失区。图旁RMSE只对应该记录view0，不代表全量均值。

- [干净输入，第1–4条](gallery/predictions_clean_1.png)
- [干净输入，第5–8条](gallery/predictions_clean_2.png)
- [约10%缺失，第1–4条](gallery/predictions_noisy10_1.png)
- [约10%缺失，第5–8条](gallery/predictions_noisy10_2.png)

四张图均实际打开检查。三模型都保留大尺度冷热分布，分组模型在232414、232565、233356等记录中部分线状边界和局部结构更清楚；细碎纹理仍明显平滑。缺失输入231773的海岸/冷热边界处三者都有形状失真，g2还有局部暖色鼓包；233073、234070缺失部分仍偏平滑。不能声称g4在每个位置都更好，或仅凭更丰富token认定细节正确。

[绘图核验](gallery/review.json)确认48个“模型×条件×记录”的图中MSE与全量评估同记录view0对齐，并保存输入、检查点、预测数组和4张PNG的哈希。附加推理与绘图墙钟11.349秒，单列，不冒充纯GPU计算时间。

## GPU时长与中断说明

上文GPU-hours按同一块GPU的作业活动区间并集统计；包含训练期完整val及检查点IO，不是CUDA内核计时。正式训练39.706–39.751小时、最终评估1.452小时、技术检查0.050小时，分别列出；不能把共享同一GPU的单模型墙钟相加当GPU-hours。

| 模型 | 活动墙钟下界–上界(h) | 与另一训练作业并行(h) | 单独运行(h) |
|---|---:|---:|---:|
'''
    for arm in ('single1024','g2k32','g4k256'):
        lo=concurrency['lower'][f'full_lst_grouped_{arm}_seed17']
        hi=concurrency['upper'][f'full_lst_grouped_{arm}_seed17']
        t=times['per_model_training_wall_hours'][arm]
        report += f"| {arm} | {t['lower']:.6f}–{t['upper']:.6f} | {lo['concurrent']:.6f}–{hi['concurrent']:.6f} | {lo['alone']:.6f}–{hi['alone']:.6f} |\n"
    report += '''
single1024前约2.306小时与g4并行，后约11.995小时单独运行；其14.301小时不能当成全程独占GPU的速度基准。两并行作业也不能机械地各分一半并集时长，当作实测的单模型GPU计算耗时。详细区间见[gpu_hours.json](gpu_hours.json)与[并发拆分](training_concurrency.json)。

9月26日原活动区间结束时刻无法精确恢复，下界05:58:16.352037 UTC、上界06:00:57.325826 UTC；上界至9月27日17:42:27.659302 UTC恢复之间约35.692小时为确定停机，已排除。此前D盘出现Ntfs/disk写入错误并发生系统关机。恢复前完成冻结源代码/6配置、数据及QA指纹、前4轮共同训练流、检查点及优化器有限值核验，证据见[恢复审查](recovery_20260927T1738/resume_review.json)。g2从17073步、g4从15500步恢复；g4未提交的15501–15529步历史日志归档后重放，最终训练记录预算仍精确一致，GPU活动时间保留这部分实际开销。

## 完成核验与复现入口

三组均从零完成5个完整epoch，17110次更新、1094875条记录呈现，尾批31按实际样本加权；共同训练流检查通过。技术检查单列，烟雾权重未用于正式训练。固定epoch5 last.pt，未按test选择模型；历史test查看记录保留，旧test沿用历史成员。train归一化只来自新v2剩余训练集，tile ID隔离不等于已证明地理footprint零重叠。

- [协议](../../../experiments/paper1/lst_grouped_vq/PROTOCOL_ZH.md)
- [五轮训练门禁](full/gate.json)与[12单元评估门禁](full/evaluation_gate.json)：完整val/test成员、两视图、同输入摘要、检查点/样本/token哈希、MSE分解、逐组逐尺度计数核验通过。
- [完整指标](results.json)、[最终交付复核](delivery_validation.json)、[完成审查](completion_review.json)。交付复核另外重算所有区域RMSE和梯度RMSE、频数top1/top10/熵，并核验预测图哈希。

报告是结果解读，不修改冻结训练、模型、配置或manifest，也没有新增训练。
'''
    (OUT/'REPORT_ZH.md').write_text(report,encoding='utf-8')
    now=datetime.now(timezone.utc).isoformat()
    write(OUT/'delivery_validation.json',dict(status='passed',utc=now,evaluation_units=12,regional_checks=regional_checks,group_scale_units=len(frequencies),raw_histogram_metrics_recomputed=True,gallery_checks=48,visual_review=dict(images=[a['path'] for a in gallery['artifacts']],method='Actual view_image inspection of all four images in this completion turn',observations='Grouped reconstructions show sharper selected structures, but fine texture remains smoothed and masked coastal boundaries can distort.'),frozen_training_modified=False))
    registry_path=ROOT/'experiments/paper1/registry.json'
    registry=read(registry_path)
    entry=registry['lst_grouped_vq_trial']
    entry.update(status='complete',updated_utc=now,completed_utc=read(OUT/'status.json')['utc'],report=(OUT/'REPORT_ZH.md').relative_to(ROOT).as_posix(),report_sha256=sha(OUT/'REPORT_ZH.md'),completion_review=(OUT/'completion_review.json').relative_to(ROOT).as_posix(),gallery=(OUT/'gallery/review.json').relative_to(ROOT).as_posix(),evaluation_units=12,final_checkpoint='epoch5 last.pt',gpu_hours={k:v for k,v in times.items() if k in ('full_training','full_evaluation','smoke_all')})
    write(registry_path,registry)
    artifacts=['REPORT_ZH.md','CODEBOOK_DETAILS_ZH.md','token_frequencies.json','results.json','gpu_hours.json','training_concurrency.json','delivery_validation.json','gallery/review.json']
    write(OUT/'completion_review.json',dict(status='passed',utc=now,units=12,report_sha256=sha(OUT/'REPORT_ZH.md'),artifact_hashes={p:sha(OUT/p) for p in artifacts},registry_entry_report_hash=entry['report_sha256'],remaining_delivery=[],visual_review_complete=True,all_training_and_evaluation_complete=True))
    print(json.dumps({'status':'passed','group_scale_units':len(frequencies),'regional_checks':len(regional_checks),'report_sha256':sha(OUT/'REPORT_ZH.md')},ensure_ascii=False))

if __name__=='__main__':
    main()
