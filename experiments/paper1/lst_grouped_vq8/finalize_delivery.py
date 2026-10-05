"""A07 report-only delivery: inspect existing results; never train or evaluate models."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'runs/paper1/lst_grouped_vq8_20260928'
OLD = ROOT / 'runs/paper1/lst_grouped_vq_20260924'

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
    new_rows = verify_evaluations('full')
    rows = read(OUT / 'comparison_results.json')['rows']
    assert len(new_rows)==8 and len(rows)==12
    gate = read(OUT / 'full/gate.json')
    assert gate['status'] == 'passed'
    detail = ['# A07 与 A06 g4 完整码字统计', '',
              '覆盖 val/test、clean/noisy10、全部组和全部10个尺度。组编号从0开始；不同组的相同编号不是同一码字。', '',
              '频数统计全部传输位置（包括参考有效区之外）。逐尺度总数=N×2×尺度²；跨尺度合计=N×2×680。图内指标先在单张图、单个视图内计算。尺度1没有相邻边，变化率记为—。', '',
              '每个尺度后的完整非零频数采用“码字编号:次数”；未列出的合法编号频数为0。[全部码字（含零）机器可读频数](token_frequencies.json)。熵和困惑度不是实际压缩文件码率。', '']
    frequencies = []
    regional_checks = []
    for r in rows:
        source_root = OLD if r['model']=='g4k256' else OUT
        dest = source_root / 'full' / f"full_lst_grouped_{r['model']}_seed17" / f"final_{r['split']}_{r['condition']}"
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
                   f"原始计数：[token_counts.npz]({Path(__import__('os').path.relpath(dest/'token_counts.npz', OUT)).as_posix()})；SHA256 `{r['token_counts_sha256']}`。", '']
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
    # Independently check the complete common stream, including the A06 reference.
    streams = []
    for arm in ('g4k256','g8k16','g8k256'):
        root = OLD if arm=='g4k256' else OUT
        run = root/'full'/f'full_lst_grouped_{arm}_seed17'
        log = [json.loads(s) for s in (run/'train.jsonl').read_text('utf-8').splitlines()]
        assert [x['step'] for x in log] == list(range(1,17111))
        members=set(np.load(run/'train_record_ids.npy').tolist())
        assert len(members)==218975
        for epoch in range(5):
            part=[x for x in log if x['epoch_index']==epoch]
            ids=[int(s.rsplit(':',1)[1]) for x in part for s in x['sample_ids']]
            assert len(part)==3422 and len(ids)==218975 and set(ids)==members
            assert part[-1]['batch_samples']==31
        assert log[-1]['records_presented']==1094875
        streams.append([x['sample_ids'] for x in log])
    assert streams[0]==streams[1]==streams[2]
    assert len(frequencies)==800 and len(regional_checks)==30
    keyed={(r['model'],r['split'],r['condition']):r for r in rows}
    def metric(arm,split,condition,field='rmse',region='lst/all'):
        r=keyed[arm,split,condition]
        return r[field] if field in ('mean_rmse','spatial_rmse') else r['regions'][region][field]
    def gain(arm,base,split,condition,field='rmse',region='lst/all'):
        return 100*(1-metric(arm,split,condition,field,region)/metric(base,split,condition,field,region))
    report='''# A07：同码率八组量化降低误差，扩到256码字的额外收益较小

## 结论

在相同21792 bit/图（含32 bit可见均温）的固定索引预算下，g8k16优于历史A06 g4k256：test clean总体RMSE由0.6936降至0.6258°C，test noisy10由0.7574降至0.6997°C。验证集也呈相同方向。这支持当前seed17、5轮预算下八组小字典的码率效率，但分组数、每组维度和字典大小同时变化，不能把差异解释为纯粹的组数因果效应。

g8k256的test clean/noisy10为0.6232/0.6961°C，相对g8k16只再降低约0.4%/0.5%，而固定索引位数翻倍。就本次测得的误差与位数权衡，g8k16更有吸引力；单个训练种子不足以证明两种八组方案的微小差异稳定存在。

缺失区仍是主要难点：g8k16的test noisy10缺失区RMSE为1.1972°C，保留区为0.6214°C。更丰富的token变化不等于参考细节正确恢复，四张固定记录图中仍可见纹理平滑和缺失边界变形。

## 比较范围与预算

只训练g8k16、g8k256，直接复用A06冻结核心；g4k256是已完成的固定epoch5历史对照，没有重训。三者均为32维、680个空间位置，分组独立选码后拼接，再使用全通道Phi。损失为可见均温残差预测、空间误差4倍，其他训练语义保持一致；仅输入LST和mask，没有HLS或Prithvi模型参与。

| 模型 | 组数×每组码字 | 每组维数 | 索引bit/位置 | 含均温bit/图 | 码本参数量 |
|---|---:|---:|---:|---:|---:|
| A06 g4k256 | 4×256 | 8 | 32 | 21792 | 8192 |
| A07 g8k16 | 8×16 | 4 | 32 | 21792 | 512 |
| A07 g8k256 | 8×256 | 4 | 64 | 43552 | 8192 |

码率是固定索引位数加一个float32均温，不包含模型权重、mask、文件头，也不是实际熵编码文件大小。g8k256的索引预算为两倍，含均温总位数略小于两倍，不能称同码率。

HLP（HLS-LST-Pairs）v2：train218975、val11684、test11757，显式使用processed_data/hlp_split_v2_20260914；归一化来自剩余新train，QA库沿用A05 split-local库。fine LST为卫星反演参考产品。tile ID隔离不等于已核验地理footprint零重叠。

各模型seed17从零训练5个完整epoch，每轮3422次更新、尾批31实际加权；总17110次更新、1094875次记录呈现，训练每记录1个视图。每轮完整val；最终固定epoch5 last.pt，未以test选模，历史test查看记录保留。技术检查67train、8val/test、2轮4步及2步保存恢复另计，10项单测通过，技术权重不用于正式训练。

## 完整val/test结果

所有误差单位为°C；梯度表示相邻有效像素温差误差。clean没有人工缺失区，记为—。

| 集合 | 模型 | 条件 | 总体RMSE | 均值RMSE | 空间RMSE | 梯度RMSE | 缺失区RMSE | 保留区RMSE |
|---|---|---|---:|---:|---:|---:|---:|---:|
'''
    for split in ('val','test'):
        for condition in ('clean','noisy10'):
            for arm in ('g4k256','g8k16','g8k256'):
                r=keyed[arm,split,condition]; z=r['regions']
                hole=z.get('lst/corrupt',{}).get('rmse')
                h='—' if hole is None else f'{hole:.6f}'
                report+=f"| {split} | {arm} | {condition} | {z['lst/all']['rmse']:.6f} | {r['mean_rmse']:.6f} | {r['spatial_rmse']:.6f} | {z['lst/all']['gradient_rmse']:.6f} | {h} | {z['lst/retained']['rmse']:.6f} |\n"
    report+='\n相对g4k256，g8k16总体RMSE降低：'
    report+='；'.join(f"{s} {c} {gain('g8k16','g4k256',s,c):.2f}%" for s in ('val','test') for c in ('clean','noisy10'))+'。\n\n'
    report+=f"test noisy10中，g8k16的空间、梯度、缺失区和保留区RMSE分别降低{gain('g8k16','g4k256','test','noisy10','spatial_rmse'):.2f}%、{gain('g8k16','g4k256','test','noisy10','gradient_rmse'):.2f}%、{gain('g8k16','g4k256','test','noisy10',region='lst/corrupt'):.2f}%和{gain('g8k16','g4k256','test','noisy10',region='lst/retained'):.2f}%。缺失区收益小于保留区收益；均值RMSE反而由0.048930升至0.050319°C，clean均值项也变差，不能声称所有指标改善。总体MSE的下降主要体现为空间项的下降。\n\n"
    report+=f"g8k256相对g8k16的test clean/noisy10总体RMSE分别降低{gain('g8k256','g8k16','test','clean'):.2f}%/{gain('g8k256','g8k16','test','noisy10'):.2f}%；noisy10缺失区再降低{gain('g8k256','g8k16','test','noisy10',region='lst/corrupt'):.2f}%。其clean均值项更好，但noisy10均值项更差。当前证据没有显示与双倍索引预算相称的大幅重建收益。\n"
    report+='''
先在记录内平均两个视图MSE，再对记录等权平均、开方；不直接平均RMSE。均值项为逐图偏差平方的平均开方，空间项为去均值误差，满足总体MSE=均值MSE+空间MSE。缺失区和保留区各自在该记录有效区域内归一，不能用单个全局缺失比例机械拼回总体RMSE。温度先反归一化再除100。最终每个模型评估val/test各两个条件、每条件两个视图，共93764次记录视图呈现。

## 码字使用、空间变化与跨视图稳定性

[完整码字统计](CODEBOOK_DETAILS_ZH.md)覆盖三模型12个评估单元、80个组单元和800个组尺度单元（A07新增640个）。每个组、每个尺度均列出频数、top1/top10、熵/困惑度、图内独有码字、图内熵和相邻变化；[含零频数JSON](token_frequencies.json)保留所有码字。频数覆盖所有传输位置，含参考无效区；逐尺度总数=N×2×尺度²。

下面为test noisy10逐组摘要。图内指标取最细16×16尺度；频数与top1/top10为组内跨尺度汇总。翻转率仅有逐组跨尺度汇总，现有数据不支持逐尺度跨视图稳定性。不同组中的相同编号不是同一码字，单组熵不是联合熵。

| 模型 | 组 | 使用/K | 困惑度 | top1 | top10 | 最细层图内独有码字 | 图内熵bit | 相邻变化 | 参考区翻转 | 共同可见区翻转 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for arm in ('g4k256','g8k16','g8k256'):
        r=keyed[arm,'test','noisy10']
        for g in r['tokens']['lst']['per_group']:
            s=g['scales'][-1]; gi=g['group']
            flip={key:next(x['flip_rate'] for x in val if x['group']==gi) for key,val in r['stability'].items()}
            report+=f"| {arm} | {gi} | {g['used_codes']}/{g['vocabulary']} | {g['perplexity']:.3f} | {g['top1_fraction']:.2%} | {g['top10_fraction']:.4%} | {s['within_image_mean_unique_codes']:.3f} | {s['within_image_mean_entropy_bits']:.3f} | {s['within_image_adjacent_change_fraction']:.2%} | {flip['reference']:.2%} | {flip['common_visible']:.2%} |\n"
    report+='''
g8k16每组累计用过13–16个码字，但top1达34.97%–89.70%，跨尺度困惑度仅1.44–3.04；g8k256每组累计用过93–128个码字，但top1达42.45%–94.06%，困惑度仅1.35–2.98。没有组完全只用一个码字，但使用仍高度集中。尤其g8k256组0、1、7的低翻转伴随91%以上top1占比，不能单凭低翻转推断抗缺失更强。

g8k256最细层图内独有码字约4.27–6.96，g8k16约2.90–3.57；字典大小和每组表示不同，不应将这些值直接当作整幅图细节正确性的排名。相邻变化只说明token变动，可能包含错误边界。参考有效区和共同可见区使用冻结实现的0.75尺度mask阈值；clean两视图相同，其零翻转不构成抗缺失证据。

## 固定8记录图：四张均已实际查看

沿用历史记录231773、232129、232414、232565、232620、233073、233356、234070，未按本次结果挑图。三模型展示view0、同输入；每行共用参考温度1%–99%分位色标，超出范围截色，灰色表示无效或缺失。图只用于定性查看，不能替代全量两视图指标。

- [干净输入，第1–4条](gallery/predictions_clean_1.png)
- [干净输入，第5–8条](gallery/predictions_clean_2.png)
- [约10%缺失，第1–4条](gallery/predictions_noisy10_1.png)
- [约10%缺失，第5–8条](gallery/predictions_noisy10_2.png)

干净第1页中，八组模型在232414、232565的部分线状结构更清楚，仍未恢复参考中的细碎纹理。干净第2页的233356局部冷热边界更清楚，234070仍明显平滑。缺失第1页的231773海岸遮挡处仍有边界偏移；232414右侧大块缺失区的线状结构不能完整恢复。缺失第2页的232620右侧、233073中部和234070遮挡处仍出现平滑或形状偏差。g8k256与g8k16视觉差异较小，没有证据说明它在每个位置都更好。标题、色标、记录标签均可读，未发现图层错位。

[绘图核验](gallery/review.json)通过全部48个模型×条件×记录MSE对齐检查（与完整评估同记录view0比较，rtol0.002），同时保存输入、检查点、预测数组和4个PNG的哈希。

## GPU用时与中断

以下按同一GPU作业活动区间并集统计，包含训练期完整val和存盘，是作业占用墙钟而非CUDA内核计时。技术检查、正式训练、最终评估、额外推理绘图分别列出；共享GPU的单模型墙钟不能相加当成GPU-hours。

| 阶段 | 下界(h) | 上界(h) |
|---|---:|---:|
'''
    for label,key in [('技术检查','smoke_all'),('正式训练（含每轮val）','full_training'),('最终完整评估','full_evaluation')]:
        report+=f"| {label} | {times[key]['lower']:.6f} | {times[key]['upper']:.6f} |\n"
    report+=f"| 额外推理与绘图墙钟 | {gallery['wall_seconds']/3600:.6f} | {gallery['wall_seconds']/3600:.6f} |\n\n绘图为{gallery['wall_seconds']:.3f}秒，包含CPU绘图与IO，不称纯GPU计算耗时。\n\n"
    report+='| 模型 | 活动墙钟下界–上界(h) | 并行下界/上界情景(h) | 独占下界/上界情景(h) |\n|---|---:|---:|---:|\n'
    for arm in ('g8k16','g8k256'):
        lo=concurrency['lower'][f'full_lst_grouped_{arm}_seed17'];hi=concurrency['upper'][f'full_lst_grouped_{arm}_seed17'];t=times['per_model_training_wall_hours'][arm]
        report+=f"| {arm} | {t['lower']:.6f}–{t['upper']:.6f} | {lo['concurrent']:.6f} / {hi['concurrent']:.6f} | {lo['alone']:.6f} / {hi['alone']:.6f} |\n"
    report+='''
并行/独占列是分别使用结束下界和上界的两种情景，不应将独占列两个数直接当作统计置信区间，也不能机械各分一半并集当作实测单模型计算时间。[原始区间及GPU用时](gpu_hours.json)、[并行情景拆分](training_concurrency.json)。

9月29日中断结束时刻未知：g8k16最后训练日志11:49:56.761022 UTC、g8k256为11:49:56.710223 UTC，作为下界；13:03:36.562732 UTC确认进程全部不存在，作为上界。11:49:59有nvlddmkm事件153，时间吻合但不能证明具体退出原因。上界至9月30日03:01:20恢复之间的确定停机全部排除，不能把整段共享墙钟计入GPU-hours。

用户要求继续后，冻结源码、4配置、调度、数据及QA哈希、检查点和优化器有限值、前两轮成员与A06 g4样本流通过核验；两组从7750步检查点恢复。未提交7751–7828日志由原训练器归档后重放，不重复计入最终训练记录预算，真实活动开销仍计入时间。[恢复证据](recovery_20260930T030027/resume_review.json)。本轮没有再次恢复，也没有重训A06。

## 完成核验

新增8单元成员、双视图、检查点/样本/token哈希、MSE分解和逐组逐尺度计数通过；与A06 g4的同split/条件输入摘要和数据指纹一致。三模型五轮17110步共同样本流重新逐步核对一致。12单元所有30个区域的总体/梯度RMSE从样本重算通过，800个组尺度的频数、熵和top1/top10从原始NPZ重算通过；四张图实际逐张view_image检查。

- [训练门禁](full/gate.json)、[新增8单元评估门禁](full/evaluation_gate.json)
- [12单元完整对照](comparison_results.json)、[新增8单元结果](results.json)
- [交付验证](delivery_validation.json)、[完成审查及哈希](completion_review.json)
- [协议](../../../experiments/paper1/lst_grouped_vq8/PROTOCOL_ZH.md)

结论适用于当前HLP v2、seed17、5轮预算和约10% QA缺失分布。没有多训练种子证据，也没有联合码字熵或实际编码文件率失真测量；不作跨种子、跨数据或已恢复全部细节的推断。报告仅整理已有训练与评估结果，冻结训练语义未改。
'''
    (OUT/'REPORT_ZH.md').write_text(report,encoding='utf-8')
    now=datetime.now(timezone.utc).isoformat()
    write(OUT/'delivery_validation.json',dict(status='passed',utc=now,new_evaluation_units=8,comparison_units=12,regional_checks=regional_checks,group_scale_units=len(frequencies),raw_histogram_metrics_recomputed=True,common_a06_training_stream_verified=True,gallery_checks=48,visual_review=dict(images=[a['path'] for a in gallery['artifacts']],method='Actual view_image inspection of each of four images in the completion turn',observations='Selected linear structures clearer with eight groups; fine texture smoothed and masked boundaries distorted; g8k256 improvement visually modest.'),frozen_training_modified=False))
    registry_path=ROOT/'experiments/paper1/registry.json'
    registry=read(registry_path)
    if not (OUT/'registry_before_completion.json').exists():
        shutil.copy2(registry_path,OUT/'registry_before_completion.json')
    entry=registry['lst_grouped_vq8_trial']
    entry.update(status='complete',updated_utc=now,completed_utc=read(OUT/'status.json')['utc'],report=(OUT/'REPORT_ZH.md').relative_to(ROOT).as_posix(),report_sha256=sha(OUT/'REPORT_ZH.md'),completion_review=(OUT/'completion_review.json').relative_to(ROOT).as_posix(),gallery=(OUT/'gallery/review.json').relative_to(ROOT).as_posix(),evaluation_units=8,comparison_units=12,final_checkpoint='epoch5 last.pt',gpu_hours={k:v for k,v in times.items() if k in ('full_training','full_evaluation','smoke_all')},additional_gallery_wall_seconds=gallery['wall_seconds'])
    write(registry_path,registry)
    artifacts=['REPORT_ZH.md','CODEBOOK_DETAILS_ZH.md','token_frequencies.json','results.json','comparison_results.json','gpu_hours.json','training_concurrency.json','delivery_validation.json','gallery/review.json']
    write(OUT/'completion_review.json',dict(status='passed',utc=now,units=8,comparison_units=12,report_sha256=sha(OUT/'REPORT_ZH.md'),artifact_hashes={p:sha(OUT/p) for p in artifacts},registry_entry_report_hash=entry['report_sha256'],remaining_delivery=[],visual_review_complete=True,all_training_and_evaluation_complete=True))
    print(json.dumps({'status':'passed','group_scale_units':len(frequencies),'regional_checks':len(regional_checks),'report_sha256':sha(OUT/'REPORT_ZH.md')},ensure_ascii=False))

if __name__=='__main__':
    main()
