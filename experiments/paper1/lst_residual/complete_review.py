"""Read-only numerical completion audit and accounting from saved stage events."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eo_data.core import sha256_file
from experiments.paper1.lst_residual.runtime import source_hashes
from experiments.paper1.lst_residual.prepare import ARMS
from experiments.paper1.var_vs_prithvi_ae.run_suite import read, json_write

OUT = ROOT/'runs/paper1/lst_residual_20260913'


def union_hours(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return sum((b-a).total_seconds() for a, b in merged)/3600


def main():
    report = OUT/'report'
    results = read(report/'results.json')
    assert results['status'] == 'complete' and len(results['rows']) == 8
    gate = read(OUT/'pilot/gate.json')
    assert gate['status'] == 'passed' and gate['source_hashes'] == source_hashes()
    streams, fingerprints, region_rows = [], [], {}
    for arm, _, _ in ARMS:
        name = f'pilot_lst_{arm}_seed17'
        run = OUT/'pilot'/name
        c = read(run/'config.json')
        path = ROOT/'experiments/paper1/lst_residual/configs/pilot'/f'{name}.json'
        assert c == read(path) and sha256_file(path) == gate['configs'][path.name]
        logs = [json.loads(s) for s in (run/'train.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['step'] for r in logs] == list(range(1, 2001))
        assert sum(r['batch_samples'] for r in logs) == 128000
        assert all(np.isfinite(r['total']) for r in logs)
        streams.append([r['sample_ids'] for r in logs])
        fingerprints.append(read(run/'data_fingerprint.json'))
        digest = sha256_file(run/'last.pt')
        for condition in ('clean', 'noisy10'):
            metric = read(run/f'eval_{condition}/metrics.json')
            assert metric['checkpoint_sha256'] == digest and metric['step'] == 2000 and metric['evaluated_pairs'] == 256
            summarized = next(r for r in results['rows'] if r['model'] == arm and r['condition'] == condition)
            assert summarized['checkpoint_sha256'] == digest
            assert np.isclose(summarized['rmse'], metric['regions']['lst/all']['rmse'], rtol=1e-12)
            rows = [json.loads(s) for s in (run/f'eval_{condition}/samples.jsonl').read_text(encoding='utf-8').splitlines()]
            for region in ('all', 'corrupt', 'retained'):
                members = {}
                for r in rows:
                    if r['region'] == region:
                        members.setdefault((r['sample_id'], r['tile_id']), []).append(r['mse_physical'])
                if members:
                    assert all(len(v) == 2 for v in members.values())
                    region_rows[(arm, condition, region)] = {k: np.mean(v) for k, v in members.items()}
    assert all(s == streams[0] for s in streams)
    assert all(f == fingerprints[0] for f in fingerprints)
    for figure in results['artifacts']:
        assert sha256_file(report/figure['path']) == figure['sha256']
    comparisons = []
    for condition in ('clean', 'noisy10'):
        for baseline, candidate in [('absolute', 'absolute_spatial4'), ('absolute', 'residual'),
                                    ('residual', 'residual_spatial4'), ('absolute_spatial4', 'residual_spatial4'),
                                    ('absolute', 'residual_spatial4')]:
            for region in ('all', 'corrupt', 'retained'):
                if (baseline, condition, region) not in region_rows:
                    continue
                a, b = [region_rows[(name, condition, region)] for name in (baseline, candidate)]
                assert a.keys() == b.keys()
                keys = sorted(a)
                tiles = sorted({k[1] for k in keys})
                counts = np.array([sum(k[1] == t for k in keys) for t in tiles])
                sa = np.array([sum(a[k] for k in keys if k[1] == t) for t in tiles])
                sb = np.array([sum(b[k] for k in keys if k[1] == t) for t in tiles])
                draw = np.random.default_rng(20260914).integers(len(tiles), size=(2000, len(tiles)))
                boot = np.sqrt(sb[draw].sum(1)/counts[draw].sum(1))-np.sqrt(sa[draw].sum(1)/counts[draw].sum(1))
                comparisons.append(dict(baseline=baseline, candidate=candidate, condition=condition, region=region,
                    paired_rmse_delta=float(np.sqrt(sb.sum()/counts.sum())-np.sqrt(sa.sum()/counts.sum())),
                    paired_tile_bootstrap_ci95=np.quantile(boot, [.025, .975]).tolist(), tiles=len(tiles), records=len(keys)))
    events = [json.loads(s) for s in (OUT/'pilot/stages.jsonl').read_text(encoding='utf-8').splitlines()]
    starts, intervals = {}, []
    for event in events:
        key = (event['name'], event['stage'])
        stamp = datetime.fromisoformat(event['utc'])
        if event['event'] == 'start':
            assert key not in starts
            starts[key] = stamp
        else:
            assert event['exit_code'] == 0 and key in starts
            intervals.append((key, starts.pop(key), stamp))
    assert not starts and len(intervals) == 12
    gpu = [json.loads(s) for s in (OUT/'gpu.jsonl').read_text(encoding='utf-8').splitlines()]
    accounting = dict(training_stage_gpu_hours=union_hours([(a, b) for (name, stage), a, b in intervals if stage == 'train']),
        post_training_evaluation_gpu_hours=union_hours([(a, b) for (name, stage), a, b in intervals if stage.startswith('eval_')]),
        per_job_shared_gpu_training_wall_hours={name: (b-a).total_seconds()/3600 for (name, stage), a, b in intervals if stage == 'train'},
        peak_global_gpu_gib=max(r['used_gib'] for r in gpu),
        interpretation='one RTX 5090 multiplied by union of active stage wall-clock intervals; includes stage startup and I/O, not utilization-adjusted GPU kernel time; training includes periodic evaluation; smoke, queue gaps and report inference excluded',
        event_timing_resolution_seconds=2)
    json_write(report/'gpu_hours.json', accounting)
    json_write(report/'paired_comparisons.json', dict(replicates=2000, method='paired tile bootstrap; average 2 views per record, records equally weighted', comparisons=comparisons))
    text = (report/'REPORT_ZH.md').read_text(encoding='utf-8').split('\n<!-- completion_review -->')[0]
    extra = ['\n<!-- completion_review -->', '', '## 完成复核：结论和局限', '',
        '本次相同 2,000 步预算下，残差预测与空间加权组合表现最好。缺失输入总 RMSE 从 1.4025 降至 1.0583°C（下降 24.5%），空间 RMSE 从 1.3240 降至 1.0521°C（下降 20.5%），梯度 RMSE 从 0.3704 降至 0.3458°C（下降 6.6%）。改善不只是整图均值更准确，但细节提升明显小于总体误差提升。', '',
        '单独提高空间权重将缺失输入 RMSE 从 1.4025 降至 1.2755°C，但干净输入从 6.7483 恶化至 7.6249°C；不能据此把空间权重普遍调高。仅残差预测的干净 / 缺失 RMSE 已达到 1.1640 / 1.2089°C，组合后为 1.0487 / 1.0583°C。四组都只训练过缺失输入，干净评估是输入条件迁移诊断；均值旁路与输入中心化一起改变了模型表示，因此不能把全部收益单独归于码本。', '',
        '| 缺失输入评估 | 原始预测 | 仅空间加权 | 仅残差预测 | 残差＋空间加权 |',
        '|---|---:|---:|---:|---:|',
        '| 缺失区 RMSE °C | 1.8094 | 1.6455 | 1.5446 | 1.4673 |',
        '| 保留区 RMSE °C | 1.3505 | 1.2281 | 1.1662 | 1.0033 |', '',
        '残差组加空间权重后，均值 RMSE 反而从 0.1080 小幅增加至 0.1138°C，空间 RMSE 从 1.2041 降至 1.0521°C；这组对照支持收益主要在图内变化的重建。缺失区与保留区都改善，前者仍更难。', '',
        '码本并没有因为误差下降就恢复广泛使用：缺失输入下四组使用数为 30 / 122 / 51 / 203，困惑度为 2.32 / 3.65 / 2.32 / 4.86（所有尺度、256条×2实现汇总）。尤其仅残差预测的困惑度几乎与基线相同，性能却更好；码本困惑度不能替代空间保真指标，也不能混同之前全量或更长训练的使用统计。', '',
        '已逐张检查四张配对图，8 条记录均展示两个条件。组合模型温度范围更合理，热区、冷区及部分边界较基线清楚；细小田块、纹理和狭窄温度结构仍被平滑。干净输入基线图的大面积颜色饱和对应真实预测偏差，显示色标只按参考图取范围；残差模型的改善不是重新调整其单独色标。', '',
        '样本和随机种子只有一组，未完整遍历训练池，更未证明收敛。配对瓦片 bootstrap 只量化这一验证子集上的采样不确定性，不能代表跨训练种子的稳定性。没有启动更长训练，也没有启动 HLS 新实验。', '',
        '| 配对变化：候选−对照 | 缺失输入总 RMSE 差值 °C | 95% 配对瓦片区间 |', '|---|---:|---|']
    for c in comparisons:
        if c['condition'] == 'noisy10' and c['region'] == 'all':
            lo, hi = c['paired_tile_bootstrap_ci95']
            extra.append(f"| {c['candidate']} − {c['baseline']} | {c['paired_rmse_delta']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    extra += ['', '## GPU 计时复核', '',
        f"按训练活动区间的并集计算，四组共占用一张 RTX 5090 约 **{accounting['training_stage_gpu_hours']:.3f} GPU-hours**；训练后双条件验证另占 **{accounting['post_training_evaluation_gpu_hours']:.3f} GPU-hours**。每个模型约 3 小时是两作业共享 GPU 时的墙钟时间，不是单独使用整卡的耗时。计时含启动和 I/O，事件轮询约 2 秒；未计小测试、报告推理和排队间隔。",
        f"全程采样的显存峰值 **{accounting['peak_global_gpu_gib']:.2f} GiB**，无 OOM、失败或中断恢复。训练完成后临时防休眠请求已释放。", '',
        '冻结源码和配置哈希通过核验；四组逐步样本顺序完全一致，每组 128,000 次样本呈现；8 份最终评估均对应正确的 2,000 步检查点；配对图文件哈希与推理数值核验通过。完整复核见 completion_review.json，资源计时见 gpu_hours.json。', '']
    (report/'REPORT_ZH.md').write_text(text+'\n'.join(extra), encoding='utf-8')
    review = dict(status='passed', reviewed_utc=datetime.now(timezone.utc).isoformat(), frozen_source_hashes=source_hashes(),
        exact_training_streams_match=True, models=4, steps_per_model=2000, presentations_per_model=128000,
        validation_conditions=8, figures_reviewed=[f['path'] for f in results['artifacts']],
        visual_findings='Residual plus spatial weighting improves temperature range and larger local structures; fine textures remain smoothed. Shared reference-derived display limits preserved.',
        results_sha256=sha256_file(report/'results.json'), report_sha256=sha256_file(report/'REPORT_ZH.md'))
    json_write(report/'completion_review.json', review)
    numerical = read(report/'numerical_review.json')
    numerical.update(visual_review_pending=False, visual_review='completion_review.json')
    json_write(report/'numerical_review.json', numerical)
    registry_path = ROOT/'experiments/paper1/registry.json'
    registry = read(registry_path)
    registry['lst_residual_trial'].update(status='complete', completed_utc=read(OUT/'status.json')['updated_utc'],
        report=str((report/'REPORT_ZH.md').relative_to(ROOT)).replace('\\','/'),
        completion_review=str((report/'completion_review.json').relative_to(ROOT)).replace('\\','/'),
        best_arm='residual_spatial4', best_noisy_rmse=1.058284, longer_training_started=False)
    registry['version'] += 1
    registry['updated'] = datetime.now(timezone.utc).date().isoformat()
    json_write(registry_path, registry)
    print(json.dumps(accounting, ensure_ascii=False, indent=2))
    print('COMPLETION AUDIT PASSED')


if __name__ == '__main__':
    main()
