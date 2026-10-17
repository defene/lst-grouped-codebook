"""Verify A03 learning/data/checkpoints and write a Chinese 4096-versus-1024 report."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from eo_data.core import sha256_file
from eo_denoise.config import load_config
from eo_denoise.data import PairedDataset
from eo_denoise.engine import source_hashes
from experiments.paper1.codebook1024.make_configs import check_configs
from experiments.paper1.var_vs_prithvi_ae.run_suite import json_write, read, now
from experiments.paper1.var_vs_prithvi_ae.inspect_codebook import describe


def baseline_root(phase):
    return ROOT / f'runs/paper1/var_vs_prithvi_ae_{phase}'


def order_hash(logs):
    return hashlib.sha256(json.dumps([r['sample_ids'] for r in logs]).encode()).hexdigest()


def validate(output, phase):
    manifest = read(output / 'manifest.json')
    paths = check_configs(phase)
    assert manifest['source_hashes'] == source_hashes()
    assert manifest['config_hashes'] == {p.name: sha256_file(p) for p in paths}
    steps, train_count, val_count = (128, 64, 64) if phase == 'pilot' else (10813, 230659, 11757)
    records = {}
    failures = []
    shared_orders = set()
    val_memberships = set()
    for path in paths:
        name = path.stem
        run = output / name
        baseline = baseline_root(phase) / name
        c = load_config(path)
        m = c['data']['modalities'][0]
        assert c == read(run / 'config.json')
        model = read(run / 'model.json')
        assert model['pretrained_weights'] is False
        assert model['parameters'] == read(baseline / 'model.json')['parameters'] - 3072 * 32
        assert model['rates'][m]['vocabulary'] == 1024
        assert model['rates'][m]['tokens'] == 680 and model['rates'][m]['bits_per_image'] == 6800
        ds = PairedDataset(c, 'train')
        expected = set(ds.record_ids.tolist())
        assert len(expected) == train_count
        assert expected == set(np.load(run / 'train_record_ids.npy').tolist())
        assert ds.fingerprint == read(run / 'data_fingerprint.json') == read(baseline / 'data_fingerprint.json')
        ds.close()
        logs = [json.loads(s) for s in (run / 'train.jsonl').read_text().splitlines()]
        old_logs = [json.loads(s) for s in (baseline / 'train.jsonl').read_text().splitlines()]
        assert [r['step'] for r in logs] == list(range(1, steps + 1))
        status = read(run / 'status.json')
        assert status['status'] == 'complete' and status['step'] == steps
        assert all(math.isfinite(r[k]) for r in logs for k in
                   ('reconstruction', 'gradient', 'bottleneck', 'kl', 'total', 'gradient_norm'))
        assert all(r['kl'] == r['cross_view'] == r['stability'] == 0 for r in logs)
        counts = Counter(int(s.rsplit(':', 1)[1]) for r in logs for s in r['sample_ids'])
        assert set(counts) == expected
        assert sum(counts.values()) == sum(r['batch_samples'] for r in logs)
        if phase == 'full':
            assert Counter(counts.values()) == {3: 230619, 4: 40}
            assert sum(counts.values()) == 692017 and logs[-1]['epoch'] == 3 and logs[-1]['consumed'] == 40
        digest = order_hash(logs)
        assert digest == order_hash(old_logs), 'A03 sample order must match its 4096 baseline'
        shared_orders.add(digest)
        early = float(np.mean([r['reconstruction'] for r in logs[:16]]))
        late = float(np.mean([r['reconstruction'] for r in logs[-16:]]))
        if phase == 'pilot' and late >= .9 * early:
            failures.append(name + ': pilot reconstruction did not improve by 10%')
        record = dict(parameters=model['parameters'], steps=steps, train_unique_records=len(expected),
            sample_presentations=sum(counts.values()), sample_order_sha256=digest,
            same_order_as_4096=True, early_reconstruction=early, late_reconstruction=late,
            reduction_percent=100*(1-late/early),
            peak_allocated_gib=max(r['memory']['peak_allocated_gib'] for r in logs), evaluations={})
        checkpoint_sha = sha256_file(run / 'last.pt')
        val_c = copy.deepcopy(c)
        val_c['data']['val_limit'] = val_count
        val_ds = PairedDataset(val_c, 'val')
        expected_val = {'prithvi:val:' + str(i) for i in val_ds.record_ids.tolist()}
        assert len(expected_val) == val_count and not expected.intersection(val_ds.record_ids.tolist())
        val_memberships.add(hashlib.sha256('\n'.join(sorted(expected_val)).encode()).hexdigest())
        for condition, cov in [('clean', 0), ('noisy10', .1)]:
            folder = run / ('eval_' + condition)
            metric = read(folder / 'metrics.json')
            baseline_metric = read(baseline / ('eval_' + condition) / 'metrics.json')
            assert metric['step'] == steps and metric['evaluated_pairs'] == val_count
            assert metric['checkpoint_sha256'] == checkpoint_sha and metric['coverage'] == cov
            assert metric['data_fingerprint'] == val_ds.fingerprint == baseline_metric['data_fingerprint']
            assert math.isfinite(metric['selection_score'])
            if phase == 'pilot' and metric['tokens'][m]['used_codes'] < 2:
                failures.append(name + '/' + condition + ': only one code used')
            by_region = defaultdict(lambda: defaultdict(list))
            all_views = Counter()
            with (folder / 'samples.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line)
                    assert row['modality'] == m and row['sample_id'] in expected_val
                    assert math.isfinite(row['mse_physical']) and row['mse_physical'] >= 0
                    by_region[row['region']][row['sample_id']].append(row['mse_physical'])
                    if row['region'] == 'all':
                        all_views[(row['sample_id'], row['view'])] += 1
            assert all_views == Counter({(sid, v): 1 for sid in expected_val for v in (0, 1)})
            recomputed = {}
            for region, samples in by_region.items():
                rmse = math.sqrt(float(np.mean([np.mean(values) for values in samples.values()])))
                assert math.isclose(rmse, metric['regions'][m + '/' + region]['rmse'], rel_tol=1e-9)
                recomputed[region] = rmse
            with np.load(folder / 'preview.npz') as fresh, np.load(baseline / ('eval_' + condition) / 'preview.npz') as old:
                for suffix in ('_target', '_reference_mask', '_image', '_input_mask', '_corruption_mask'):
                    assert np.array_equal(fresh[m + suffix], old[m + suffix])
            record['evaluations'][condition] = dict(recomputed_rmse=recomputed,
                checkpoint_sha256=checkpoint_sha, metric_sha256=sha256_file(folder / 'metrics.json'),
                pairs=val_count, views=2, tokens=metric['tokens'][m])
        val_ds.close()
        records[name] = record
        print(json.dumps(dict(name=name, checked=True, reconstruction_reduction_percent=record['reduction_percent'])), flush=True)
    assert len(shared_orders) == len(val_memberships) == 1
    gpu = [json.loads(s) for s in (output / 'gpu.jsonl').read_text().splitlines()]
    result = dict(status='failed' if failures else 'passed', phase=phase, verified_utc=now(),
        source_hashes=source_hashes(), configs=manifest['config_hashes'], rows=list(records.values()),
        runs=records, failures=failures, frozen_model_and_data=True, only_codebook_size_changed=True,
        identical_training_sample_order_to_4096=True, raw_sample_rmse_verified=True,
        peak_device_used_gib=max(r['used_gib'] for r in gpu),
        minimum_sampled_device_free_gib=min(r['total_gib']-r['used_gib'] for r in gpu),
        source_manifest_sha256=sha256_file(output / 'manifest.json'),
        validates='finite losses; exact budget/order/data/config/checkpoint; paired full evaluation; pilot learning gate',
        does_not_establish='convergence, benefit from smaller codebook, or absence of code underutilization')
    (output / 'report').mkdir(exist_ok=True)
    json_write(output / ('gate.json' if phase == 'pilot' else 'report/completion_audit.json'), result)
    if failures:
        raise ValueError('; '.join(failures))
    return result


def report(output, phase, audit):
    target = output / 'report'
    target.mkdir(exist_ok=True)
    paths = check_configs(phase)
    lines = ['# A03：卷积 VQ-VAE 码本 4096 与 1024 对比', '',
        '仅修改码本大小；四组 HLS/LST × 干净/加噪模型均随机初始化。Prithvi 连续 AE 无码本，本轮沿用 A02 结果。', '',
        ('小范围测试：64 条训练与 64 条验证记录，每组 128 步。以下结果只用于运行与学习检查，不代表全量效果。' if phase == 'pilot' else
         '全量训练：230,659 条训练记录，每组 10,813 步，三遍完整数据加 40 条；最终在全部 11,757 条验证记录上评估，每条两次输入实现。'), '',
        '每幅图像仍为 680 个索引，码字维度仍为 32。固定长度索引载荷从 8,160 bit 降至 6,800 bit（减少 16.7%，不含模型、掩膜或文件头）。', '',
        '| 模态 | 训练输入 | 验证输入 | 4096 RMSE | 1024 RMSE | RMSE 变化 | 4096 有效码字数 | 1024 有效码字数 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    rows = []
    sources = {}
    for path in paths:
        c = read(path)
        m = c['data']['modalities'][0]
        training = '干净' if c['noise']['coverages'] == [0] else '10% 缺失'
        for condition in ('clean', 'noisy10'):
            files = [baseline_root(phase) / path.stem / ('eval_' + condition) / 'metrics.json',
                     output / path.stem / ('eval_' + condition) / 'metrics.json']
            old, new = [read(p) for p in files]
            for p in files:
                sources[str(p.relative_to(ROOT))] = sha256_file(p)
            a, b = [d['regions'][m + '/all']['rmse'] for d in (old, new)]
            delta = 100*(b/a-1)
            op, np_ = [d['tokens'][m]['perplexity'] for d in (old, new)]
            lines.append(f"| {m.upper()} | {training} | {'干净' if condition == 'clean' else '10% 缺失'} | {a:.6f} | {b:.6f} | {delta:+.2f}% | {op:.2f} | {np_:.2f} |")
            rows.append(dict(name=path.stem, modality=m, training=training, evaluation=condition,
                rmse_change_percent=delta, baseline_regions=old['regions'], regions=new['regions'],
                baseline_tokens=old['tokens'][m], tokens=new['tokens'][m], evaluated_pairs=new['evaluated_pairs']))
    if phase == 'full':
        lookup = {(r['name'], r['evaluation']): r for r in rows}
        matched = [lookup[(f'full_{m}_conv_{n}_seed17', 'clean' if n == 'clean' else 'noisy10')]
                   for m in ('hls', 'lst') for n in ('clean', 'noisy')]
        lines[2:2] = [
            '**结论：将码本从 4096 缩小到 1024，没有解决码字使用集中的问题。HLS 两组及 LST 干净组重建略差；LST 加噪组的总体误差小幅下降。**', '',
            '在训练与验证输入条件匹配时，四组 RMSE 变化依次为：' + '；'.join(
                f"{r['modality'].upper()} {r['training']} {r['rmse_change_percent']:+.2f}%" for r in matched) + '。这是单次训练的观测差异，尚未检验差异的统计显著性。', '',
            '本轮仅训练卷积 VQ-VAE（采用 FoundationVision/VAR 的编码器、解码器与量化器，并适配遥感通道和掩膜）；未训练 VAR 自回归生成模型。训练目标沿用冻结的重建、梯度与 VQ 损失，并非复现原仓库的完整 tokenizer 训练方案。', '',
            '“加噪”沿用既定实验：按云形状人为遮掉约 10% 可靠像素，并提供输入掩膜；加噪组不混入干净输入。', '']
    lines += ['', 'HLS 单位为反射率，LST 为 °C（存储值已是摄氏度 × 100，只除以 100）。RMSE 变化为 (1024 / 4096 − 1)，负值表示误差下降。两次实现先在记录内平均，再对记录等权平均。', '',
        '有效码字数是使用频率分布的困惑度，不等于曾出现过的不同码字总数。应同时比较绝对有效码字数、使用比例与重建误差；仅因为分母缩小而提高使用率，不能证明集中问题得到解决。', '',
        '| 模态 | 训练输入 | 验证输入 | 4096 使用过的码字 | 1024 使用过的码字 |',
        '|---|---|---|---:|---:|']
    for r in rows:
        old, new = r['baseline_tokens'], r['tokens']
        condition = '干净' if r['evaluation'] == 'clean' else '10% 缺失'
        lines.append(f"| {r['modality'].upper()} | {r['training']} | {condition} | {old['used_codes']} / 4096 ({old['used_codes']/4096:.1%}) | {new['used_codes']} / 1024 ({new['used_codes']/1024:.1%}) |")
    if phase == 'full':
        lines += ['', '| 模态 | 训练输入 | 4096 缺失区 RMSE | 1024 缺失区 RMSE | 4096 保留区 RMSE | 1024 保留区 RMSE |',
                  '|---|---|---:|---:|---:|---:|']
        for r in rows:
            if r['evaluation'] == 'noisy10':
                m = r['modality']
                values = [r[k][m + '/' + region]['rmse'] for region in ('corrupt', 'retained') for k in ('baseline_regions', 'regions')]
                lines.append(f"| {m.upper()} | {r['training']} | " + ' | '.join(f'{v:.6f}' for v in values) + ' |')
        r = lookup[('full_lst_conv_noisy_seed17', 'noisy10')]
        old, new = r['baseline_regions'], r['regions']
        clean = lookup[('full_lst_conv_noisy_seed17', 'clean')]
        lines += ['', f"LST 加噪组的缺失区 RMSE 为 {old['lst/corrupt']['rmse']:.6f} → {new['lst/corrupt']['rmse']:.6f} °C，基本未改善；保留区为 {old['lst/retained']['rmse']:.6f} → {new['lst/retained']['rmse']:.6f} °C。总体改善主要体现在保留区，不能解读为缺失区恢复更好。", '',
            f"LST 加噪组在干净输入上的 RMSE 从 {clean['baseline_regions']['lst/all']['rmse']:.3f} 降至 {clean['regions']['lst/all']['rmse']:.3f} °C，但仍明显高于干净训练模型的 {lookup[('full_lst_conv_clean_seed17', 'clean')]['regions']['lst/all']['rmse']:.3f} °C；输入条件变化时的退化仍然存在。"]
        codebook_path = target / 'codebook_audit/results.json'
        if codebook_path.exists():
            fresh = read(codebook_path)
            old = read(baseline_root(phase) / 'report/codebook_audit/results.json')
            assert fresh['status'] == 'complete' and len(fresh['runs']) == 4
            lines += ['', '固定 256 条验证记录的码字集中度（每条两次输入实现；以下使用各模型训练对应的输入条件）：', '',
                '| 模态 | 训练输入 | 4096 前 10 码字占比 | 1024 前 10 码字占比 |', '|---|---|---:|---:|']
            for path in paths:
                c = read(path)
                condition = 'clean' if c['noise']['coverages'] == [0] else 'noisy10'
                for cond in ('clean', 'noisy10'):
                    counts_path = target / f'codebook_audit/{path.stem}_{cond}_counts.npz'
                    with np.load(counts_path) as z:
                        assert z['counts'].shape == (10, 1024)
                        assert np.array_equal(z['counts'].sum(1), z['scales']**2*512)
                        summary = describe(z['counts'].sum(0))
                        assert summary == fresh['runs'][path.stem]['evaluations'][cond]['pooled']
                a, b = [d['runs'][path.stem]['evaluations'][condition]['pooled']['top10_fraction'] for d in (old, fresh)]
                lines.append(f"| {c['data']['modalities'][0].upper()} | {'干净' if condition == 'clean' else '10% 缺失'} | {a:.2%} | {b:.2%} |")
            lines += ['', '分尺度原始计数及参考有效位置计数保存在 codebook_audit 目录。未使用单进程下未记录的 EMA 缓冲区作坍缩判断。']
            assert all(r['exact_distinct_embedding_vectors'] == 1024 for r in fresh['runs'].values())
            lines += ['', '全量验证中，匹配输入条件的 HLS 有效码字数约 11.50–12.22，LST 约 7.79–8.02。固定 256 条记录上，HLS 前 10 码字占比从约 87% 升至约 92%–94%；LST 从约 99.5% 降至约 98%–99%，仍高度集中。仅保留参考有效位置后，占比结论基本不变。', '',
                '四个模型的 1024 个码字向量均两两不完全相同，也没有单一码字占据全部位置；问题表现为索引使用集中。当前统计不能证明训练过程中码字集合持续缩小，也不能单独确定根因。']
        lines += ['', '| 模态 | 训练输入 | 验证输入 | 4096 MAE | 1024 MAE | 4096 RMSE 95% 区间 | 1024 RMSE 95% 区间 |',
                  '|---|---|---|---:|---:|---|---|']
        for r in rows:
            a, b = [r[k][r['modality'] + '/all'] for k in ('baseline_regions', 'regions')]
            intervals = ['[' + ', '.join(f'{v:.5f}' for v in d['rmse_tile_ci95']) + ']' for d in (a, b)]
            condition = '干净' if r['evaluation'] == 'clean' else '10% 缺失'
            lines.append(f"| {r['modality'].upper()} | {r['training']} | {condition} | {a['mae']:.6f} | {b['mae']:.6f} | {intervals[0]} | {intervals[1]} |")
        lines += ['', '区间来自按瓦片重采样的各模型 RMSE 估计，未包含多训练种子的波动，也不是两模型差值的置信区间；不能用区间重叠与否代替配对差异检验。']
    lines += ['', f"核查：**{audit['status']}**。模型代码、归一化、样本顺序、噪声配置与 4096 基线一致；每份验证结果的逐样本 RMSE 和最终检查点身份已核对。", '',
        f"设备显存采样峰值 {audit['peak_device_used_gib']:.2f} GiB；最少剩余 {audit['minimum_sampled_device_free_gib']:.2f} GiB。", '',
        '限制：单随机种子，只有验证集，无独立测试集；训练预算不证明充分收敛。码本张量形状改变会改变部分后续随机初始化的抽样位置，因此同种子不代表所有共享层初始权重逐位相同。', '',
        '完整区域指标、置信区间及来源 SHA-256 见 results.json。']
    gpu_hours = None
    if phase == 'full':
        from experiments.paper1.codebook1024.gpu_hours import summarize as summarize_gpu_hours, markdown as gpu_hours_markdown
        gpu_hours = summarize_gpu_hours(output)
        json_write(target / 'gpu_hours.json', gpu_hours)
        lines += gpu_hours_markdown(gpu_hours)
        figures_path = target / 'figure_sources.json'
        if figures_path.exists():
            figures = read(figures_path)
            assert all((target / name).exists() for name in figures['figures'])
            lines += ['', '图像对比使用固定首条验证记录 prithvi:val:230660，输入、参考与掩膜在全部列中逐元素一致。本图 RMSE 只针对该记录的这一个视图，不代表全量结果。', '',
                'HLS 仍明显丢失细节；LST 仍出现平滑、偏差和输入条件敏感性。该 LST 参考图温度范围较窄，统一参考色标会使预测饱和；每幅预测下方给出本图实际 RMSE。显示截断不参与任何指标计算。', '',
                '![HLS 重建对比](hls_reconstruction_zh.png)', '',
                '![LST 重建对比](lst_reconstruction_zh.png)', '',
                '![码字使用集中度与分尺度困惑度](codebook_usage_zh.png)', '',
                '训练曲线使用归一化重建 MSE 的 100 步滑动平均。末期变化较慢，但三遍数据的预算及这些曲线均不足以证明充分收敛，或证明增加训练时长就能解决细节损失。', '',
                '![训练重建损失](reconstruction_loss_zh.png)', '',
                '图片可由 experiments/paper1/codebook1024/report_figures.py 重新生成；源文件哈希、显示范围与计数核查结果见 figure_sources.json。']
    json_write(target / 'results.json', dict(phase=phase, rows=rows, sources=sources, audit_status=audit['status'], gpu_hours=gpu_hours))
    (target / 'RESULTS_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(str(target / 'RESULTS_ZH.md'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['pilot', 'full'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    audit = read(output / ('gate.json' if args.phase == 'pilot' else 'report/completion_audit.json')) if args.report_only else validate(output, args.phase)
    report(output, args.phase, audit)
