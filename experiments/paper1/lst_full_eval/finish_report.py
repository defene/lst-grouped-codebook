"""Add reviewed full-test interpretations and paired tile intervals to the Chinese report."""
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from eo_data.core import json_write, sha256_file
from experiments.paper1.lst_full_eval.run import OUTPUT, ARMS, summarize_report


def main():
    out = OUTPUT
    review = json.loads((out/'completion_review.json').read_text(encoding='utf-8'))
    assert review['status'] == 'passed' and len(review['units']) == 16
    summarize_report(out)
    rows = json.loads((out/'results.json').read_text(encoding='utf-8'))['rows']
    lookup = {(r['model'], r['split'], r['condition']): r for r in rows}
    test = {a: lookup[a, 'test', 'noisy10'] for a, _, _ in ARMS}
    sums = {}
    tile_counts = None
    for arm in test:
        groups = defaultdict(list)
        for line in (out/arm/'test/noisy10/samples.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['region'] == 'all':
                groups[row['sample_id']].append(row)
        tiles = defaultdict(list)
        for entries in groups.values():
            assert sorted(r['view'] for r in entries) == [0, 1]
            tiles[entries[0]['tile_id']].append(np.mean([r['mse_physical'] for r in entries]))
        names = sorted(tiles)
        counts = np.array([len(tiles[t]) for t in names])
        if tile_counts is None:
            tile_counts = counts
            tile_names = names
        else:
            assert names == tile_names and np.array_equal(counts, tile_counts)
        sums[arm] = np.array([sum(tiles[t]) for t in names])
    rng = np.random.default_rng(20260915)
    draws = rng.integers(len(tile_counts), size=(2000, len(tile_counts)))
    denominator = tile_counts[draws].sum(1)
    bootstrap = {a: np.sqrt(s[draws].sum(1)/denominator) for a, s in sums.items()}
    pairs = [('absolute_spatial4', 'absolute'), ('residual', 'absolute'),
             ('residual_spatial4', 'residual'), ('residual_spatial4', 'absolute_spatial4'),
             ('residual_spatial4', 'absolute')]
    comparisons = []
    for new, old in pairs:
        a, b = test[new]['regions']['lst/all']['rmse'], test[old]['regions']['lst/all']['rmse']
        comparisons.append({'new': new, 'reference': old, 'rmse_difference_C': a-b,
                            'relative_reduction': 1-a/b,
                            'paired_tile_ci95_C': np.quantile(bootstrap[new]-bootstrap[old], [.025, .975]).tolist()})
    json_write(out/'paired_comparisons.json', {'split': 'test', 'condition': 'noisy10', 'tiles': len(tile_counts),
        'pairs': int(tile_counts.sum()), 'replicates': 2000, 'seed': 20260915, 'comparisons': comparisons,
        'scope': 'tile resampling uncertainty conditional on these four fixed single-seed checkpoints; not training-seed uncertainty'})
    baseline, best = test['absolute'], test['residual_spatial4']
    reduction = 1-best['regions']['lst/all']['rmse']/baseline['regions']['lst/all']['rmse']
    spatial = 1-best['spatial_rmse']/baseline['spatial_rmse']
    gradient = 1-best['regions']['lst/all']['gradient_rmse']/baseline['regions']['lst/all']['gradient_rmse']
    status = json.loads((out/'status.json').read_text(encoding='utf-8'))
    runtime = {'evaluation_job_wall_hours': status['inference_job_wall_hours'],
        'sum_evaluation_unit_wall_hours': sum(r['elapsed_seconds'] for r in rows)/3600,
        'peak_allocated_gib': max(r['memory']['peak_allocated_gib'] for r in rows),
        'peak_reserved_gib': max(r['memory']['peak_reserved_gib'] for r in rows),
        'gpu_count': 1, 'new_training_hours': 0,
        'definition': 'elapsed evaluation job on one GPU, includes data loading/CPU metrics/I/O; not GPU kernel busy time or training cost',
        'scope': 'full evaluation only; preliminary 8-sample smoke and post-hoc CPU review excluded'}
    json_write(out/'evaluation_runtime.json', runtime)
    report = out/'REPORT_ZH.md'
    original = report.read_text(encoding='utf-8')
    first, rest = original.split('\n', 1)
    lead = (f'完整 test 上，“残差预测＋空间4倍”在这四个固定检查点中误差最低：缺失输入 RMSE '
            f"{best['regions']['lst/all']['rmse']:.4f}°C，相对原始预测降低 {reduction:.2%}；"
            f'空间 RMSE 降低 {spatial:.2%}，梯度 RMSE 降低 {gradient:.2%}。'
            '这支持残差预测与空间损失加权的组合在当前短程实验中有效，但不足以认定图像纹理已充分恢复。')
    extra = ['','## 完整测试集的解释','',
        f"单独提高空间损失，缺失输入总体 RMSE 从 {baseline['regions']['lst/all']['rmse']:.4f} 降至 {test['absolute_spatial4']['regions']['lst/all']['rmse']:.4f}°C，但均值 RMSE 从 {baseline['mean_rmse']:.4f} 升至 {test['absolute_spatial4']['mean_rmse']:.4f}°C。它主要改善图内结构误差，同时牺牲了部分逐图温度均值精度。", '',
        f"单独改为残差预测，均值 RMSE 降至 {test['residual']['mean_rmse']:.4f}°C，总体 RMSE 为 {test['residual']['regions']['lst/all']['rmse']:.4f}°C；继续加空间损失后，空间 RMSE 从 {test['residual']['spatial_rmse']:.4f} 降至 {best['spatial_rmse']:.4f}°C。两项改变改善的误差组成不同。", '',
        '四组都只用约 10% 缺失输入训练。原始预测两组遇到干净输入时 RMSE 分别为 6.7785 和 7.5818°C，明显高于缺失输入；残差两组的干净与缺失误差接近。这与之前短程图像观察一致，说明当前模型对输入条件的依赖不同；这次补测未单独确定分布偏移的机制。', '',
        f"组合模型缺失区 RMSE 为 {best['regions']['lst/corrupt']['rmse']:.4f}°C，保留区为 {best['regions']['lst/retained']['rmse']:.4f}°C。缺失区仍更困难，不能只用总体均分判断补全质量。", '',
        '| 模型 | test 缺失输入梯度 RMSE °C/像素间差分 | 使用码字 / 1024 | 码字困惑度 |',
        '|---|---:|---:|---:|']
    for arm in test:
        r = test[arm]
        extra.append(f"| {r['label']} | {r['regions']['lst/all']['gradient_rmse']:.4f} | {r['tokens']['lst']['used_codes']} | {r['tokens']['lst']['perplexity']:.2f} |")
    extra += ['', '组合模型使用 710 个码字，但困惑度只有约 4.98，说明总体 token 频率仍很集中。全量样本更多也更容易覆盖稀有码字，不能把本次使用码字数量直接与历史 256 条子集比较后声称码本改善；困惑度增长也不等于图内细节已恢复。本次没有新生成或重新审阅预测图像，关于纹理的判断仍需配对图和针对细节的指标支持。', '',
        '## 同一 test 记录上的配对比较','',
        '固定四个模型，对 151 个 tile 做 2,000 次配对重采样；每次按抽中 tile 的记录数加权汇总 MSE 后开方。负差值表示前者更好。区间只反映该测试集合的 tile 重采样变动，不包含随机初始化或训练种子变动，也不能消除历史 test 监测带来的选择影响。', '',
        '| 比较（前者减后者） | RMSE 差 °C | 95% 配对 tile 区间 °C |','|---|---:|---:|']
    for c in comparisons:
        lo, hi = c['paired_tile_ci95_C']
        extra.append(f"| {test[c['new']]['label']} − {test[c['reference']]['label']} | {c['rmse_difference_C']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    extra += ['', '## 完成核验与推理成本','',
        '16 项均通过完整成员、两次实现、四模型同条件输入摘要、检查点与逐条误差哈希校验；逐条重新计算总体 RMSE，并核对总体 MSE＝均值 MSE＋空间 MSE。校验脚本曾因 Windows 默认 GBK 编码无法读取中文字段而报错，已显式改为 UTF-8 后通过；无需重新推理，原始指标与模型未修改。', '',
        f"完整补测作业经过 {runtime['evaluation_job_wall_hours']:.3f} 小时，使用一张 RTX 5090；逐单元计时合计 {runtime['sum_evaluation_unit_wall_hours']:.3f} 小时。它包括读取数据、CPU 指标和文件写入，不是纯 GPU 内核运算时间。本次没有训练，不能计为单个 VQ-VAE 的训练 GPU-hours。推理峰值张量分配 {runtime['peak_allocated_gib']:.3f} GiB，分配器峰值保留 {runtime['peak_reserved_gib']:.3f} GiB；这不等于整卡系统占用。", '',
        '复现证据：[逐项复核](completion_review.json)、[配对比较](paired_comparisons.json)、[推理计时](evaluation_runtime.json)、[完整机器可读指标](results.json)、[数据和兼容协议](protocol.json)。', '']
    report.write_text(first+'\n\n'+lead+'\n'+rest+'\n'+'\n'.join(extra), encoding='utf-8')
    registry_path = Path('experiments/paper1/registry.json')
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    registry['version'] += 1
    registry['lst_full_evaluation'].update(status='complete', completed_utc=status['utc'],
        verified_units=16, review='runs/paper1/lst_full_eval_20260914/completion_review.json',
        report='runs/paper1/lst_full_eval_20260914/REPORT_ZH.md',
        test_noisy_best_fixed_checkpoint='residual_spatial4', test_noisy_rmse=best['regions']['lst/all']['rmse'],
        relative_rmse_reduction_vs_absolute=reduction, inference_job_wall_hours=runtime['evaluation_job_wall_hours'])
    json_write(registry_path, registry)
    print(json.dumps({'reduction': reduction, 'spatial_reduction': spatial, 'gradient_reduction': gradient,
                      'comparisons': comparisons, 'report_sha256': sha256_file(report)}, indent=2))


if __name__ == '__main__':
    main()
