"""Append reviewed interpretation and fixed-gallery evidence to the A05 report."""
import json
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
from eo_data.core import json_write, sha256_file
from .prepare import OUTPUT, ROOT, HERE, ARMS
from .report import read, LABELS
from .train import source_hashes


def main():
    manifest = read(OUTPUT/'manifest.json')
    assert manifest['sources'] == source_hashes()
    for path, digest in manifest['configs'].items():
        assert sha256_file(HERE/path) == digest
    results = read(OUTPUT/'results.json')
    assert results['verified_units'] == 16
    rows = {(r['model'], r['split'], r['condition']): r for r in results['rows']}
    gallery = read(OUTPUT/'gallery/review.json')
    assert gallery['status'] == 'passed' and len(gallery['checks']) == 64
    for artifact in gallery['artifacts']:
        assert sha256_file(OUTPUT/'gallery'/artifact['path']) == artifact['sha256']
    assert sha256_file(OUTPUT/'gallery/predictions_8_records.npz') == gallery['predictions_sha256']
    tile_values = {}
    for arm in ARMS:
        groups = defaultdict(list)
        path = OUTPUT/'full'/f'full_lst_v2_{arm}_seed17/final_test_noisy10/samples.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r['region'] == 'all':
                groups[r['tile_id']].append(r['mse_physical'])
        tiles = sorted(groups)
        assert len(tiles) == 151
        tile_values[arm] = np.array([[sum(groups[t]), len(groups[t])] for t in tiles])
    rng = np.random.default_rng(20260918)
    draws = rng.integers(0, 151, size=(2000, 151))
    comparisons = []
    for a, b in [('absolute', 'absolute_spatial4'), ('absolute', 'residual'),
                 ('residual', 'residual_spatial4'), ('absolute', 'residual_spatial4')]:
        boot = []
        for arm in (a, b):
            values = tile_values[arm][draws].sum(axis=1)
            boot.append(np.sqrt(values[:, 0]/values[:, 1]))
        x = rows[a, 'test', 'noisy10']['regions']['lst/all']['rmse']
        y = rows[b, 'test', 'noisy10']['regions']['lst/all']['rmse']
        comparisons.append(dict(baseline=a, candidate=b, delta_rmse=y-x,
            relative_reduction=(x-y)/x, paired_tile_ci95=np.quantile(boot[1]-boot[0], [.025, .975]).tolist()))
    json_write(OUTPUT/'paired_comparisons.json', dict(split='test', condition='noisy10',
        tiles=151, records=11757, realizations=2, draws=2000, seed=20260918,
        method='Paired tile-cluster percentile bootstrap; sample-equal MSE then square root; conditional on one trained seed, not training-seed uncertainty.',
        comparisons=comparisons))
    base = rows['absolute', 'test', 'noisy10']
    best = rows['residual_spatial4', 'test', 'noisy10']
    def reduction(a, b):
        return 100*(1-b/a)
    lines = ['', '## 对照结论与尚未解决的问题', '',
        f"本轮固定5轮检查点中，残差预测＋空间4倍在完整val/test的干净与缺失条件下，总体RMSE均最低。test缺失输入相对原始预测：总体RMSE下降{reduction(base['regions']['lst/all']['rmse'], best['regions']['lst/all']['rmse']):.2f}%，空间RMSE下降{reduction(base['spatial_rmse'],best['spatial_rmse']):.2f}%，梯度RMSE下降{reduction(base['regions']['lst/all']['gradient_rmse'],best['regions']['lst/all']['gradient_rmse']):.2f}%。缺失区仍为{best['regions']['lst/corrupt']['rmse']:.4f}°C，高于保留区{best['regions']['lst/retained']['rmse']:.4f}°C。", '',
        '分开看两个因素：残差预测主要削弱整图温度偏移；在残差预测上增加空间权重，test缺失输入的空间RMSE从1.0260降至0.9791°C，但均值RMSE从0.0766升至0.1086°C。因此空间加权存在均值与空间误差的权衡，并非所有分量都改善。', '',
        '原始预测在干净输入下明显失稳（test RMSE 5.3100°C；均值项4.3502°C），空间4倍只能部分缓解（2.0259°C）。残差两组干净输入为0.9874/0.9481°C。所有模型训练时都使用约10%缺失输入，因此这里的干净条件是同一去噪模型的输入条件泛化测试，不是另行训练的干净图像模型。结果符合绝对温度预测对缺失条件变化敏感的现象，尚不能据此确定内部原因。干净条件的两个实现不代表两份独立随机扰动。', '',
        '以下差值为候选模型减去对照模型；置信区间用相同151个test tiles配对重采样2000次、记录等权MSE后开方。它只反映当前模型在这些tiles上的抽样不确定性，不包含训练随机种子变化；本轮仅seed17。', '',
        '| 对照 → 候选 | test缺失RMSE差值 °C | 配对95%区间 | 降幅 |',
        '|---|---:|---|---:|']
    labels = dict(zip(ARMS, LABELS))
    for c in comparisons:
        lo, hi = c['paired_tile_ci95']
        lines.append(f"| {labels[c['baseline']]} → {labels[c['candidate']]} | {c['delta_rmse']:.4f} | [{lo:.4f}, {hi:.4f}] | {100*c['relative_reduction']:.2f}% |")
    lines += ['', '## 码字使用分布摘要', '',
        '下表统计全量成员、两个实现、所有680个多尺度索引位置的合并使用。已用码字数只表示至少出现过一次；困惑度表示按出现频率计算的有效词汇规模，不能把它理解成每幅图或最细网格的码字数。未保存逐码频数或分尺度直方图，因此不能从本报告判断具体哪些码字占据头部或最细层是否坍缩。', '',
        '| 集合 | 模型 | 输入 | 已用/1024 | 使用率 | 熵 bit/index | 困惑度 |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in results['rows']:
        t = r['tokens']['lst']
        lines.append(f"| {r['split']} | {r['label']} | {'干净' if r['condition']=='clean' else '缺失'} | {t['used_codes']} | {100*t['usage_fraction']:.1f}% | {t['empirical_entropy_bits']:.3f} | {t['perplexity']:.2f} |")
    lines += ['',
        'test缺失条件下，组合模型使用885/1024码字，但困惑度仅9.26；原始预测为382/1024、5.42。码字覆盖更广，但频率仍高度不均，不能称为码本问题已经解决，也不能仅凭困惑度宣称细节恢复。固定长度索引码率仍为6800 bit/图；残差组另加32 bit均值，共6832 bit/图，均不含共同掩码、头信息和模型。熵不是实际压缩文件大小。', '',
        '## 固定配对预测图（已逐张查看）', '',
        '沿用之前固定选择的8条test记录：231773、232129、232414、232565、232620、233073、233356、234070；未按本轮结果挑选。这是供检查现象的固定示例，不代表总体随机样本。四组使用完全相同输入和view0，每行共用参考温度1%–99%色标；超范围截色仅用于显示。摄氏度由v2反归一化后除以100得到。64个模型/条件/记录预测MSE与完整评估view0结果核对通过，容差0.2%。', '',
        '视觉观察：缺失条件下四组均保留主要冷热区域，但在232129、232414、232565等记录中，细碎纹理和窄线状结构仍被明显平滑。组合模型部分中尺度结构更清楚，但没有一致恢复参考图细节，不能宣称消除了过度平滑。干净条件下，原始预测多幅图出现大片超出显示上限的区域；残差两组更接近参考整体温度。231773的主要冷热边界可重建，但边界过渡与右侧细纹仍变粗。以上视觉描述只针对这8条固定记录，整体排序以完整评估为准。', '']
    for condition in ('noisy10', 'clean'):
        for page in (1, 2):
            image = OUTPUT/'gallery'/f'predictions_{condition}_{page}.png'
            lines.append(f"![{'缺失' if condition=='noisy10' else '干净'}输入，第{page}页]({image.as_posix()})")
            lines.append('')
    lines += [f"固定8例的额外推理及绘图墙钟共{gallery['wall_seconds']:.2f}秒，单独记录在gallery/review.json，未混入正式训练或全量评估GPU-hours。", '',
        '复现：先运行 `python -m experiments.paper1.lst_v2_5epoch.report` 复核全量评估，再运行 `python -m experiments.paper1.lst_v2_5epoch.gallery` 生成固定图，最后运行 `python -m experiments.paper1.lst_v2_5epoch.complete_report` 补充本节。冻结核心、manifest、配置及检查点未改动。', '']
    report = OUTPUT/'REPORT_ZH.md'
    original = report.read_text(encoding='utf-8').split('\n## 对照结论与尚未解决的问题')[0].rstrip()
    report.write_text(original+'\n'+'\n'.join(lines), encoding='utf-8')
    review = read(OUTPUT/'completion_review.json')
    review.update(report_sha256=sha256_file(report), frozen_sources_and_configs='passed',
        gallery_review='gallery/review.json', gallery_checks=64, gallery_images_viewed=4,
        paired_comparisons_sha256=sha256_file(OUTPUT/'paired_comparisons.json'),
        completed_utc=datetime.now(timezone.utc).isoformat())
    json_write(OUTPUT/'completion_review.json', review)
    registry_path = ROOT/'experiments/paper1/registry.json'
    registry = read(registry_path)
    registry['lst_v2_5epoch_trial'].update(status='complete', completed_utc=review['completed_utc'],
        evaluation='16/16 full units passed; matched inputs, membership, two views, checkpoint/sample hashes and MSE decomposition',
        report=str(report.relative_to(ROOT)).replace('\\','/'),
        gallery=str((OUTPUT/'gallery').relative_to(ROOT)).replace('\\','/'),
        gpu_hours=read(OUTPUT/'gpu_hours.json'), completion_review=str((OUTPUT/'completion_review.json').relative_to(ROOT)).replace('\\','/'))
    json_write(registry_path, registry)
    print(json.dumps(comparisons))


if __name__ == '__main__':
    main()
