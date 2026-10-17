"""Render Chinese scientific figures from frozen A02/A03 evaluation artifacts."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eo_data import open_dataset
from eo_data.core import sha256_file

COLORS = {4096: '#17668B', 1024: '#BE6421'}
STYLES = {'clean': '-', 'noisy': '--'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def render(output):
    baseline = ROOT / 'runs/paper1/var_vs_prithvi_ae_full'
    report = output / 'report'
    assert read(report / 'completion_audit.json')['status'] == 'passed'
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus': False})
    sources = {}
    roots = {4096: baseline, 1024: output}

    def record(path):
        sources[str(path.relative_to(ROOT))] = sha256_file(path)
        return path

    def label(size, training):
        return f'{size}／' + ('干净训练' if training == 'clean' else '加噪训练')

    # Identical sample, target, masks, input and physical display transform in every column.
    labels = ['干净参考图', '模型输入'] + [label(k, n).replace('／', '\n')
                for k in roots for n in STYLES]
    display_metadata = {}
    for m in ('hls', 'lst'):
        ds = open_dataset('prithvi', m, 'val', root=ROOT)
        mean = np.asarray(ds.stats['mean'])[:, None, None]
        std = np.asarray(ds.stats['std'])[:, None, None]
        ds.close()
        fig, axes = plt.subplots(2, 6, figsize=(16, 6.4), layout='constrained')
        limits, im, first_id = None, None, None
        for row, condition in enumerate(('clean', 'noisy10')):
            previews = []
            for size, root in roots.items():
                for training in STYLES:
                    folder = root / f'full_{m}_conv_{training}_seed17' / ('eval_' + condition)
                    with np.load(record(folder / 'preview.npz')) as z:
                        previews.append({k: z[k] for k in z.files})
                    with (folder / 'samples.jsonl').open(encoding='utf-8') as f:
                        sid = json.loads(next(f))['sample_id']
                    if first_id is None:
                        first_id = sid
                    assert sid == first_id
            base = previews[0]
            for p in previews[1:]:
                for suffix in ('target', 'reference_mask', 'image', 'input_mask', 'corruption_mask'):
                    assert np.array_equal(base[m + '_' + suffix], p[m + '_' + suffix])
            reference = base[m + '_reference_mask'][0].astype(bool)
            arrays = [base[m + '_target'], base[m + '_image']] + [p[m + '_prediction'] for p in previews]
            masks = [reference, base[m + '_input_mask'][0].astype(bool)] + [reference] * 4
            for col, (array, mask) in enumerate(zip(arrays, masks)):
                physical = array * std + mean
                ax = axes[row, col]
                ax.set_xticks([])
                ax.set_yticks([])
                if m == 'hls':
                    rgb = np.clip(physical[[2, 1, 0]].transpose(1, 2, 0) / .3, 0, 1)**(1 / 2.2)
                    rgb[~mask] = .35
                    ax.imshow(rgb, interpolation='nearest')
                else:
                    value = physical[0] / 100
                    if limits is None:
                        limits = np.quantile(value[reference], [.01, .99])
                    cmap = plt.get_cmap('inferno').copy()
                    cmap.set_bad('#595959')
                    im = ax.imshow(np.ma.masked_where(~mask, value), cmap=cmap,
                                   vmin=limits[0], vmax=limits[1], interpolation='nearest')
                if row == 0:
                    ax.set_title(labels[col], fontsize=10)
                if col == 0:
                    ax.set_ylabel('干净输入验证' if row == 0 else '10% 缺失输入验证', fontsize=10)
                if col >= 2:
                    error = ((array - base[m + '_target']) * std)[:, reference]
                    rmse = float(np.sqrt(np.mean(error**2))) / (100 if m == 'lst' else 1)
                    unit = ' °C' if m == 'lst' else ' 反射率'
                    ax.set_xlabel(f'本图有效像素 RMSE：{rmse:.4f}{unit}', fontsize=8)
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.8, extend='both',
                         label='地表温度（°C）；参考图 1%–99% 分位数统一色标')
        display = ('RGB：反射率 ÷ 0.3，伽马 1/2.2' if m == 'hls' else '统一色标，超出范围仅在显示时截断')
        fig.suptitle(f'{m.upper()} · 卷积 VQ-VAE 码本大小消融 · {first_id}\n{display}；灰色为无效或缺失像素', fontsize=12)
        fig.savefig(report / f'{m}_reconstruction_zh.png', dpi=150)
        plt.close(fig)
        display_metadata[m] = dict(sample_id=first_id, shared_inputs_verified=True,
                                   lst_limits_celsius=limits.tolist() if limits is not None else None)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    count_check = []
    for col, m in enumerate(('hls', 'lst')):
        for size, root in roots.items():
            audit_dir = root / 'report/codebook_audit'
            audit = read(record(audit_dir / 'results.json'))
            assert audit['status'] == 'complete'
            for training in STYLES:
                name = f'full_{m}_conv_{training}_seed17'
                condition = 'clean' if training == 'clean' else 'noisy10'
                evaluation = audit['runs'][name]['evaluations'][condition]
                with np.load(record(audit_dir / f'{name}_{condition}_counts.npz')) as z:
                    counts = z['counts']
                    scales = z['scales']
                    ids = z['record_ids']
                    assert counts.shape == (10, size) and len(ids) == 256
                    assert np.array_equal(counts.sum(1), scales**2 * 512)
                    if not count_check:
                        common_ids = ids.copy()
                    assert np.array_equal(common_ids, ids)
                    assert np.array_equal(scales, [1, 2, 3, 4, 5, 6, 8, 10, 13, 16])
                    p = counts / counts.sum(axis=1, keepdims=True)
                    entropy = -(p * np.log2(np.where(p > 0, p, 1))).sum(1)
                    pp = 2**entropy
                    assert np.allclose(pp, [s['all_positions']['perplexity'] for s in evaluation['per_scale']])
                    total = np.sort(counts.sum(0))[::-1]
                    assert np.isclose(total[:10].sum() / total.sum(), evaluation['pooled']['top10_fraction'])
                    count_check.append(dict(size=size, run=name, condition=condition,
                                            top10_fraction=float(total[:10].sum() / total.sum())))
                style = dict(color=COLORS[size], linestyle=STYLES[training], label=label(size, training))
                axes[0, col].plot(np.arange(1, size + 1), total.cumsum() / total.sum() * 100, **style)
                axes[1, col].plot(np.arange(10), pp, marker='o' if training == 'clean' else 's', markersize=4, **style)
        top, bottom = axes[:, col]
        top.set_title(m.upper() + '：码字使用集中度')
        top.set_xscale('log')
        top.set_xlim(1, 4096)
        top.set_ylim(0, 102)
        top.axvline(10, color='#666666', linestyle=':', linewidth=1)
        top.set_xlabel('按使用频率排序的前 k 个码字（对数刻度）')
        top.set_ylabel('累计索引占比（%）')
        bottom.set_title(m.upper() + '：分尺度有效码字数')
        bottom.set_xticks(np.arange(10), [f'{s}×{s}' for s in scales], rotation=45)
        bottom.set_ylim(0, 22)
        bottom.set_xlabel('残差量化的空间尺度')
        bottom.set_ylabel('有效码字数（困惑度）')
        for ax in (top, bottom):
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('固定 256 条验证记录 · 每条两个视图 · 训练与验证输入条件匹配\n统计全部索引位置；每幅 680 个索引；此图不是全量验证分布', fontsize=12)
    fig.savefig(report / 'codebook_usage_zh.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    late_windows = []
    for ax, m in zip(axes, ('hls', 'lst')):
        for size, root in roots.items():
            for training in STYLES:
                name = f'full_{m}_conv_{training}_seed17'
                path = record(root / name / 'train.jsonl')
                with path.open(encoding='utf-8') as f:
                    curve = np.array([(r['step'], r['reconstruction']) for r in map(json.loads, f)])
                assert np.array_equal(curve[:, 0], np.arange(1, 10814))
                assert np.isfinite(curve).all() and (curve[:, 1] > 0).all()
                y = np.convolve(curve[:, 1], np.ones(100) / 100, mode='valid')
                ax.plot(curve[99:, 0], y, color=COLORS[size], linestyle=STYLES[training], label=label(size, training))
                before, after = [float(v.mean()) for v in (curve[-2000:-1000, 1], curve[-1000:, 1])]
                late_windows.append(dict(size=size, run=name, before=before, after=after,
                                         reduction_percent=100 * (1 - after / before)))
        ax.set_title(m.upper())
        ax.set_xlabel('优化器更新步数')
        ax.set_ylabel('归一化重建均方误差（MSE）')
        ax.set_yscale('log')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('仅重建损失 · 最近 100 步滑动平均 · 纵轴为对数刻度\n各模态独立归一化；曲线不代表总损失或物理单位误差', fontsize=12)
    fig.savefig(report / 'reconstruction_loss_zh.png', dpi=150)
    plt.close(fig)
    metadata = dict(sources=sources, display=display_metadata, codebook_counts_verified=count_check,
                    training_late_windows=late_windows,
                    figures=[f'{m}_reconstruction_zh.png' for m in ('hls', 'lst')]
                            + ['codebook_usage_zh.png', 'reconstruction_loss_zh.png'])
    (report / 'figure_sources.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'figures': metadata['figures'], 'counts_checked': len(count_check)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/paper1/codebook1024/full')
    render(parser.parse_args().output.resolve())
