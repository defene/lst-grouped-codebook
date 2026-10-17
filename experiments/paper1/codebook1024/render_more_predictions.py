"""Scientific image plates for the saved, read-only 16-pair prediction gallery."""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.paper1.codebook1024.more_predictions import DEST, read, write, region_metrics
from eo_data.core import sha256_file

MODELS = [(4096, 'clean'), (4096, 'noisy'), (1024, 'clean'), (1024, 'noisy')]
LABELS = ['原始参考', '实际输入'] + [f'{k} 码字\n' + ('干净训练' if t == 'clean' else '加噪训练') for k, t in MODELS]


def main(destination=DEST, model_keys=MODELS, column_labels=LABELS, write_vq_report=True):
    DEST, MODELS, LABELS = Path(destination), model_keys, column_labels
    results = read(DEST/'results.json')
    assert results['status'] == 'complete' and len(results['rows']) == 256
    images = DEST/'images'
    images.mkdir(exist_ok=True)
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(), 'axes.unicode_minus': False})
    lookup = {(r['model'], r['condition'], r['record_id']): r for r in results['rows']}
    provenance = dict(selection_sha256=sha256_file(DEST/'selection.json'), results_sha256=sha256_file(DEST/'results.json'), pages=[], detail_contacts=[])
    for m in ('hls', 'lst'):
        with np.load(DEST/f'{m}_inputs.npz') as z:
            arrays = {k:z[k] for k in z.files}
        samples = read(DEST/f'{m}_samples.json')
        predictions = {}
        for k, t in MODELS:
            with np.load(DEST/f'{k}_{m}_{t}_predictions.npz') as z:
                predictions[(k, t)] = {n:z[n] for n in z.files}
        mean, std = arrays['mean'], arrays['std']
        unit = '°C' if m == 'lst' else '反射率'
        def physical(x):
            return (x*std+mean)/(100 if m == 'lst' else 1)
        def show(ax, value, mask, limits, crop=None):
            if crop is not None:
                y, x = crop['y'], crop['x']
                value, mask = value[:, y:y+64, x:x+64], mask[y:y+64, x:x+64]
            ax.set_xticks([])
            ax.set_yticks([])
            if m == 'hls':
                rgb = np.clip((value[[2, 1, 0]].transpose(1, 2, 0)-limits[0])/(limits[1]-limits[0]), 0, 1)**(1/2.2)
                rgb[~mask] = .35
                return ax.imshow(rgb, interpolation='nearest')
            cmap = plt.get_cmap('inferno').copy()
            cmap.set_bad('#595959')
            return ax.imshow(np.ma.masked_where(~mask, value[0]), vmin=limits[0], vmax=limits[1], cmap=cmap, interpolation='nearest')

        for i, sample in enumerate(samples):
            rid = sample['record_id']
            crop = sample['crop']
            target = physical(arrays['target'][i])
            mask = arrays['reference_mask'][i, 0].astype(bool)
            limits = np.quantile(target[[2, 1, 0]][:, mask], [.01, .99]).tolist() if m == 'hls' else sample['display_limits']
            if limits is not None and limits[1]-limits[0] < 1e-6:
                limits = [limits[0]-.05, limits[1]+.05]
            fig, axes = plt.subplots(4, 6, figsize=(16, 12.4), layout='constrained')
            for ci, condition in enumerate(('clean', 'noisy10')):
                values = [target, physical(arrays[condition+'_image'][i])] + [physical(predictions[key][condition+'_prediction'][i]) for key in MODELS]
                supports = [mask, arrays[condition+'_input_mask'][i, 0].astype(bool)] + [mask]*4
                for row_offset in (0, 1):
                    row = ci*2 + row_offset
                    zoom = crop if row_offset else None
                    for col, (value, support) in enumerate(zip(values, supports)):
                        ax = axes[row, col]
                        im = show(ax, value, support, limits, zoom)
                        if row == 0:
                            ax.set_title(LABELS[col], fontsize=11)
                        if col == 0:
                            ax.set_ylabel(('干净输入' if ci == 0 else '10% 缺失输入') + ('\n局部 64×64' if zoom else '\n完整 256×256'), fontsize=10)
                            if not zoom:
                                ax.add_patch(Rectangle((crop['x']-.5, crop['y']-.5), 64, 64, fill=False, edgecolor='white', linewidth=1.7))
                                ax.add_patch(Rectangle((crop['x']+.5, crop['y']+.5), 62, 62, fill=False, edgecolor='#202020', linewidth=.7))
                        if col >= 2:
                            k, training = MODELS[col-2]
                            region = lookup[(f'{k}_{m}_{training}', condition, rid)]['regions']['all']
                            if zoom:
                                y, x = crop['y'], crop['x']
                                region = region_metrics(value[:, y:y+64, x:x+64], target[:, y:y+64, x:x+64], mask[y:y+64, x:x+64])
                            rmse = region['rmse']
                            ratio = region['gradient_amplitude_ratio']
                            detail = f'梯度幅值比 {ratio:.1%}' if ratio is not None else '梯度幅值比不可用'
                            ax.set_xlabel(f'RMSE {rmse:.4f} {unit}\n{detail}', fontsize=8)
            if m == 'lst':
                fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.65, aspect=45, extend='both',
                             label='地表温度（°C）；本页所有列与放大图共用参考图 1%-99% 色标')
            coverage = float(arrays['noisy10_actual_coverage'][i])
            fig.suptitle(f"{m.upper()} · 样本 {i+1:02d}/16 · {sample['sample_id']} · 瓦片 {sample['tile_id']}\n"
                         f"{sample['datetime']} · 参考有效 {sample['valid_fraction']:.1%} · 本次新增缺失 {coverage:.1%}", fontsize=12)
            note = (f'HLS 仅显示 RGB；六波段计算 RMSE。各列共用参考 RGB 的 1%-99% 范围 [{limits[0]:.3f}, {limits[1]:.3f}]，伽马 1/2.2。' if m == 'hls' else 'LST 存储值为摄氏度×100，读取后除以100；色标外数值仅在显示时截断。')
            note += '\n灰色表示无效/缺失；框为参考图梯度最大的有效局部，所有模型共用；未做锐化或平滑。梯度幅值比不是细节正确率。'
            fig.supxlabel(note, fontsize=9)
            name = f'{m}_{i+1:02d}_{rid}.png'
            fig.savefig(images/name, dpi=160)
            plt.close(fig)
            provenance['pages'].append(dict(modality=m, sample_number=i+1, record_id=rid,
                                            image='images/'+name, display_limits=limits, sha256=sha256_file(images/name)))
            print(f'PLATE {m} {i+1}/16', flush=True)

        # Contact sheet selected only by target texture; all 16 complete plates remain available.
        indexes = sorted(range(16), key=lambda i:samples[i]['reference_gradient_rms'], reverse=True)[:4]
        for condition in ('clean', 'noisy10'):
            fig, axes = plt.subplots(4, 5, figsize=(13.4, 11.8), layout='constrained')
            for row, i in enumerate(indexes):
                sample = samples[i]
                crop = sample['crop']
                target = physical(arrays['target'][i])
                mask = arrays['reference_mask'][i, 0].astype(bool)
                limits = np.quantile(target[[2, 1, 0]][:, mask], [.01, .99]).tolist() if m == 'hls' else sample['display_limits']
                values = [target]+[physical(predictions[key][condition+'_prediction'][i]) for key in MODELS]
                for col, value in enumerate(values):
                    show(axes[row, col], value, mask, limits, crop)
                    if row == 0:
                        axes[row, col].set_title((['原始参考']+LABELS[2:])[col], fontsize=11)
                    if col == 0:
                        axes[row, col].set_ylabel(f"样本 {i+1:02d}\n{sample['tile_id']}", fontsize=10)
                    if col:
                        y, x = crop['y'], crop['x']
                        r = region_metrics(value[:, y:y+64, x:x+64], target[:, y:y+64, x:x+64], mask[y:y+64, x:x+64])
                        axes[row, col].set_xlabel(f"局部 RMSE {r['rmse']:.4f} {unit}", fontsize=8)
            fig.suptitle(f"{m.upper()} 局部细节 · {'干净输入' if condition=='clean' else '10% 缺失输入'}\n从16张中按参考图梯度选取前4张，未按预测优劣挑选；64×64 原始像素，最近邻放大", fontsize=12)
            fig.supxlabel('每行所有模型共用参考图 1%-99% 显示范围；HLS 另用伽马 1/2.2。完整输入掩膜及全图见对应图册页。', fontsize=9)
            name = f'{m}_detail_{condition}.png'
            fig.savefig(DEST/name, dpi=160)
            plt.close(fig)
            provenance['detail_contacts'].append(dict(modality=m, condition=condition, sample_numbers=[i+1 for i in indexes], image=name, sha256=sha256_file(DEST/name)))
    write(DEST/'figure_manifest.json', provenance)
    if not write_vq_report:
        print('FIGURES COMPLETE', flush=True)
        return
    lines = ['# VQ-VAE：16 对新增预测与细节对比', '',
             '使用已有 4096/1024 码本最终检查点，只做推理。16 个不同验证瓦片各取一条记录，共 16 对 HLS-LST；按固定哈希种子 20260913 选择瓦片与记录，未按预测好坏选样。原先展示的 prithvi:val:230660 已排除。', '',
             '每个模态含四个模型：4096/1024 × 干净/加噪训练；分别测试干净/10% 缺失输入，固定使用原全量验证的第 0 个视图。共生成 256 张模型预测，所有预测保留原始浮点数组。', '',
             '各页四行依次为：干净输入全图、干净输入局部、缺失输入全图、缺失输入局部。参考图上的方框是固定 64×64 放大区域，由参考图梯度及有效掩膜选择，所有模型使用同一裁剪。', '',
             '**“梯度幅值比”是预测与参考的相邻像素变化 RMS 之比。低于 1 说明空间变化减少，但它不是细节正确率；过大的值也可能来自伪纹理。不能单独据此认定加噪训练是模糊的原因。**', '',
             'HLS 图仅显示 RGB，而 RMSE 使用六个波段；RGB 统一按本页参考 RGB 的 1%-99% 分位范围拉伸并应用 1/2.2 伽马，使高反射率图像的细节可见。LST 单位为 °C，统一采用本页参考图的 1%-99% 色标。每页内所有模型和裁剪共用范围，不比较不同页面的颜色亮度。没有给图像添加锐化、平滑或插值细节，显示截断不影响指标。', '',
             '以下仅为这 16 张图的梯度幅值比中位数，不能当作全量估计：', '',
             '| 模态 | 码本 | 训练 | 干净输入 | 缺失输入 |', '|---|---:|---|---:|---:|']
    for m in ('hls', 'lst'):
        for size, training in MODELS:
            values = [np.median([r['regions']['all']['gradient_amplitude_ratio'] for r in results['rows']
                                if r['model']==f'{size}_{m}_{training}' and r['condition']==c]) for c in ('clean', 'noisy10')]
            lines.append(f"| {m.upper()} | {size} | {'干净' if training=='clean' else '加噪'} | {values[0]:.1%} | {values[1]:.1%} |")
    lines += ['', 'PDF 图册位于工作区 output/pdf；下表提供全部单页高清图。', '',
              '| 样本 | 记录 | 瓦片 | HLS | LST |', '|---:|---|---|---|---|']
    for s in results['samples']:
        i, rid = s['number'], s['record_id']
        lines.append(f"| {i} | {rid} | {s['tile_id']} | [查看](images/hls_{i:02d}_{rid}.png) | [查看](images/lst_{i:02d}_{rid}.png) |")
    lines += ['', '原始输入与预测：*_inputs.npz、*_predictions.npz；指标和检查点哈希：results.json；选样规则：selection.json；逐页来源：figure_manifest.json。', '',
              '256 份单图预测的 RMSE 已与原全量验证同一视图核对；检查点读取使用冻结代码身份检查。原模型、训练结果及预处理配置未修改。']
    (DEST/'README_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('FIGURES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
