"""Source-backed Chinese report and standalone scientific figures for the audit."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.paper1.codebook1024.audit_spatial_tokens import DEST
from experiments.paper1.codebook1024.more_predictions import read, write
from eo_data.core import sha256_file


def main():
    d = read(DEST/'results.json')
    assert d['status'] == 'complete' and len(d['models']) == 8
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    def ev(size, m, t, condition=None):
        return d['models'][f'{size}_{m}_{t}']['evaluations'][condition or ('clean' if t == 'clean' else 'noisy10')]
    matched = [(m, t) for m in ('hls', 'lst') for t in ('clean', 'noisy')]
    labels = [m.upper()+'\n'+('干净→干净' if t == 'clean' else '缺失→缺失') for m, t in matched]
    manifest = dict(results_sha256=sha256_file(DEST/'results.json'), figures=[])
    def save(fig, name, description):
        fig.savefig(DEST/name, dpi=180)
        plt.close(fig)
        manifest['figures'].append(dict(file=name, sha256=sha256_file(DEST/name), description=description))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout='constrained')
    data = [ev(1024, m, 'clean')['physical']['target']['between_fraction'] for m in ('hls', 'lst')]
    axes[0].bar(['HLS', 'LST'], data, color='#356aa0', label='图像均值之间')
    axes[0].bar(['HLS', 'LST'], 1-np.array(data), bottom=data, color='#e3a542', label='每幅图内部')
    for i, v in enumerate(data):
        axes[0].text(i, v/2, f'{v:.1%}', ha='center', va='center', color='white', fontsize=14)
        axes[0].text(i, v+(1-v)/2 if 1-v >= .04 else 1.025, f'{1-v:.1%}', ha='center', va='center', fontsize=10)
    axes[0].set_ylim(0, 1.06)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title('原始数据：总方差如何分配')
    axes[0].legend(loc='lower center')
    for size, dx, color in [(4096, -.18, '#8f9faa'), (1024, .18, '#216fa6')]:
        x = np.arange(4)+dx
        vals = [ev(size, m, t)['physical']['spatial_r2'] for m, t in matched]
        axes[1].bar(x, vals, .34, color=color, label=f'{size} 码字')
        lat = [ev(size, m, t)['latent_scales'][-1]['spatial_r2'] for m, t in matched]
        axes[2].bar(x, lat, .34, color=color, label=f'{size} 码字')
        for ax, values in [(axes[1], vals), (axes[2], lat)]:
            for xx, value in zip(x, values):
                ax.text(xx, value+.014, f'{value:.0%}', ha='center', fontsize=9)
    for ax in axes[1:]:
        ax.set_xticks(range(4), labels)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=.15)
    axes[1].set_title('像素空间：去均值后的重建 R²')
    axes[2].set_title('量化空间：去均值后的潜变量 R²')
    fig.suptitle('图间差异占主导，图内信息仍有较大重建损失', fontsize=15)
    fig.supxlabel('同一批 256 条验证记录、实现 0；每图等权。缺失为约 10% 已知掩码缺失。R² = 1 − 去均值误差 / 原始图内方差。', fontsize=10)
    save(fig, 'variance_and_spatial_fidelity.png', 'Original physical variance decomposition; pixel and latent spatial R2, matched input/training conditions.')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, (m, t) in zip(axes.ravel(), matched):
        for size, color in [(4096, '#929ba4'), (1024, '#246fa5')]:
            scales = ev(size, m, t)['token_scales']
            x = [s['scale'] for s in scales]
            ax.plot(x, [s['marginal_entropy_bits'] for s in scales], '-o', color=color, ms=3,
                    label=f'{size}：总体熵')
            ax.plot(x, [s['conditional_within_entropy_bits'] for s in scales], '--s', color=color, ms=3,
                    label=f'{size}：平均图内熵')
        ax.set_title(m.upper()+' · '+('干净训练 / 干净输入' if t == 'clean' else '加噪训练 / 缺失输入'))
        ax.set_xlabel('量化网格边长（逐尺度独立统计）')
        ax.set_ylabel('码字熵（bit）')
        ax.set_xticks([1, 4, 8, 13, 16])
        ax.set_ylim(0, 7)
        ax.grid(alpha=.15)
        ax.legend(fontsize=8, ncol=2, loc='upper right')
    fig.suptitle('最细尺度仍有图内 token 多样性', fontsize=15)
    fig.supxlabel('实线减虚线为单个 token 与图像身份的经验互信息，受有限网格正偏差影响；报告另给置换基线。1×1 无图内熵是结构必然。', fontsize=9)
    save(fig, 'token_entropy_by_scale.png', 'Categorical marginal and within-image entropy per scale; no numerical distances between code IDs.')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for ax, m in zip(axes, ('hls', 'lst')):
        for t, color in [('clean', '#246fa5'), ('noisy', '#cc8423')]:
            ls = ev(1024, m, t)['latent_scales']
            x = [v['scale'] for v in ls]
            title = '干净训练' if t == 'clean' else '加噪训练'
            ax.plot(x, [v['image_mean_r2'] for v in ls], '-o', color=color, label=title+'：图像均值')
            ax.plot(x, [v['spatial_r2'] for v in ls], '--s', color=color, label=title+'：图内残差')
        ax.set_title(m.upper()+' · 1024 码字')
        ax.set_xlabel('累加到的量化网格边长')
        ax.set_ylabel('量化潜变量相对编码器输出的 R²')
        ax.set_ylim(0, 1.04)
        ax.set_xticks([1, 4, 8, 13, 16])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.suptitle('累计量化更准确地保留图像均值，图内残差的保留较弱', fontsize=14)
    fig.supxlabel('干净训练以干净输入评估，加噪训练以缺失输入评估；参照为各自编码器的量化前特征，不是原始像素。', fontsize=9)
    save(fig, 'latent_fidelity_by_scale.png', 'Cumulative quantization mean and spatial fidelity; internal diagnostic, not a separately trained baseline.')

    fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
    for mode, dx, color, label in [('shuffle_finest', -.19, '#e2ac4a', '只打乱 16×16'),
                                  ('shuffle_all', .19, '#a1546c', '打乱所有非 1×1 尺度')]:
        ratios = []
        for m, t in matched:
            p = ev(1024, m, t)['spatial_shuffle_probes'][mode]['centered_mse']
            ratios.append(p['changed']/p['native'])
        ax.bar(np.arange(4)+dx, ratios, .36, color=color, label=label)
        for xx, value in zip(np.arange(4)+dx, ratios):
            ax.text(xx, value+.035, f'{value:.2f}×', ha='center')
    ax.axhline(1, color='#333333', ls='--', lw=1, label='原始 token 排列')
    ax.set_xticks(range(4), labels)
    ax.set_ylabel('打乱后的去均值 MSE / 原始去均值 MSE')
    ax.set_ylim(0, max(6, ax.get_ylim()[1]*1.12))
    ax.legend(loc='upper left', ncol=3, fontsize=9)
    ax.set_title('保持码字频率不变，打乱位置仍会破坏空间重建', fontsize=14)
    fig.supxlabel('1024 码本；32 个不同瓦片、每瓦片 1 条记录；有效 token 的频率严格不变。该操作也破坏跨尺度配合，不能单独归因于局部细节。', fontsize=9)
    save(fig, 'spatial_shuffle.png', 'Counterfactual spatial MSE ratios; exact per-image token histograms held fixed. Paired bootstrap intervals in report.')

    for m, t in matched:
        condition = 'clean' if t == 'clean' else 'noisy10'
        a = np.load(DEST/ev(1024, m, t)['arrays_file'])
        mean, std = a['mean'][0], a['std'][0]
        unit, band = (100, 0) if m == 'lst' else (1, 3)
        colors = np.random.default_rng(20260914).uniform(.12, .93, size=(1024, 3))
        fig, axes = plt.subplots(4, 5, figsize=(14, 11.5), layout='constrained')
        names = ['参考图去均值', '重建图去均值', '打乱 16×16 后去均值', '8×8 码字位置', '16×16 码字位置']
        for i, pos in enumerate(d['illustration_positions']):
            support = a['illustration_mask'][i].astype(bool)
            fields = [(a['illustration_'+name][i]*std+mean)[band]/unit for name in ('target', 'native', 'shuffle_finest')]
            fields = [f-f[support].mean() for f in fields]
            limit = max(float(np.quantile(np.abs(fields[0][support]), .98)), 1e-6)
            for j, field in enumerate(fields):
                cmap = plt.get_cmap('RdBu_r').copy()
                cmap.set_bad('#777777')
                im = axes[i, j].imshow(np.ma.masked_where(~support, field), cmap=cmap, vmin=-limit, vmax=limit, interpolation='nearest')
            fig.colorbar(im, ax=axes[i, :3], fraction=.016, pad=.015, label='°C' if m == 'lst' else 'NIR 反射率')
            for j, side in enumerate((8, 16), 3):
                offset = sum(int(s*s) for s in a['scales'] if s < side)
                ids = a['ids'][pos, offset:offset+side*side].reshape(side, side)
                valid = a['support'][pos, offset:offset+side*side].reshape(side, side)
                rgb = colors[ids].copy()
                rgb[~valid] = .47
                axes[i, j].imshow(rgb, interpolation='nearest')
                axes[i, j].set_xlabel(f'{len(np.unique(ids[valid]))} 种码字', fontsize=9)
            for j, ax in enumerate(axes[i]):
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(names[j], fontsize=10)
            axes[i, 0].set_ylabel(f"{d['tile_ids'][pos]}\n记录 {d['record_ids'][pos]}", fontsize=9)
        fig.suptitle(m.upper()+' · 1024 码本 · '+('干净训练 / 干净输入' if t == 'clean' else '加噪训练 / 缺失输入'), fontsize=15)
        fig.supxlabel('每行三幅残差图共用参考图色标；分别减去各自有效像素均值。码字颜色仅代表类别，无数值大小或距离含义；灰色为无效。\n展示预先选定的前 4 条探针记录，所有模型共用；HLS 这里只显示 NIR 波段，统计使用全部 6 波段。', fontsize=9)
        save(fig, f'token_maps_{m}_{t}.png', 'Preselected illustrations: centered physical reference, native and shuffled reconstruction, categorical 8x8 and 16x16 maps.')
        a.close()

    p = lambda x: f'{x:.1%}'
    lines = ['# 图间 / 图内差异与码字空间表达验证', '',
        '结论：你的观察部分成立。原始影像确实以图像均值之间的差异为主；当前卷积 VQ-VAE 也明显更准确地保留均值，图内空间变化损失较大。但“码本只能区分图片、不能区分图片内部”过于绝对：最细尺度仍有图内码字多样性，保留频率而打乱位置后重建显著变差。', '',
        '本次为冻结检查点的诊断，没有新训练。8 个模型（1024 / 4096 码本 × HLS / LST × 干净 / 加噪训练），每个评估干净和约 10% 已知掩码缺失两种输入。使用原有固定 256 条验证记录、噪声实现 0；不是全量 11,757 条验证集的新汇总，也不是独立测试集。', '',
        '## 1. patch 是否对齐', '',
        '| 项目 | 卷积 VQ-VAE | Prithvi 连续 AE |', '|---|---|---|',
        '| 输入裁片 | 256×256 | 256×256 |',
        '| 最细空间网格 | 16×16 | 16×16 |',
        '| 到达该网格的方式 | 4 次 stride 2 下采样，卷积感受野重叠并带注意力 | 16×16 像素、stride 16 的 patch embedding，再经 ViT |',
        '| 瓶颈 | 10 个残差量化尺度，共 680 个离散索引；码字维度 32 | 256 个空间 token 加 CLS，连续维度 768 |',
        '| 对齐范围 | 输入、样本、最细网格和名义步距对齐 | 编码机制、感受野与表示容量没有对齐 |', '',
        '不能把 680 个索引理解为 680 个互不重叠的原始图像 patch；不同尺度量化的是逐步剩余的潜变量残差。当前对比不等码率。这里只核实模型像素网格，未据此推断两个传感器的原生地面分辨率相同。', '',
        '## 2. 数据本身是否以图间差异为主', '',
        '逐波段计算每张图有效像素均值 μ，图间方差 B 为这些均值的方差；图内方差 W 为每张图减去自身均值后的方差，再对图片和波段等权平均。总方差严格等于 B+W。HLS 使用物理反射率，LST 为摄氏度；LST 仅按已保存的摄氏度×100 除以 100，没有重复 Landsat 换算。', '',
        '| 数据 | 图间方差占比 | 图内方差占比 | 训练标准化空间的图间占比 |', '|---|---:|---:|---:|']
    for m in ('hls', 'lst'):
        e = ev(1024, m, 'clean')
        b = e['physical']['target']['between_fraction']
        lines.append(f"| {m.upper()} | {p(b)} | {p(1-b)} | {p(e['normalized']['target']['between_fraction'])} |")
    lines += ['', '这个总体比例不能推导为每一张图内部都平坦，也不等于训练当前梯度主要由均值误差贡献。均值已经拟合好以后，剩余误差可以主要是空间误差。', '',
        '## 3. 重建保留了什么', '',
        'MSE 严格分解为均值误差与去均值空间误差。表中“空间 R²”=1−空间误差/原图图内方差；“空间方差比”是预测/原图的图内方差，只描述幅值，不能代替细节准确率。去均值残差仍包含大尺度空间渐变，并非纯高频纹理；LST 空间 R² 较高也不能否定边界或细小结构被平滑。', '',
        '| 码本 | 模态 | 训练→评估 | 均值 R² | 空间 R² | 空间方差比 | 总误差中空间部分 |', '|---|---|---|---:|---:|---:|---:|']
    for size in (1024, 4096):
        for m, t in matched:
            for condition in ('clean', 'noisy10'):
                e = ev(size, m, t, condition)['physical']
                label = ('干净' if t == 'clean' else '加噪')+'→'+('干净' if condition == 'clean' else '缺失')
                lines.append(f"| {size} | {m.upper()} | {label} | {p(e['image_mean_r2'])} | {p(e['spatial_r2'])} | {p(e['spatial_variance_ratio'])} | {p(e['centered_mse']/e['mse'])} |")
    lines += ['', '负 R² 表示该部分误差比相应方差基准还大；特别要区分加噪训练 LST 遇到干净输入的分布偏移，不能把它都归为码本空间表达问题。', '',
        '## 4. 码本是否只有图间区分能力', '',
        '以下是 16×16 最细尺度；每张图等权，排除参考有效比例低于 75% 的网格。唯一数量是每张图的中位数。图内有效码字数为 2 的平均图内熵次方，并非每张图实际唯一数。', '',
        '| 码本 | 模态 / 匹配输入 | 每图唯一码字中位数 | 最常用码字占比中位数 | 总体熵 | 图内熵 | 经验互信息 | 置换基线 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for size in (1024, 4096):
        for m, t in matched:
            s = ev(size, m, t)['token_scales'][-1]
            lines.append(f"| {size} | {m.upper()} / {'干净' if t == 'clean' else '缺失'} | {s['per_image_unique_median']:.0f} | {p(s['per_image_dominant_fraction_median'])} | {s['marginal_entropy_bits']:.3f} | {s['conditional_within_entropy_bits']:.3f} | {s['image_identity_mutual_information_bits']:.3f} | {s['permutation_null_mi_mean']:.3f} |")
    lines += ['', '熵和互信息的单位为 bit。互信息衡量随机取一个 token 的身份与图片身份的关系，不能当作整张图 token 直方图的分类准确率。有限 token 网格会带来经验互信息正偏差，因此给出跨图片随机置换 32 次的基线。1×1 的图内熵恒为 0，不能据此判定坍缩。码字 ID 只是类别编号，不能用编号的方差或距离衡量空间变化。', '',
        '## 5. 保留频率、打乱位置的反事实验证', '',
        '从固定子集中按预先确定的哈希选出 32 个不同瓦片，每瓦片 1 条记录。仅在每幅图、每个尺度的有效位置内置换，逐图逐尺度检查排序后的 token 完全一致。以下使用 1024 码本、训练和评估条件匹配。', '',
        '| 模态 / 输入 | 打乱方式 | 原始空间 MSE | 打乱后空间 MSE | 误差倍数 | 配对差值 95% 区间 |', '|---|---|---:|---:|---:|---|']
    for m, t in matched:
        for mode, title in [('shuffle_finest', '仅 16×16'), ('shuffle_all', '全部空间尺度')]:
            r = ev(1024, m, t)['spatial_shuffle_probes'][mode]['centered_mse']
            ci = r['delta_ci95']
            lines.append(f"| {m.upper()} / {'干净' if t == 'clean' else '缺失'} | {title} | {r['native']:.6g} | {r['changed']:.6g} | {r['changed']/r['native']:.2f}× | [{ci[0]:.6g}, {ci[1]:.6g}] |")
    lines += ['', 'HLS MSE 单位为反射率²，LST 为 °C²。置信区间来自 2,000 次配对瓦片 bootstrap。置换会同时破坏空间排列和跨尺度配合，并可能产生训练分布外的组合；它能否定“解码器只用直方图、完全忽略位置”，但不能精确给出模型编码了多少真实细节。一次置换和 32 个瓦片也不代表所有空间扰动。', '',
        '## 6. 量化前后定位', '',
        '对同一编码器输出 z 与累计量化结果分别做均值 / 图内残差分解，避免只看 z 的总能量。以下为最后一个尺度累加完成后的结果。', '',
        '| 码本 | 模态 / 匹配输入 | 潜变量均值 R² | 潜变量图内残差 R² |', '|---|---|---:|---:|']
    for size in (1024, 4096):
        for m, t in matched:
            r = ev(size, m, t)['latent_scales'][-1]
            lines.append(f"| {size} | {m.upper()} / {'干净' if t == 'clean' else '缺失'} | {p(r['image_mean_r2'])} | {p(r['spatial_r2'])} |")
    lines += ['', '这定位到量化后对图内残差的逼近较弱，但不能单凭该结果证明原图细节只在量化器丢失：编码器本身可能已丢失信息，解码器还会重新变换潜变量。Prithvi 的连续瓶颈容量不同，也不能用于单独归因 VQ。', '',
        '## 7. 下一步训练建议', '',
        '优先做小范围的损失配比消融，保持当前 1024 码本、256×256 输入、16×16 最细网格、数据划分和训练预算不变。保留绝对辐射值重建目标，在现有像素 MSE 与权重 0.1 的梯度损失上，增加明确的逐图去均值空间残差项。可先比较额外权重 0 / 1 / 3；这些权重是待验证的实验设计，并非已证实的最优值。', '',
        '具体可写为 L = MSE + λ·MSE((预测−预测有效均值)−(参考−参考有效均值)) + 0.1·梯度损失 + VQ 损失。去均值只用于损失计算，不能把干净参考的均值或标准差额外送进编码器 / 解码器，也不改变全数据集统一的训练集归一化。均值已拟合很好、当前剩余误差主要是空间误差时，这个消融更直接检验的是空间重建相对 VQ 损失的权重，而不是假定均值误差仍主导训练。', '',
        '同时记录每个尺度的空间残差 R²、图内熵、图像均值误差和梯度误差，按原图内部方差分组检查。若给低对比度图提高权重，应仅用训练集确定方差下限并限制最大权重，避免把近常量图或噪声放大。LST 加噪模型在干净输入上的退化应另做干净 / 缺失混合训练消融，不与码本结构修改混在一组。', '',
        '若损失消融仍不能提高图内残差保真，再单独比较粗尺度与细尺度的 VQ 损失配重、分离粗细码本或更高空间分辨率瓶颈。当前模型已经是多尺度残差量化，不需要把“加入多尺度”当作尚未使用的方法。改变网格或码本划分时必须同时报告总索引数、码率与参数量。仅继续缩小码本或强制全局均匀使用码字，不能保证真实空间细节改善。', '',
        '相关方法背景：[VQ-VAE-2](https://arxiv.org/abs/1906.00446) 讨论分层多尺度离散表示；[Huh 等，ICML 2023](https://proceedings.mlr.press/v202/huh23a.html) 分析编码器特征与码向量分布不匹配带来的量化优化困难。以上训练顺序主要依据本次实验诊断，是待检验假设，不是这些论文对本数据的结论。', '',
        '## 8. 校验与可复现文件', '',
        f"- {len(set(d['tile_ids']))} 个验证瓦片中的 256 条记录；单个训练随机种子 17，每个检查点 10,813 步。",
        f"- 16 个模型 / 评估条件的重建与原全量评估逐记录结果一致，最大 RMSE 相对差 {max(e['saved_full_evaluation_max_relative_rmse_difference'] for v in d['models'].values() for e in v['evaluations'].values()):.3g}。",
        '- 干净输入的逐尺度码字计数与原审计严格一致（原审计有两份相同干净视图，本次取一份）。',
        '- 冻结模型 / 数据源哈希一致；统计函数的 4 项数学校验通过；位置打乱前后逐尺度直方图完全相同。',
        '- results.json 保存所有模型、两种输入的全部统计；16 个 NPZ 保存逐图 token、有效支持、误差分解、探针与示例。selection.json 固定记录与探针选择；figure_manifest.json 记录图像哈希。',
        '- 脚本：experiments/paper1/codebook1024/audit_spatial_tokens.py、spatial_token_stats.py、report_spatial_tokens.py。训练、数据预处理和检查点没有更改。', '',
        '## 图像', '']
    for f in manifest['figures']:
        lines += [f"![{f['file']}]({f['file']})", '']
    (DEST/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(DEST/'figure_manifest.json', manifest)
    print('REPORT AND FIGURES COMPLETE')


if __name__ == '__main__':
    main()
