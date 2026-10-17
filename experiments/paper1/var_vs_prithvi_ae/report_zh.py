"""Chinese presentation of existing A02 results, without rerunning inference."""
from pathlib import Path
import json
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def model(backbone):
    return '卷积 VQ-VAE' if backbone=='conv' else 'Prithvi 连续 AE'


def training(value):
    return '干净训练' if value=='clean' else '加噪训练'


def number(value,modality):
    return f'{value:.5f}' if modality=='hls' else f'{value:.3f}'


def write_report(report,phase,records,coverage,curves):
    if phase!='full':raise ValueError('中文版报告目前面向已完成的 A02 全量实验')
    output=report.parent
    audit=read(report/'completion_audit.json')
    assert audit['status']=='passed' and len(records)==16
    lookup={(r['modality'],r['backbone'],r['training'],r['evaluation']):r for r in records}
    get=lambda m,b,n,e:lookup[(m,b,n,e)]
    lines=['# A02 中文实验报告：卷积 VQ-VAE 与 Prithvi 连续自编码器', '',
        '**实验已完成：HLS／LST × 两种模型 × 干净／加噪训练，共 8 组，全部从零开始。**', '',
        '## 1. 模型名称与实际训练内容', '',
        '**卷积版确实是 VQ-VAE。** 此前表格中的“VAR”是其来源仓库 FoundationVision/VAR 的简称，英文全称为 Visual Autoregressive Modeling（视觉自回归建模）。本轮使用该仓库的 VQ-VAE 编码、量化与解码模块，没有训练其中的自回归图像生成模型。后续模型名称统一为“卷积 VQ-VAE”和“Prithvi 连续 AE”。', '',
        '| 模型 | 本轮实际网络结构 | HLS 参数量 | LST 参数量 |',
        '|---|---|---:|---:|']
    for b in ('conv','prithvi'):
        params={m:read(output/f'full_{m}_{b}_clean_seed17'/'model.json')['parameters'] for m in ('hls','lst')}
        structure=('ResNet 风格卷积编码器 → 多尺度残差向量量化 → 卷积解码器' if b=='conv' else
                   '空间 ViT 编码器 → 原生连续特征 → ViT 解码器；无时间输入、无 VQ、无 KL')
        lines.append(f"| {model(b)} | {structure} | {params['hls']:,} | {params['lst']:,} |")
    lines+=['',
        '卷积 VQ-VAE 保留来源代码的多尺度离散瓶颈：4096 个码字、每幅图像 680 个索引。Prithvi 连续 AE 使用 256 个图像块向量和 1 个 CLS 向量，特征维度为 768。Prithvi 此处是确定性自编码器（AE），没有高斯采样或 KL 正则。', '',
        '两者参数量接近，但潜在表示容量和压缩率没有对齐。因此，这轮比较同时改变了网络骨干和表示类型，不能单独证明 VQ 的作用。卷积版采用来源仓库的网络结构，训练目标按本实验统一设置，并非复现原论文完整的分词器训练方案。', '',
        '## 2. 数据与实验设置', '',
        '| 项目 | 设置 |', '|---|---|',
        '| 数据 | 仅使用 prithvi_data 的一对一 HLS–LST 记录；HLS、LST 分别训练，未与 LSTSR 合并 |',
        '| 训练／验证 | 230,659 条训练记录／11,757 条验证记录；训练与验证记录不重叠 |',
        '| 预处理 | 所有模型共用冻结的有效像素掩膜及各模态训练集归一化统计 |',
        '| 物理单位 | HLS 为反射率；LST 已存为“摄氏度 × 100”，反归一化后只除以 100 |',
        '| 加噪方式 | 使用预先固定的掩膜形状，人工遮挡约 10% 的有效区域，并提供输入掩膜 |',
        '| 重建目标 | 同一幅干净图像的有效参考像素；加噪训练不混入额外的干净输入条件 |',
        '| 训练预算 | 每组 10,813 次优化器更新；完整遍历训练集 3 次，再读取 40 条，共 692,017 次样本呈现 |',
        '| 批量／优化器 | 微批量 8，梯度累积 8；有效批量通常为 64，跨轮次边界允许较小批量；AdamW，初始目标学习率 1e-4 |',
        '| 损失 | 归一化重建 MSE + 0.1 × 梯度 MSE；卷积 VQ-VAE 额外使用原生 VQ 损失 |',
        '| Prithvi 瓶颈 | 连续特征；VQ、KL、跨视图和稳定性损失均为 0 |',
        '| 初始化／随机种子 | 随机初始化，无预训练权重；本轮训练种子为 17 |',
        '| 最终验证 | 使用每组第 10,813 步的同一最终检查点，分别验证干净输入和 10% 缺失输入，每条记录 2 个固定视图 |', '',
        '这里的“加噪”具体指已知掩膜下的缺失区域恢复，当前结果不覆盖高斯噪声、条带噪声等其他污染。RMSE 是先在样本内平均两个视图的均方误差，再对样本等权平均并开平方；不是逐图 RMSE 的简单平均。95% 置信区间按空间瓦片分组重采样。', '',
        '## 3. 整体重建结果', '',
        'RMSE 越低越好。HLS 数值单位为反射率，LST 数值单位为摄氏度（°C）。所有格子均来自全部 11,757 条验证记录。', '',
        '| 数据 | 模型 | 训练输入 | 干净输入 RMSE | 10% 缺失输入 RMSE |',
        '|---|---|---|---:|---:|']
    for m in ('hls','lst'):
        for b in ('conv','prithvi'):
            for n in ('clean','noisy'):
                clean=get(m,b,n,'clean');noisy=get(m,b,n,'noisy10')
                lines.append(f"| {m.upper()} | {model(b)} | {training(n)} | {number(clean['rmse'],m)} | {number(noisy['rmse'],m)} |")
    improvements={m:(1-get(m,'prithvi','noisy','noisy10')['rmse']/get(m,'prithvi','clean','noisy10')['rmse'])*100 for m in ('hls','lst')}
    lines+=['',
        f"Prithvi 连续 AE 经加噪训练后，缺失输入的整体 RMSE 在 HLS 上下降 {improvements['hls']:.1f}%，在 LST 上下降 {improvements['lst']:.1f}%；干净输入的误差有所上升。这些百分比是本轮单个训练种子的观测结果。", '',
        '## 4. 单独看缺失区域与可见区域', '',
        '整体指标同时包含约 10% 缺失区域和约 90% 可见区域。判断补全能力时，需要单独查看被遮挡部分。以下均为 10% 缺失输入验证。', '',
        '| 数据 | 模型 | 训练输入 | 缺失区域 RMSE | 可见区域 RMSE |',
        '|---|---|---|---:|---:|']
    for row in records:
        if row['evaluation']!='noisy10':continue
        m=row['modality']
        lines.append(f"| {m.upper()} | {model(row['backbone'])} | {training(row['training'])} | {number(row['corrupt_rmse'],m)} | {number(row['retained_rmse'],m)} |")
    lines+=['',
        '两种模型经过加噪训练后，缺失区域的重建误差很接近：', '',
        f"- HLS：卷积 VQ-VAE 为 {get('hls','conv','noisy','noisy10')['corrupt_rmse']:.5f}，Prithvi 连续 AE 为 {get('hls','prithvi','noisy','noisy10')['corrupt_rmse']:.5f}。",
        f"- LST：卷积 VQ-VAE 为 {get('lst','conv','noisy','noisy10')['corrupt_rmse']:.3f}°C，Prithvi 连续 AE 为 {get('lst','prithvi','noisy','noisy10')['corrupt_rmse']:.3f}°C。", '',
        'Prithvi 的整体误差优势主要来自可见区域的重建。对于上述缺失区域的小幅差异，本报告不作统计显著性判断。', '',
        '## 5. 需要注意的退化', '',
        f"**LST 卷积 VQ-VAE 在仅用缺失输入训练后，对干净输入的适应性明显变差。** 其干净输入 RMSE 为 {get('lst','conv','noisy','clean')['rmse']:.3f}°C，干净训练版本为 {get('lst','conv','clean','clean')['rmse']:.3f}°C；而缺失输入 RMSE 为 {get('lst','conv','noisy','noisy10')['rmse']:.3f}°C。", '',
        '这表明模型对输入是否含缺失区域较敏感。数据、最终检查点及逐样本指标已核对，但具体成因尚未隔离验证，不能直接断言是量化模块、掩膜通道或某一种归一化造成的。', '',
        '## 6. 是否欠拟合、是否已经收敛', '',
        '**仍有细节欠拟合迹象，不能认为已经充分收敛。** 固定验证样例中，卷积 VQ-VAE 有较强的平滑现象，Prithvi 连续 AE 仍存在人工细碎纹理。细节重建不足可以来自信息瓶颈或优化不充分，目前不能分离两者，也不能保证仅增加训练轮数就能解决。', '',
        '| 数据 | 模型 | 训练输入 | 第 8814–9813 步平均重建 MSE | 第 9814–10813 步平均重建 MSE | 下降幅度 |',
        '|---|---|---|---:|---:|---:|']
    for name,curve in curves.items():
        _,m,b,n,_=name.split('_')
        values=np.asarray(curve)[:,1]
        before=float(values[-2000:-1000].mean());after=float(values[-1000:].mean())
        lines.append(f'| {m.upper()} | {model(b)} | {training(n)} | {before:.6f} | {after:.6f} | {(1-after/before)*100:.2f}% |')
    lines+=['',
        '两个训练窗口包含不同样本，上表用于描述趋势。固定 256 条记录的阶段性诊断验证显示，Prithvi 在末期仍有改善；LST 卷积 VQ-VAE 更接近平台，末期缺失输入误差略有回升。因此，不能把所有问题都归结为训练时间不够。最终比较始终使用预先约定的最后一步检查点，没有按各模型最佳分数重新挑选。', '',
        '## 7. 重建图与训练曲线', '',
        '以下展示首条固定验证记录 `prithvi:val:230660`，用于观察重建现象，不代表整个数据集。所有模型共用同一参考图和同一条件的输入。图中的显示截断不影响指标计算。', '',
        '### HLS 重建', '',
        '展示红、绿、蓝三个波段，统一采用反射率 ÷ 0.3 及伽马 1/2.2；灰色为无效或缺失像素。', '',
        '![HLS 重建对比](hls_reconstruction_zh.png)', '',
        '### LST 重建', '',
        '温度反归一化后除以 100。所有格子统一使用参考图 1%–99% 分位数范围的色标，因此超出该范围的重建值会饱和显示。', '',
        '![LST 重建对比](lst_reconstruction_zh.png)', '',
        '### 训练曲线', '',
        '仅绘制归一化重建 MSE，最近 100 步滑动平均，纵轴使用对数刻度；不将卷积版独有的 VQ 损失混入比较。', '',
        '![重建损失曲线](reconstruction_loss_zh.png)', '',
        '## 8. 验证范围与局限', '',
        f"- 8 组均通过完整性核查，每组覆盖全部训练记录，样本顺序一致；16 份最终验证均覆盖全部验证记录，每条记录有 2 个视图。",
        '- 核查了源代码、配置、数据指纹及检查点哈希，并从保存的逐样本误差复算各区域 RMSE。',
        '- 本轮只有训练种子 17，且没有独立测试集；置信区间反映验证空间瓦片的不确定性，不包含重复训练的随机性。',
        '- 两个模型同时改变了骨干与离散／连续瓶颈，压缩率不相同，不能单独归因于是否使用 VQ。',
        f"- 使用一张 RTX 5090；全局显存采样峰值为 {audit['global_gpu_peak_gib']:.2f} GiB，最小采样空闲显存为 {audit['minimum_sampled_device_free_gib']:.2f} GiB。卷积与 Prithvi 使用两个受显存限制的队列。",
        '- 中途一次调度器状态文件访问错误已修复，LST 卷积干净组从第 1887 步检查点恢复；恢复后的样本顺序连续性已核验。', '',
        '## 附录：完整验证统计', '',
        'HLS 单位为反射率，LST 单位为 °C。所有行均为 11,757 条验证记录。', '',
        '| 数据 | 模型 | 训练输入 | 验证输入 | RMSE | 95% 瓦片重采样区间 | MAE |',
        '|---|---|---|---|---:|---|---:|']
    for row in records:
        m=row['modality'];ci=row['rmse_tile_ci95'];condition='干净' if row['evaluation']=='clean' else '10% 缺失'
        lines.append(f"| {m.upper()} | {model(row['backbone'])} | {training(row['training'])} | {condition} | {number(row['rmse'],m)} | [{number(ci[0],m)}, {number(ci[1],m)}] | {number(row['mae'],m)} |")
    lines+=['', '## 数据与代码来源', '',
        '- [机器可读结果](results.json) · [完成核查记录](completion_audit.json) · [中文报告核查记录](visual_review_zh.json)。',
        '- [卷积 VQ-VAE 来源：FoundationVision/VAR，固定版本](https://github.com/FoundationVision/VAR/tree/78b95394fc5896192e3a003e4b295f8ea743c48f)。',
        '- [Prithvi EO 2.0 架构来源，固定版本](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/tree/9eb1b1102806593963daa333bcc491b1c6f8562f)。',
        '- 实验内部目录仍保留 `var_vs_prithvi_ae` 历史名称，用于匹配已保存的检查点与核查记录；它不表示本轮训练了自回归模型。', '']
    if (report/'CODEBOOK_ZH.md').exists():
        lines+=['## 补充：码本使用情况', '',
            '后续核查发现，卷积 VQ-VAE 的码字选择高度集中，存在严重利用不足和部分索引坍缩迹象，LST 尤其明显。全量验证的有效码字数约为 7–15；固定 256 条记录上，LST 在训练对应条件下，最常用的 10 个码字占索引的 99% 以上。', '',
            '这些结果不表示只有 7–15 个编号出现过，也不能单独证明训练过程中码字集合持续缩小。完整定义、统计口径、分尺度结果及核查限制见 [中文码本核查报告](CODEBOOK_ZH.md)。', '']
    (report/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
