"""Reproducible Chinese report from completed, verified statistics."""
from pathlib import Path
import json,hashlib
from color_analyze import OUT,ROOT

def main():
    result=OUT/'results';d=json.loads((result/'summary.json').read_text(encoding='utf-8'))
    assert d['status']=='complete'
    a=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    s=d['statistics'];pct=lambda x:f'{x*100:.2f}%'
    old=json.loads((ROOT/'runs/openimages_hlp_color_20260917/results/summary.json').read_text(encoding='utf-8'))['statistics']
    stability='\n'.join(f"| {label} | {old[k]['within_variance_share']*100:.2f}% | {s[k]['within_variance_share']*100:.2f}% | {(s[k]['within_variance_share']-old[k]['within_variance_share'])*100:+.2f} 个百分点 |" for k,label in [('oi_encoded','编码 RGB'),('oi_linear','按 sRGB 线性化')])
    names={'oi_encoded':'OpenImages 编码 RGB','oi_linear':'OpenImages 按 sRGB 线性化','hlp_rgb':'HLP 可见光 RGB 反射率','hlp_six':'HLP 六波段反射率','hlp_lst':'HLP LST（°C）'}
    table='\n'.join(f"| {names[k]} | {v['n']:,} | {pct(v['within_variance_share'])} | {pct(v['between_variance_share'])} | {v['spatial_rms_quantiles']['0.5']:.4f} |" for k,v in s.items())
    pca='\n'.join(f"| {names[k]} | {pct(s[k]['pca_total']['pc1_share'])} | {pct(s[k]['pca_within']['pc1_share'])} | {s[k]['pca_within']['effective_rank']:.3f} |" for k in ['oi_encoded','oi_linear','hlp_rgb','hlp_six'])
    channels=['Blue','Green','Red','NIR','SWIR1','SWIR2'];h=s['hlp_six']
    bands='\n'.join(f"| {name} | {h['mean'][i]:.4f} | {h['total_std'][i]:.4f} | {h['pooled_within_std'][i]:.4f} | {pct(h['within_share_per_channel'][i])} | {pct(h['boundary_one_fraction'][i])} |" for i,name in enumerate(channels))
    sensitivities='\n'.join(f"| 平均 RGB < {threshold} | {d['bright_image_sensitivity']['mean_rgb_below_'+str(threshold)]['n']:,} | {d['bright_image_sensitivity']['mean_rgb_below_'+str(threshold)]['excluded_images']:,} | {pct(d['bright_image_sensitivity']['mean_rgb_below_'+str(threshold)]['within_variance_share'])} |" for threshold in [.5,.8])
    hier='\n'.join(f"| {names[k]} | {pct(s[k]['tile_decomposition']['pixel_within_image_share'])} | {pct(s[k]['tile_decomposition']['image_mean_within_tile_share'])} | {pct(s[k]['tile_decomposition']['tile_mean_share'])} |" for k in ['hlp_rgb','hlp_six','hlp_lst'])
    histinfo='\n'.join(f"- {names[k]}：颜色比例图排除的近黑抽样像素为 {pct(s[k]['rg_excluded_sample_pixel_fraction'])}，保留至少一个非黑抽样像素的图片 {s[k]['rg_nonblack_images']:,} 张。" for k in ['oi_encoded','oi_linear','hlp_rgb'])
    text=f'''# 同规模 OpenImages 与 HLP 的颜色及空间变化比较

统计日期：2026-09-17。只分析数据，不训练模型，不修改冻结数据、划分或归一化。

## 主要发现

OpenImages 编码 RGB 的总方差中，**{pct(s['oi_encoded']['within_variance_share'])} 来自图片内部变化**；HLP 可见光 RGB 对应为 **{pct(s['hlp_rgb']['within_variance_share'])}**，其余来自不同图片的均值差异。LST 图内变化占 **{pct(s['hlp_lst']['within_variance_share'])}**。这直接检验了此前“图内差异小、图间差异大”的数据分布观察。

自然照片按 sRGB 曲线线性化后，图内比例为 **{pct(s['oi_linear']['within_variance_share'])}**。因此，这个现象不能仅用照片的非线性编码来解释。不过 HLP 是地表反射率，照片包含相机曝光、白平衡、色调处理和混合 ICC 配置；即使线性化，也不构成同一物理颜色空间。

HLP 的 RGB 变化更集中在少数方向：整体协方差的第一主成分解释 **{pct(s['hlp_rgb']['pca_total']['pc1_share'])}**，自然照片为 **{pct(s['oi_encoded']['pca_total']['pc1_share'])}**。去除每幅图的各通道均值后，HLP RGB 第一主成分仍解释 **{pct(s['hlp_rgb']['pca_within']['pc1_share'])}**。这支持“可见光通道倾向共同变化”，但不能把主成分直接等同于真实照度或地物类别。

模型的 HLS 输入有六个波段。六波段图内变化比例为 **{pct(s['hlp_six']['within_variance_share'])}**；按当前 HLP v2 规定的训练集逐波段 z-score 后为 **{pct(d['hlp_six_training_zscore']['within_variance_share'])}**。红外波段包含 RGB 图不能表达的差异，不能依据可见光结果判定六波段信息贫乏。本次是当前 v2 train 数据分析，不是旧模型的 token 归因分析。

![方差比较](results/variance_comparison.png)

## 扩样前后稳定性

自然图像从原100,000张扩展至218,975张，与HLP v2 train的记录数一致。沿用seed=20260916及相同CVDF训练池的SHA256排序；旧100,000张是新清单的完整前缀，补充118,975张，没有根据颜色、类别、质量或重建结果挑选。旧子集及旧报告保留不变。

| 自然图像坐标 | 原100,000张图内方差占比 | 新218,975张图内方差占比 | 变化 |
|---|---:|---:|---:|
{stability}

百分点变化由未四舍五入的数值计算，因此可能与表中已保留两位小数的百分比直接相减略有差别。

两次样本集合嵌套，上表用于描述扩样稳定性，不是两次独立重复实验。等规模只对齐记录数，并未对齐独立场景数、物理视野或有效像素数；HLP含同tile的多条记录，自然图像也未做视觉近重复排查。此次没有训练模型；本报告仍是数据分布对照。

## 数据范围与可比性

- 自然图像：已下载的 OpenImages train218975，218,975 张全部纳入。它由 CVDF 的 1,743,042 张 boxable train 池按固定哈希随机抽取，不代表 OpenImages 全部约 9M 图像，也不是原来的 200 张 validation。原下载清单和 SHA256 保存在 `data/openimages_train218975_20260917`。
- HLP：`hlp-tile-validation-v2-20260914` 的 **218,975 对 train、2,796 个 tile**，全部纳入；没有使用 val/test 调参或改变成员。
- 自然图像保持此前 VAR 重建实验入口：PIL 默认帧 → RGB → 短边 288 Lanczos → 中心裁剪 256×256 → 除以 255。未应用 EXIF 旋转、逐图自动对比度或 ICC 转换。MPO 仍取默认首帧。
- HLS：原生 256×256；顺序 Blue、Green、Red、NIR、SWIR1、SWIR2；RGB 取 `[2,1,0]`，恢复到反射率。使用冻结读取器的有效 mask，填充值不参与统计。
- LST：同一对记录，按保存值 /100 转为 °C；有效 mask 与 HLS 分开。温度没有“颜色空间”，本报告只比较其空间方差，未把伪彩色当数据。
- 两者都是 256×256 输入，但自然照片视野、对象尺度与卫星 30m 网格不同；这里没有把物理空间尺度对齐。
- 全部矩、协方差、边界比例使用全部有效像素。各图像等权，与 HLP 训练归一化的统计权重一致，不让有效像素略多的图获得更大权重。

HLS 的波段与反射率性质可查 [Earth Engine 官方目录](https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSL30_v002)，本地保存尺度及上游裁剪的证据见 [HLP 来源核验](../../data_inspection/HLP_SOURCE_PROVENANCE.md)。sRGB 线性化公式采用 [W3C CSS Color 4](https://www.w3.org/TR/css-color-4/)。它只用于编码敏感性对照，未将 HLS 宣称为标准 sRGB、CIE Lab 或经过色度标定的 XYZ。

## 方差分解：图内与图间

对第 i 张图，μᵢ 是有效像素的逐通道均值，Σᵢ 是像素协方差，所有方差均采用总体定义（ddof=0）：

`Σ_total = mean_i(Σ_i) + Cov_i(μ_i)`

前项为图内空间变化，后项为图间均值变化。比例用协方差的迹（各通道方差之和）计算。此分解在“先等概率选图片，再在该图有效像素中均匀选像素”的分布上严格成立。

| 数据 / 坐标 | 图片数 | 图内方差占比 | 图间方差占比 | 每图空间 RMS 中位数 |
|---|---:|---:|---:|---:|
{table}

空间 RMS 定义为 `sqrt(mean_channels(Var_valid_pixels(channel)))`。RGB/反射率范围以 0–1 表示，LST 为 °C。不同数据的绝对 RMS **不是同一物理量**，不能把二者相除直接解释为“信息量差多少倍”。比例对所有通道共同的固定线性增益和固定偏置不变，但对非线性变换或不同的逐通道增益不保证不变。

方差图 B 采用无量纲的相对 RMS：先将每个通道除以其在各自数据集中的全局标准差，再计算图内 RMS。中位数分别为自然编码 RGB **{s['oi_encoded']['relative_spatial_rms_quantiles']['0.5']:.3f}**、自然线性化 RGB **{s['oi_linear']['relative_spatial_rms_quantiles']['0.5']:.3f}**、HLP RGB **{s['hlp_rgb']['relative_spatial_rms_quantiles']['0.5']:.3f}**。这是每幅图的局部变化相对于自身数据集总体变化的大小，不是颜色的物理等价变换，也未对每张图单独做标准差归一化。

进一步在 HLP 中按 tile 分解图间项；每个 tile 的权重仍是其记录数占比，没有改为 tile 等权：

| HLP 模态 | 图内像素变化 | 同 tile 不同图片均值 | 不同 tile 均值 |
|---|---:|---:|---:|
{hier}

“同 tile”中既有不同日期，也有不同位置，不能把这一项全部称为时间变化。图间差异也不能简单等同于模型是否认出了地理位置。

## 颜色方向与通道相关性

![RGB 通道比例](results/rgb_ratio_density.png)

图中 r=R/(R+G+B)，g=G/(R+G+B) 是**通道比例**，不是 CIE xy。R=G=B 对应 (1/3,1/3)。每张图固定种子抽 64 个有效像素，仅用于绘制密度图；排除 R+G+B≤0.01 后，在每张含非黑样本的图片内重新归一化权重。三图采用同一 64×64 网格、坐标和对数色标。

颜色分布集中在某些方向，不等于更接近灰色或感知饱和度更低。这里通道比例到 (1/3,1/3) 的平均距离为自然编码 RGB {s['oi_encoded']['rg_distance_from_gray_mean']:.3f}、自然线性化 RGB {s['oi_linear']['rg_distance_from_gray_mean']:.3f}、HLP RGB {s['hlp_rgb']['rg_distance_from_gray_mean']:.3f}；结果受坐标编码影响。因此本报告的“集中”主要指方差方向及图内／图间分配，不把它改写成“HLP 全都灰、没有颜色”。

{histinfo}

![通道相关性](results/channel_correlations.png)

上排包含图间均值变化；下排先去掉每图各通道均值，保留图内残差。这避免把“所有图片总体一起亮暗变化”混同于“每张图内部的颜色丰富程度”。

| 数据 | 整体 PC1 方差占比 | 图内残差 PC1 占比 | 图内协方差有效秩 |
|---|---:|---:|---:|
{pca}

有效秩为 `exp(-Σ p_k log p_k)`，p_k 为归一化特征值。RGB 的上限为 3，六波段为 6。这仅测量通道协方差集中程度，**不等于潜变量维数、类别数或应使用的 codebook size**。空间结构、语义组合和多尺度编码都没有被这一个指标概括。

## 六波段细分与亮端敏感性

| 波段 | 全局均值 | 总体标准差 | 图内合并标准差 | 图内方差占比 | 有效像素等于 1 的比例 |
|---|---:|---:|---:|---:|---:|
{bands}

HLP 当前下载实现将保存值裁剪到 [0,10000]；冻结读取器不再额外裁剪。因此边界等于 1 的质量既可能与高反射率场景有关，也受存储前处理影响。没有回查原始未裁剪影像，不能把这些像素全称为传感器饱和或全称为错误。

为检查亮图是否主导图间方差，额外按每图有效 RGB 均值的三个通道平均值筛选。以下只是敏感性分析，**未删除或修改任何训练样本**；0.5 和 0.8 是明确给出的诊断阈值，不是新的数据质量规则。

| 子集条件 | 保留图片 | 排除图片 | RGB 图内方差占比 |
|---|---:|---:|---:|
| 全量 | 218,975 | 0 | {pct(s['hlp_rgb']['within_variance_share'])} |
{sensitivities}

这些结果用于识别总体分布的组成影响；不能把全量百分比推广成“每个 HLP patch 都没有内部差异”。

另对处理顺序靠前的 12 张平均 RGB >0.8 的图片做定向回查，12 张均有超过 99% 的有效像素带 Fmask 雪／冰标记（明细 `bright_examples_audit.json`）。这提供了高亮场景的一种具体来源；该抽查不是随机样本，不能据此估计全量雪图比例，也不能把所有亮端裁剪都归因于雪。Fmask 位定义见上述 Earth Engine 官方目录。

## 显示与 ICC 敏感性检查

![固定显示示例](fixed_display_examples.png)

示例由完整数据池按固定哈希取前 6 张，不按视觉效果筛选。HLP 第二行直接显示反射率，第三行统一采用 `sRGB_encode(clip(reflectance/0.3,0,1))`，只是观看用的固定显示增强。全部定量统计均未使用此增强。增强后能看到结构，说明“直接显示很暗”并不代表原始数据没有细节。

自然图像中有嵌入 ICC 的文件共 **{d['natural_embedded_icc_count']:,} 张**，不能宣称所有输入都严格来自同一标准色域。另从完整 218,975 张中固定哈希抽取 **1,024 张**进行 ICC 敏感性检查：保留当前模型读取方式时图内方差占比 **{pct(a['encoded_within_share_before_icc'])}**；对可转换的已标记图片应用 ICC→sRGB 后为 **{pct(a['encoded_within_share_after_icc'])}**。有 **{len(a['profile_conversion_errors'])} 张**转换未成功而保留原读取值；无 ICC 文件也保留原读取值。这个抽查提示已处理 ICC 的影响在本指标上较小，但不是全量色度标定或对所有色彩指标无影响的证明。

## 与 VQ-VAE 问题的关系

**得到数据支持的部分：**HLP 的全局图间均值变化占比明显更大，RGB 通道变化更集中；现有逐波段全局 z-score 不能自动消除图片间的均值差异。温度和 HLS 的图内、图间项值得分开评价。

**尚未证明的部分：**本次未读取模型 token，因此不能证明码本正在用 token 区分图片身份，不能由 RGB PCA 推断码本坍缩，也不能凭颜色空间决定 1024 与 4096 哪个更好。此前自然图像预训练模型与遥感模型在数据、波段、训练预算上的差异仍需模型实验控制。

**下一步可检验的训练假设：**将各通道均值与空间残差分开预测，或单独控制去均值残差损失权重，可能使小幅局部变化获得更明确的训练信号。LST 使用温度均值与温度残差；HLS 应考虑六维波段均值及六波段残差，并保留恢复原始反射率的约束。新的正式对照应按既定 HLP v2 成员、训练集归一化、5 个完整 epoch 和相同有效 batch，从零训练。

评价时至少同时保留原单位 masked RMSE/MAE、逐图均值误差、去均值残差 RMSE、梯度/边缘误差；HLS 可另列光谱一致性和 RGB 视觉结果。用于重建诊断的“真实均值 + 常数场”是带真实均值的参考基线，不能在部署时把不可得的干净均值当成免费输入。

## 核验与复现

- 全量计数与唯一 ID 校验通过：218,975 张自然图像、218,975 对 HLP、2,796 个 tile。
- HLP HLS/LST 记录索引逐条对齐；每个压缩记录读取时校验解码字节 digest。无效像素不计入矩统计。
- HLP 全量均值和总体标准差与冻结 v2 train 归一化交叉核对，误差见 `results/summary.json → normalization_crosscheck`。
- 在 12 个固定哈希 HLP 样本上，与公共读取器和 NumPy 独立协方差函数交叉核对；20 张自然图像预处理与此前 torchvision 流程逐像素完全相同。
- 原数据、训练 split、归一化与检查点均未修改。本分析为总体描述，没有输出将同 tile 记录假定独立的显著性检验。
- `protocol.json`：数据指纹、下载清单哈希、脚本哈希、公式与范围；本次控制器源码快照为 `color_analyze_executed.py`，统计函数实现为 `statistics_implementation.py`；沿用旧报告的统计公式。
- `results/summary.json`：全量指标、协方差、密度网格、亮图子集敏感性。
- `results/natural_per_image_moments.npz`、`results/hlp_per_image_moments.npz`：带原始 ID 的逐图矩，可复算。
- `audit.json`：ICC 抽样成员及转换错误、独立核验详情；`example_members.json`：示例成员。

从项目根目录运行：

```powershell
.venv-paper1/Scripts/python.exe experiments/openimages_hlp_matched/color_analyze.py
.venv-paper1/Scripts/python.exe experiments/openimages_hlp_matched/audit_examples.py
.venv-paper1/Scripts/python.exe experiments/openimages_hlp_matched/summarize.py
.venv-paper1/Scripts/python.exe experiments/openimages_hlp_matched/write_report.py
```

HLP 的全部分块矩，以及自然图像前 99,840 张的完整分块矩，在验证输入指纹和成员后复用。跨越原 100,000 张边界的分块重新计算；新增图片按完全相同公式计算。复用不会改变图片权重，全部 218,975 张自然图像与全部 218,975 对 HLP 各计入一次。已有分块结果可续用。统计脚本、源指纹及当前代码校验清单共同构成复现记录；如果更改统计公式，应建立新输出版本，避免混用缓存。
'''
    (OUT/'REPORT_ZH.md').write_text(text,encoding='utf-8')
    paths=list((ROOT/'experiments/openimages_hlp_matched').glob('*.py'))+[OUT/n for n in ('protocol.json','audit.json','REPORT_ZH.md','fixed_display_examples.png','example_members.json','bright_examples_audit.json','progress.json','color_analyze_executed.py','statistics_implementation.py','verification.json')]+list(result.glob('*'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    (OUT/'delivery_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT/'REPORT_ZH.md')

if __name__=='__main__':main()
