# VQ-VAE / spatial Prithvi: clean versus corrupted training

> 2026-09-14 划分完成：HLP v2 已建立218,975对训练、11,684对独立验证、11,757对固定测试。新读取须指定 `prepared="processed_data/hlp_split_v2_20260914"`，并使用新训练统计。见[成员、规则和读取方式](../../../data_inspection/hlp_split_20260914/README.md)。下文“无独立验证”指旧v1及历史实验；已有训练配置尚未自动切换，新实验须接入v2并重新训练。

2026-09-14 术语校正：本文件记录已执行 pilot。按[当前数据集文档](../../../data_inspection/FINAL_DATASET_SUMMARY.md)，HLP 源 `val` 对应测试集；旧监测流程使用了该集合，因此这里保留其历史用途，不将更名后的结果自动视为未参与开发的独立测试。后续调参须使用从训练 tiles 另建的验证集。Prithvi 在网络名称中继续指模型架构。

本轮按用户要求：以 HLS-LST-Pairs（HLP）的 HLS–fine LST 配对为主，HLS 与 LST 独立训练；比较卷积 VQ-VAE 和去掉时间的 Prithvi-EO-2.0 网络结构，全部随机初始化。用户指定 VAR 论文 VQ-VAE 的参数量作为上限，不要求使用 300M Prithvi。

## 固定设置

| 项目 | 本轮设置 |
|---|---|
| 因素 | 2 模态 × 2 架构 × 2 训练输入，共 8 个独立模型 |
| 初始化 | seed 17，从零开始；同架构同模态两组具有相同初始权重 |
| 训练数据 | 既有 230,659 个 train 对的固定随机顺序；每组训练 1,000 步 × 8 张 = 8,000 张，无重复 epoch |
| 训练输入 | 原始可靠像元；或每张图约 10% 的 QA 云/影/邻接形状遮挡 |
| 目标 | 始终是原始可靠图；无参考像元不计入监督 |
| 训练视图 | 每样本一个视图；S=0、C=0，本轮隔离架构和训练加噪的效果 |
| VQ | 两类架构相同：32 维 latent，16×16 token，512 个码，commitment=0.25 |
| loss | 样本等权标准化 MSE + 0.1 梯度 MSE + VQ 码本/commitment loss |
| 优化 | AdamW，lr=1e-4，100 步 warmup，cosine 到 0.1×，clip=1；BF16 CUDA |
| 定量评价 | 同一固定 256 个源 val（测试成员）对，各两次固定 realization；干净输入与约 10% 遮挡输入分别评价 |
| checkpoint | 全部使用第 1,000 步的 last.pt，不按结果挑选各自最优步数 |
| 单位 | HLS 反射率；LST 摄氏度。复用冻结归一化，LST 反归一化后除以 100 |
| 不确定性 | tile-cluster bootstrap；仅评价样本不确定性，不代表多随机种子训练方差 |

中途每 250 步使用固定 64 个 val 样本作学习曲线监测；这些样本与最终 256 个评价子集使用同一排序规则。所有模型中途都用 10% 遮挡验证，与训练是否加噪无关。最终图像案例固定选择 256 个评价索引中的第 0、64、128、192 个，不根据模型效果挑图。

这是首轮等预算 pilot，8,000 张约占训练集合的 3.47%，不是全数据收敛训练；HLP 无独立验证集；本轮使用的源 val 属于测试成员。不同模型的计算量并未强行匹配，报告同时记录参数量与训练耗时。

## 网络与参数量

| 模态 | 卷积 VQ-VAE | Spatial Prithvi VQ-VAE | 差异 |
|---|---:|---:|---:|
| HLS | 9,328,406 | 9,638,672 | +3.33% |
| LST | 9,322,641 | 8,982,032 | −3.65% |

VAR 的官方 `vae_ch160v4096z32` RGB VQ-VAE 实例化统计为 **108,948,355** 个参数，包含 encoder、decoder、quantizer 和量化前后卷积。本轮全部模型低于该上限。这里对齐的是参数量约束；本轮未移植 VAR 的多尺度残差量化、GAN/LPIPS 或自回归生成器。计数证据见 [parameter_ceiling.json](parameter_ceiling.json)。

Prithvi 使用官方源代码中的 `PrithviViT`、`MAEDecoder` 和 timm `Block`，encoder 改为 320 维 / 6 层 / 8 heads，decoder 为 192 维 / 3 层 / 6 heads，MLP ratio=4。按用户要求将 patch projection 改为二维 Conv2d，位置编码仅含 x/y，无时间或地理元数据编码；输入为单图 `[B,C,256,256]`。原始 MAE 的随机丢 token 替换为本实验的可见 mask 和 VQ 瓶颈，encoder CLS 不向 decoder 传递连续信息，decoder CLS 为固定的可训练模型参数。该模型是缩小并适配 VQ 的 Prithvi-EO-2.0 结构，不等同于原始预训练模型。

卷积模型采用既有四次下采样的 ResNet codec。两种网络都追加一个输入 mask 通道，输出没有额外 sigmoid/clipping，均不允许图像特征绕过离散瓶颈。

## 复现入口

在项目根目录运行：

```powershell
.\.venv-paper1\Scripts\python.exe experiments/paper1/ablation_vq_prithvi/run_ablation.py
.\.venv-paper1\Scripts\python.exe experiments/paper1/ablation_vq_prithvi/report.py
```

首条命令遇到已完成的组会继续下一组，中断的组从其本次运行的 checkpoint 恢复。不会加载以往 smoke 或预训练模型。配置或框架代码变化时拒绝直接续训；需要一个新的运行目录。完整代码快照、配置、样本索引、日志和 checkpoint 在每组目录中保存。

新 `prithvi_eo2_spatial` 及单视图接口改动之后的源码指纹与上一轮框架 smoke 不同；旧 smoke 的有效性记录对应其保存的 `source.zip`。本轮不复用旧 checkpoint。

来源：[VAR 官方实现](https://github.com/FoundationVision/VAR/blob/78b95394fc5896192e3a003e4b295f8ea743c48f/models/vqvae.py)、[Prithvi-EO-2.0 官方实现](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/blob/9eb1b1102806593963daa333bcc491b1c6f8562f/prithvi_mae.py)。`sources/` 保存固定版本代码与 SHA256，未下载任何预训练权重。
