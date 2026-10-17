# 单尺度 VQ 第二轮（2026-10-04）

用户授权两组各覆盖 R、G、C、G_power、C_power、G_asinh、C_asinh、C_affine。
服务器独立目录：`/scratch/zm2864/lst-restoration-bench/output/stage2_vq_20261004_attempt01`。
历史 `stage1_20260930_attempt02` 的源码、配置、检查点保持冻结。

## 两组方法

| 项目 | VQGAN 预训练初始化 + LST 微调 | Drifting VQ-VAE LST 迁移版 |
|---|---|---|
| 初始化 | CompVis 官方 ImageNet f16 K1024 encoder/codebook/decoder | 现有 ch128、3 residual blocks 的单尺度骨干从零训练 |
| latent | 16×16×256 | 16×16×32 |
| 码本 | 1024；官方完整加载 | 1024；作者 notebook 的 uniform(-0.5,0.5) 初始化范围 |
| 重建 | 物理 LST 像素 MSE + 0.1 空间梯度 MSE | 相同 |
| 量化目标 | canonical codebook MSE + 0.25 commitment MSE | 0.25 × 吸引／排斥能量 |
| 重建梯度 | 编码器 STE；码本由 VQ loss 更新 | 作者 notebook 的 hard embedding；编码器由能量项更新，重建梯度进入被选码本和解码器 |
| 微批 / 有效 batch | 8 / 64 | 16 / 64 |
| 初始学习率 | 1e-5 | 1e-4 |

这不是等参数的初始化单因素对照。VQGAN 微调阶段不使用判别器或 LPIPS。
R 表示通过复制到 RGB 输入、输出通道均值接入官方 VQGAN；其余表示保持 RGB 接口。
原输出头、encoder、decoder、quant_conv、post_quant_conv、码本均保留预训练权重。
新的观测适配器初始为恒等映射，mean 分支初始为零；预检对照原始官方模块的直接输出。

## Drifting 来源与明确的图像迁移改动

原始研究为 Ziming Liu 的 [Drifting VQ-VAE 博客与二维 toy notebook](https://kindxiaoming.github.io/blog/2026/representation-1/)，不是成熟图像基准的复现。
原 notebook 下载并保存 SHA256。保留电荷 +1/N、-1/K，排除 self-pair，same-type 无序对只计一次。

图像迁移采用固定 RMS 欧氏距离 `||h-c||/sqrt(D)`、固定 tau=0.2，以及 `U/tau²` 的解析势能归一化。
每个训练微批均匀无放回采样最多 512 个 encoder tokens；保留全部 1024 个码字。
不使用训练后验或验证集调整带宽，不增加 EMA、dead-code restart、GAN 或其他损失。
token 采样使用训练 RNG，checkpoint 保存与恢复 CPU/CUDA RNG。
能量可为负，不用两种方法的 total loss 数值直接判优；比较物理误差、有效码数、perplexity、量化误差。

## 数据、预算和执行门槛

- HLP v2 显式 prepared 路径、剩余 train 归一化；训练 218,975 条记录。
- 每个模型 5 个完整 epoch，共 1,094,875 次记录呈现、17,110 次更新。
- 每轮尾更新 31 条，按真实样本数加权，不跨 epoch 补齐。
- 与前轮相同 pixel10 退化、seed17、样本顺序、物理监督；representation loss 权重为零。
- 验证仅使用 val；10 个条件、每条两个 view；不使用 test 调参。
- VQGAN 先完整记录未微调验证基线，再训练并验证最终 epoch；drifting 训练后验证。
- 每个配置先通过 components、data/budget/resume、GPU model/resource、独立 mechanism/真实数据 smoke 门槛。
- smoke 每种方法/表示只使用 train 的 256 条记录、4 次更新，权重丢弃，不计入正式预算。
- 离散解码检查使用相同 BF16 batch 形状，要求索引、均温、latent、解码结果精确一致。
- 配置、源文件和权重哈希绑定后才允许正式训练；任何前置任务失败都会阻止依赖的正式任务。

源代码部署到独立快照；本目录保留新增实现及一次性准备脚本。服务器 manifest 和提交记录为运行状态事实来源。
