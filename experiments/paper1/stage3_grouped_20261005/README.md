# g8k256 接入八种 LST 表示（2026-10-05）

来源：[defene/lst-grouped-codebook](https://github.com/defene/lst-grouped-codebook)，固定 commit `9465a23adbdc5688093be33c62cadf8860301f2e`。
服务器新目录：`/scratch/zm2864/lst-restoration-bench/output/stage3_grouped_20261005_attempt01`。

## 参数选择依据

仓库已测五种配置中，g8k256 在完整 **validation** 的 clean、noisy10 两个条件均有最低总体 RMSE：0.644670°C、0.721231°C。
g8k16 对应为 0.651494°C、0.727901°C。选择依据不使用 test；仓库已公开的历史 test 查看事实保留。
这是单种子、已测候选中的最优配置，不是全局最优超参数证明。

## 本轮范围与参数

- 8 个从零训练的新 run：R、G、C、G_power、C_power、G_asinh、C_asinh、C_affine，seed17。
- 多尺度分组 VQ：8 组 × 每组 256 码字，总 latent 32 维，每组 4 维。
- 骨干 ch160，2 个 residual blocks；尺度 `[1,2,3,4,5,6,8,10,13,16]`，每组 680 个位置。
- 分组选码后拼回 32 维，经全通道 Phi；quant_resi=0.5，share_quant_resi=4，commitment=0.25。
- VQ loss 按全部通道和尺度平均，不因 8 组而放大 8 倍。
- AdamW，lr=1e-4，betas=(0.9,0.95)，weight_decay=0.01，500 步 warmup 后 cosine，clip=1。
- 微批 8，梯度累积 8，有效 batch64；BF16，单独完成真实微批显存预检。
- 名义索引 43,520 bit/图，加均温 32 bit 为 43,552 bit/图；不含掩码、文件头、共享权重。不是前轮单尺度 2,592 bit/图的同码率对照。

## 明确的适配边界

本轮将仓库最佳验证**架构/码本配置和学习率设置**接入当前八表示对照协议，并非仓库 A07 历史实验的原样复跑。

保留当前的 HLP v2 数据与归一化、pixel10 退化、oracle mean、物理 LST pixel MSE + 0.1 gradient MSE + VQ loss、representation loss=0、观测适配器及零初始化输出头/均温支路。
仓库原实验则使用 QA bank 约10%缺失、visible mean、归一化 LST 单表示，以及额外 3×逐图去均值空间 MSE；该 QA bank 不在此次公开仓库且当前服务器没有其旧路径。
这些差异已写入服务器 manifest；不能拿两个协议的历史分数直接宣称改进，也不能把 g8k256 在本轮称为已验证优于其他架构。

## 数据、预算与验证

显式 `prepared="processed_data/hlp_split_v2_20260914"`，train=218,975，val=11,684；剩余 train 归一化。
每模型 5 个完整 epoch：1,094,875 次记录呈现，17,110 次更新；每轮尾更新 31 条实际加权，不丢弃、补齐或跨 epoch 累积。
固定 epoch5 作为最终检查点。完整 val 的 10 个退化条件、每条 2 个 view；不以 test 调参，本轮不运行 test。

预检包括源文件/配置指纹、数据成员/归一化、尾批和 checkpoint 恢复、GPU 正反向与显存、256 条真实 train 的 4 次技术更新。
技术检查权重丢弃，正式训练重新初始化。所有 run 预检通过后，Slurm 依赖才允许正式任务启动。

## 量化与指标检查

从固定仓库提取原 `GroupedQuantizer` 类，保留量化、全通道 Phi、损失及梯度语义；仅移除未使用的 `VarCodec` 导入和后续不使用的 codec 类。
适配器与该原类核对输出、索引、loss、encoder/codebook/Phi 梯度完全一致。
索引解码测试在相同 BF16 batch 形状下，要求 latent 与重建精确一致；没有连续特征旁路。
分别保存每组、每尺度原始频数、使用码数、entropy/perplexity、组级 top1/top10、图内独有码数/熵及邻接相同率。
不同子码本的同编号不合并统计，也不把理论笛卡尔积容量称为实际码本利用率。

W&B 沿用项目 `lst-restoration-stage1-20260930`，新 group 为 `stage3-grouped-g8k256-20261005`。
服务器 `manifest.json`、`submission.json`、`checks/`、`failures/`、`runs/` 和 Slurm 是实际状态依据。
