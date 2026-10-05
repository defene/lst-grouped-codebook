# Prithvi + RAE：14 层 ViT decoder 方案

状态：架构方案，尚未实现训练器、完成前向适配验证或提交实验。此次发布仅包含方案与显存估算；没有训练结果、模型权重或 GPU 实测结果。

## 架构与接口

- 输入与重建目标：单帧、六波段 HLS，224×224；拟对 256×256 输入取固定中心窗口 `[16:240, 16:240]`，有效 mask 同步裁剪。
- Encoder：官方预训练 `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`，24 层、宽度 1024、16 个注意力头，约 3.05 亿参数。冻结权重，前向使用 `no_grad`，输入不随机遮挡。
- 去掉 encoder 的 CLS token，保留全部 196×1024 patch 特征；潜空间为连续表示，不做 VQ 量化。
- 新 decoder：`Linear(1024, 1152)` → 14 个 ViT block → LayerNorm → 每个 patch 输出 16×16×6 个值 → 六波段影像。
- Decoder 宽度 1152、MLP 宽度 4096、16 个注意力头；保留 RAE GeneralDecoder 的位置编码与可训练 CLS token 设计，输出重建时去掉 decoder CLS token。
- 原 Prithvi MAE decoder 不保留。EMA 仅复制新 decoder；encoder 无需第二份 EMA。

这是将 RAE ViT-XL decoder 的深度从 28 层减到 14 层，其他宽度保持不变。减半后的 decoder 有 **209,596,160 个可训练参数**（不含固定位置编码），约为原方案的一半。该数值按结构解析计算，尚未经实际模型实例化核验。

本方案覆盖 RAE 第一阶段自编码器重建。潜空间 DiT / flow 生成器属于后续独立阶段，尚未确定。

## 显存预算

假设每卡微批 8、单帧 224×224、BF16 autocast、FP32 参数/梯度/AdamW 状态、decoder EMA，不启用激活检查点。单位为 GiB。

| 项目 | 估算 |
|---|---:|
| 冻结 encoder FP32 权重 | 1.14 |
| Decoder 参数、梯度、AdamW 两份状态、EMA | 3.90 |
| 上述常驻部分 | 5.04 |
| 重建训练峰值预算 | 10–16 |
| 加入感知与对抗损失的峰值预算 | 16–26 |

常驻内存按 encoder 权重每元素 4 字节、decoder 每可训练参数 20 字节计算。峰值区间另留激活、混合精度转换、算子临时内存及分配器余量；这些区间是规划估算，不是测量值。六波段感知损失和判别器设计尚未确定，不能直接套用 RGB LPIPS；该部分显存区间的不确定性更大。微批、attention 实现、优化器和分布式缓冲都会影响实际峰值。

## 后续实现约定

使用 HLP v2：显式传入 `prepared="processed_data/hlp_split_v2_20260914"`，train / val / test 成员分别为 218,975 / 11,684 / 11,757 对；仅使用 val 选模，不以 test 调参。

默认正式预算为 5 个完整 epoch，共 1,094,875 次记录呈现。有效 batch 若为 64，则每轮 3,422 次更新，共 17,110 次；尾批按实际样本数加权，不丢弃、不重复补齐。每卡微批 8 是显存估算条件，不等于有效 batch。

实现时需核验 HLS 波段顺序、反射率单位，以及数据归一化与官方 encoder 输入归一化之间的显式转换；训练集统计只能来自 HLP v2 train。数据入口、损失、缓存指纹、初始化与来源哈希、epoch 边界及运行配置尚未完成接入。六波段重建与感知/对抗目标需单独确定，不能把自然 RGB 图像方案直接视为已适配。

用户当前未授权提交训练或 GPU 预检。本文件和 `proposal.json` 不是可执行启动配置。

## 来源

- [Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)，本地已有权重对应 revision `9eb1b1102806593963daa333bcc491b1c6f8562f`。
- [RAE ViT-XL 配置](https://github.com/bytetriper/RAE/blob/main/configs/decoder/ViTXL/config.json)
- [RAE GeneralDecoder](https://github.com/bytetriper/RAE/blob/main/src/stage1/decoders/decoder.py)
- [RAE 第一阶段训练器](https://github.com/bytetriper/RAE/blob/main/src/train_stage1.py)

机器可读方案见 [proposal.json](proposal.json)。本目录是 2026-10-05 新增方案，不属于仓库原 A06/A07 已完成实验，也不属于原始 `EXPORT_MANIFEST.json` 的历史导出范围。
