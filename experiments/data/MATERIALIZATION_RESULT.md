# 全量预处理结果

两套数据已分别完成全量预处理与读取验收，共用相同处理方法；未合并数据、未加入人工噪声、未启动模型训练。

| 数据源 / 模态 | train | val | test |
|---|---:|---:|---:|
| prithvi / hls | 230659 | 11757 | — |
| prithvi / lst_hr | 230659 | 11757 | — |
| lstsr_tb / hls | 250449 | 13694 | 13665 |
| lstsr_tb / lst_hr | 2608772 | 141711 | 141524 |

Prithvi 原目录逻辑体积 222.124 GiB；新单文件 ZIP 115.611 GiB，含外部读取索引的完整工作集 115.887 GiB，减少 47.83%。共有 474 个压缩分片，全部 242,416 条记录已逐条解压并核对原始字节 SHA-256。

已按用户授权删除原始 Prithvi 解压目录、旧 ZIP 和独立分片；保留最新 ZIP 与必要的读取索引。相对原目录和旧 ZIP 合计体积，净减少 243.624 GiB。LSTSR 复用现有 NPZ，不另存整套 float32 数组。

最新压缩包：`C:\Users\rog\Desktop\iclr\artifacts\prithvi_data_v1_20260909.zip`。整个文件 SHA-256：`2764097028fe9caa44fb55eb5c204c949788f2e6ebfba49b6c37d4171b9c9879`。压缩包内逐文件 SHA-256 校验与原目录变更检查均通过。

使用入口见 [eo_data/README.md](../../eo_data/README.md)，一键查看四个训练 profile：`python examples/read_eo_data.py`。

四组归一化参数分别位于两个数据集的 `normalization.json`，仅使用各自 train。HLS 按波段；LST 按原始整数编码空间，温度物理单位尚未完全确认。普通 VAE 与 Prithvi 架构 VAE 应共用同一 profile 的样本索引、参数和 mask。

验收包括每个 split/modality 的真实样本读取与反归一化、Windows spawn 进程读入一致性及 8 条 Prithvi 原文件恢复。当前环境未安装 PyTorch，提供的 PyTorch DataLoader 适配器尚未在本环境执行。详细结果见 `materialization_validation.json`。

最后一轮恢复测试新生成的临时目录仍保留（13.98 MiB）；自动审批拒绝清理，工具只返回 `blocked by policy`。此目录不参与训练。
