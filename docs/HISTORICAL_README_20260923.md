# 历史首页（整理前副本，2026-09-23）

本文件保留整理前首页；其中相对链接基于原根目录，当前入口见 [新首页](../README.md)。以下启动命令和状态按原文历史语义解释。

统一工作名称为 HLP / HLC；旧读取 ID `prithvi` / `lstsr_tb` 和存储路径保留兼容。HLP 已建立独立验证划分 v2：218,975 对训练、11,684 对验证、11,757 对测试；使用 `prepared="processed_data/hlp_split_v2_20260914"` 和 `train/val/test` 读取。旧 v1 源 `val` 仍表示测试；HLC 的 train/val/test 分别为训练/验证/测试。Prithvi-EO-2.0 在模型描述中继续使用原模型名。

当前论文一：**Cross-View Discrete Tokenization for Noise-Robust Earth Observation**。
主数据为 **HLS-LST-Pairs（HLP）** 的 HLS–fine LST patch 配对记录；已完成冻结预处理和独立可运行的训练框架。**HLS-LST-CrossSensor（HLC）** 为附 HLS 条件影像的 MODIS coarse–Landsat fine LST 月份集合，两份数据独立使用。

- [项目固定约定（后续任务遵守）](AGENTS.md)
- [EFDiff-x₀：HLP v2 全量、5 轮预算的 Modal 复现](experiments/efdiff/README.md)
- [当前 EFDiff：HLC 合成退化，两模型各1个全量epoch、一天期限、每模型两卡；含已完成pilot结果](experiments/efdiff_hlc/README.md)
- [去噪实验框架与运行命令](eo_denoise/README.md)
- [模型与消融配置](experiments/paper1/configs)
- [框架验收记录：19 项测试与 7 个 GPU 短训配置](experiments/paper1/framework_validation.json)
- [当前实验登记表](experiments/paper1/registry.json)
- [本轮消融：卷积 VQ-VAE / 无时间 Prithvi，干净与加噪训练](experiments/paper1/ablation_vq_prithvi/PROTOCOL.md)
- [首轮消融结果与图像对照](runs/paper1/ablation_vq_prithvi_20260910/report/RESULTS.md)
- [数据集命名、来源与划分](data_inspection/FINAL_DATASET_SUMMARY.md)
- [HLP 独立验证划分与读取](data_inspection/hlp_split_20260914/README.md)
- [机器可读名称与读取映射](experiments/data/dataset_catalog.json)
- [统一数据读取接口](eo_data/README.md)
- [冻结数据处理协议](experiments/data/UNIFIED_DATA_PROTOCOL.md)
- [LST 摄氏度 × 100 编码核验](experiments/data/LST_ENCODING_VERIFICATION.md)
- [原研究计划](PAPER1_EXPERIMENT_PLAN.md)

工作区已配置 `.venv-paper1` 和分 split 的 QA 形状库。运行一次真实数据的短训练验收：

```powershell
.\run_paper1.ps1 -Mode smoke
```

两套数据的 fine LST 按摄氏度 × 100 解码；统一接口已标准化，恢复温度需要 `ds.denormalize(sample['image'])/100` 并使用有效 mask。HLP 直接从现有压缩包读取。
