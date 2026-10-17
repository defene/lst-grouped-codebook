# 地球观测去噪与离散表示实验

本工作区包含 HLS-LST-Pairs（HLP）、HLS-LST-CrossSensor（HLC）的数据读取、离散表示与 EFDiff 实验。准备在其他机器扩展实验时，从下面三个入口开始。整理日期：2026-09-23。

- [目录与实验导航](docs/REPOSITORY_MAP.md)：代码、近期实验、历史结果与交付包的位置。
- [迁移与扩展实验](docs/MIGRATION.md)：环境、数据、权重、路径和启动前验收。
- [固定项目约定](AGENTS.md)：数据版本、归一化、验证/测试和训练预算。

## 当前数据入口

| 数据集 | 读取 ID | 必须明确的 prepared | train / val / test |
|---|---|---|---|
| HLP | `prithvi` | `processed_data/hlp_split_v2_20260914` | 218,975 / 11,684 / 11,757 对 |
| HLC | `lstsr_tb` | `processed_data/v1` | 2,608,772 / 141,711 / 141,524 条 fine LST 月份样本 |

HLP 的冻结读取器默认仍为旧 v1，旧 `val` 是测试集。新实验须显式使用 v2；HLC 的 `sequence_id` 是月份集合，不默认表示连续时间序列。温度恢复为 `ds.denormalize(sample['image']) / 100.0`，同时使用有效 mask。

正式对照默认 5 个完整 epoch，预算见 [training_budget.json](experiments/paper1/training_budget.json)。HLC 两模型全量 1 轮是单独预算覆盖。

## 导出和迁移检查

以下工具只整理文件或检查读取，不启动训练：

```powershell
python tools/export_workspace.py
python tools/export_workspace.py --output output/workspace_code_20260923.zip
python tools/export_workspace.py --verify output/workspace_code_20260923.zip
# 数据迁移后，安装 eo_data/requirements.txt 再检查
python tools/check_workspace.py --dataset all
```

导出包包含代码、配置、文档、固定成员清单和选定结果说明；不包含原始数据、prepared 索引、模型权重、完整 runs/output 或虚拟环境。包内 `WORKSPACE_MANIFEST.json` 记录逐文件 SHA-256，数据另按[迁移说明](docs/MIGRATION.md)搬运。

根目录 `run_paper1.ps1`、`prepare_data.ps1` 和 `experiments/paper1/configs/` 是历史入口，不能直接当作新的 HLP v2 实验模板。近期 v2 入口及依赖见目录导航。

## 原始说明与证据

- [当前数据集报告](data_inspection/FINAL_DATASET_SUMMARY.md)、[机器可读 catalog](experiments/data/dataset_catalog.json)
- [HLP v2 成员与协议](data_inspection/hlp_split_20260914/README.md)、[读取 API](eo_data/README.md)
- [冻结预处理协议](experiments/data/UNIFIED_DATA_PROTOCOL.md)、[温度单位核验](experiments/data/LST_ENCODING_VERIFICATION.md)
- [训练框架](eo_denoise/README.md)、[历史登记](experiments/paper1/registry.json)
- [原研究计划](PAPER1_EXPERIMENT_PLAN.md)、[跨视图计划](ICLR_CROSS_VIEW_DISCRETE_TOKENIZATION_PLAN.md)
- [整理前首页副本](docs/HISTORICAL_README_20260923.md)

当前根目录没有 `.git/`；导出包是文件快照，不包含版本历史。整理保留冻结代码、数据和实验目录。
