# 仓库导航（2026-09-23）

## 目录职责

| 路径 | 内容与用途 | 迁移方式 |
|---|---|---|
| `eo_data/` | 冻结读取、压缩与索引实现 | 代码包包含；保持原字节 |
| `eo_denoise/` | 模型、噪声、损失、训练公共代码 | 代码包包含 |
| `experiments/` | 专用入口、配置、来源及预算 | 代码包包含；历史配置不自动迁移 |
| `tests/`、`examples/` | 功能检查、读取示例 | 代码包包含；部分依赖本地资产 |
| `docs/`、`tools/` | 导航、迁移与导出/验收工具 | 代码包包含 |
| `processed_data/` | SQLite、split NPY、归一化与存储定位 | 单独迁移当前版本；保留旧 v1 |
| `artifacts/` | HLP 原 ZIP、部分预训练权重 | 大文件单独迁移 |
| `data_inspection/` | 数据报告、划分证据及 HLC 原始 NPZ | 包内只含当前报告和 HLP v2 划分证据 |
| `data/`、`models/` | OpenImages 等辅助数据、SD VAE 等权重 | 按实验迁移并核对来源/哈希 |
| `runs/` | 本地权重、冻结配置、缓存、评估与报告 | 按实验带完整依赖；包内仅选定报告/状态 |
| `output/` | 云端证据、推理权重、交付与导出 ZIP | 按目的选择，不视为统一训练源 |
| `reports/` | 展示网站、报告构建与审阅记录 | 重建展示时单独迁移 |
| `tmp/`、`.venv-*`、缓存 | 临时副本、本机环境、生成缓存 | 排除；目标环境重建 |

没有移动实验目录：冻结配置、检查点指纹和报告脚本相互引用现有路径。`.gitignore` 用于未来 Git 管理；导出范围由 `tools/export_workspace.py` 独立控制。

## 扩展实验入口

状态来自整理时本地文件，不是云端实时查询。保存为 `running` 不证明进程目前仍存活。

| 实验 | 代码与协议 | 数据/预算 | 本地证据与限制 |
|---|---|---|---|
| A05：原始/残差 × 空间损失 | [lst_v2_5epoch](../experiments/paper1/lst_v2_5epoch/PROTOCOL_ZH.md) | HLP v2，四组各 5 轮 | [状态](../runs/paper1/lst_v2_5epoch_20260915/status.json) complete；[报告](../runs/paper1/lst_v2_5epoch_20260915/REPORT_ZH.md) |
| A06：残差尺度 | [prepare.py](../experiments/paper1/lst_residual_scale/prepare.py) | HLP v2，依赖 A05 资产 | [状态](../runs/paper1/lst_residual_scale_20260918/status.json) complete；[报告](../runs/paper1/lst_residual_scale_20260918/REPORT_ZH.md) |
| A07：SD VAE 颜色编码 | [prepare.py](../experiments/paper1/lst_sd_color/prepare.py) | HLP v2，三组各 5 轮计划 | [报告](../runs/paper1/lst_sd_color_20260921/REPORT_ZH.md)说明预训练评估完成、微调未完成；[状态](../runs/paper1/lst_sd_color_20260921/status.json) running |
| EFDiff HLP | [efdiff](../experiments/efdiff/README.md) | HLP v2，5 轮方案 | 后续 HLC 说明记载 HLP 试跑已停止；启动证据不代表完成 |
| EFDiff HLC x₀ / ε | [efdiff_hlc](../experiments/efdiff_hlc/README.md) | HLC v1，全量各 1 轮；固定 512 条 val | [训练完成核验](../output/efdiff_hlc_epoch1_deadline_io16_20260916/completion_verified.json)；完整 test [状态](../output/efdiff_hlc_epoch1_deadline_io16_20260916/full_test_20260921/status.json)仍为 running，未见最终 REPORT.md |
| OpenImages 对照 | `experiments/openimages_var200/`、`openimages_hlp_matched/`、`openimages_hlp_color/` | 辅助自然图像实验 | 对应 `runs/openimages_*`；不可直接混作 HLP 等预算结果 |

扩展 HLP 的 VQ 温度实验，优先阅读 A05 的 `data.py`、`train.py`、`mask_bank.py` 和 `tests/test_lst_v2_epochs.py`。A06/A07 有跨实验依赖，并非复制一个目录即可运行。扩展 HLC 扩散实验，阅读 `deadline_fast_train.py`、对应 YAML/预算与 `full_test_local.py`；当前输入是 fine LST 合成退化，不是真实 MODIS coarse LST。

## 历史实验与交付

`ablation_vq_prithvi`、`var_vs_prithvi_ae`、`var_aligned_v2`、`codebook1024`、`lst_residual`、`lst_full_eval` 保留历史结果。`var_aligned_v2` 的 v2 是实验名称，不能据此推断使用 HLP 划分 v2；以实际 prepared/指纹为准。`experiments/paper1/registry.json` 包含旧训练池 230,659 条及旧 val 语义，不是当前划分总登记表。

最新合作方示例包为 [Prithvi_Partner_Final_EN_20260923.zip](../output/Prithvi_Partner_Final_EN_20260923.zip)，见 [README](../output/prithvi_partner_final_20260923/README.md)。它是官方样例重建交付，不是训练数据迁移包。旧交付继续保留。

EFDiff 上游来源、固定提交及许可证保留在 `experiments/efdiff/upstream/`；导出保留源码和许可证，排除 `.git/`。其他模型许可证随代码保留；未新增统一开源许可证。
