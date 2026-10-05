# 迁移与扩展实验

## 1. 代码快照

根目录执行 `python tools/export_workspace.py --output output/workspace_code_20260923.zip`。拒绝覆盖已有文件，换名称可再次导出；`--verify 包路径` 核验文件集合与 SHA-256。默认不带 `--output` 仅列出范围/大小。

包内含代码、配置、文档、HLP v2 固定成员清单、选定报告/状态。它不是完整续训备份：不含原始数据、prepared 资产、权重、完整运行目录、报告图片或虚拟环境。状态仅为导出时快照。继续旧实验须另带完整检查点、优化器/RNG/EMA、冻结配置/代码、统计、成员顺序、缓存和运行证据；推理 EMA 导出不含完整续训状态。

## 2. 数据资产

保持以下相对路径，无需重新划分或生成数据。

| 范围 | 必需资产 |
|---|---|
| HLP 新实验 | `artifacts/prithvi_data_v1_20260909.zip`（124,136,280,470 bytes，约 115.61 GiB）及 `.sha256`/verification；整个 `processed_data/hlp_split_v2_20260914/`；`data_inspection/hlp_split_20260914/`；当前读取代码/报告/catalog |
| HLP 历史复现 | 另保留 `processed_data/v1/prithvi/`；不以旧归一化替换 v2 |
| HLC | 整个 `processed_data/v1/lstsr_tb/`；整个 `data_inspection/extracted_metadata/lstsr_tb_final_dataset_20260719/`（源 NPZ、manifest/splits）；原上传归档按来源保留 |
| 模型/辅助实验 | 按入口迁移 `artifacts/pretrained/`、`models/`、所需 `output/.../models/`、OpenImages 数据及相关缓存 |

HLP ZIP 已登记 SHA-256：`2764097028fe9caa44fb55eb5c204c949788f2e6ebfba49b6c37d4171b9c9879`。来源为既有记录，本次整理未重读整个 ZIP 验哈希。

**时间戳属于现有读取契约。** HLP ZIP 的 size/mtime_ns 必须匹配 `storage.json`；HLC 每个 NPZ 必须匹配冻结索引。采用保留纳秒时间戳的传输方式，并在目标文件系统核验；普通 ZIP 解压可能丢失精度。不要以 `verify=False` 绕过或修改冻结 SQLite。若传输改变时间戳，先用源内容哈希证明字节一致，再按来源恢复时间戳或建立有审计记录的新版本。

`attach-prithvi` 附着归档内旧 v1 索引，不会自动补齐独立验证 v2。仅传 ZIP/执行 attach 不足以迁移 v2；不要用归档内旧代码覆盖当前代码。SQLite 在无写入进程时复制；有非空 `-wal` 时不能随意丢弃或只复制主库。复制后按 bundle 哈希验收。

## 3. 环境和读取验收

使用 Python 3.10 或更高版本，优先匹配原实验 packages/pip freeze。最小读取环境：

```powershell
python -m venv .venv
# Windows；Linux 使用 .venv/bin/python
.venv/Scripts/python -m pip install -r eo_data/requirements.txt
.venv/Scripts/python tools/check_workspace.py --dataset all
.venv/Scripts/python tools/check_workspace.py --dataset hlp
# 会顺序读取约 115.61 GiB
.venv/Scripts/python tools/check_workspace.py --dataset hlp --verify-hlp-archive
```

工具从脚本位置定位项目根，可用 `--root` 指定迁移根。两种模态的全部 split 均检查 bundle 资产、样本计数，各读取首条记录，输出指纹和有效像元数。这不是全量源数据审计，也不验证模型环境/预算实现；HLC 全量来源须结合原上传校验清单验收。

2026-09-23 在原工作区实测：HLP/HLC × train/val/test × HLS/LST 共 12 项读取检查通过；结果保存于 `output/workspace_organization_20260923/data_preflight.json`。这是原环境证据，目标环境仍需重跑。

训练依赖见 `experiments/paper1/requirements.txt`；PyTorch/CUDA 匹配目标硬件。EFDiff 额外依赖见 `experiments/efdiff/upstream/requirements.txt` 和各 Modal 脚本的镜像定义；A07 依赖核对其 prepare、packages 和模型实现。不要跨机器直接搬 `.venv-paper1`/`.venv-modal`。`setup_paper1.ps1` 默认本机 Anaconda、system-site-packages 和特定 CUDA，不是通用安装器。

## 4. 新实验新目录

在 `experiments/<新实验名>/` 放新代码/配置，在 `runs/<新实验名>/` 或新云端输出目录保存结果。复制适用实现后逐项适配，不全仓替换历史配置：

- `data.root`、`noise.bank_root`、权重位置、`output_root` 及脚本固定输出目录；Linux 不能直接使用旧 Windows 绝对路径。
- HLP 显式 `prepared="processed_data/hlp_split_v2_20260914"`，HLC 显式 `prepared="processed_data/v1"`。直接调用读取器时，相对 prepared 按当前工作目录解析，建议传 `root / prepared` 的绝对路径。
- QA 形状库/任务统计/缓存匹配成员、split、数据指纹。A06 引用 A05 的 runs；A07 还引用 A06 的 `residual_scale.json`，复制代码不等于复制依赖。
- 默认 5 个完整 epoch。有效 batch=64 时 HLP 每轮 3,422 更新，末次 31 条；不丢弃/补齐、不跨轮凑梯度，按实际样本数加权。仅改 steps 不等于实现 epoch 预算。
- 仅 train 拟合统计，val 调参/选模，test 不参与选择。旧模型见过新 val，不能据其分数声称独立验证。

先在新输出目录进行包含尾批和断点恢复的短程检查，再正式训练。整理/迁移不代表旧任务应自动恢复。Modal profile/workspace/volume 是原部署环境，目标账号需单独配置；Windows 字体路径也需适配。

## 5. Git

根目录没有 Git 历史。本次添加 `.gitignore`，排除大数据、权重、运行输出、缓存和本机环境；没有初始化 Git/上传远端。后续初始化时检查暂存清单；数据契约、配置、预算、划分证据与许可证随代码，大文件独立存储并校验。
