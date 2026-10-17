# EFDiff 在 HLP v2 上的复现

目标：在 Modal `runlongyu/main` 上复现 EFDiff-x₀ 的模型与退化方法，采用本项目的独立验证划分及 **5 个完整 epoch** 预算。

- 论文：https://arxiv.org/abs/2604.16841
- 源码：https://github.com/defene/EFDiff
- 固定提交：`773f67375991432d790c1e99076597524dbe0f95`；`upstream/` 保持原样。
- 数据：HLS-LST-Pairs（HLP），`processed_data/hlp_split_v2_20260914`。
- train/val/test：218,975 / 11,684 / 11,757；2,796 / 147 / 151 tiles。
- 正式训练：有效 batch 64，微批 4，17,110 次更新，1,094,875 次记录呈现；每轮尾更新 31 条，按实际样本数加权。
- 首轮模型：EFDiff-x₀，官方 64 基础通道 UNet、冻结预训练 Prithvi-EO-2.0-300M、15 步 ResShift、Adam 5e-5、bf16、梯度裁剪 1、EMA 0.9999；继承官方 5,000 次更新 warmup 和 EMA 启动时点。

## 数据和复现差异

256×256 原始记录经随机 224×224 裁剪和翻转用于训练，验证使用中心裁剪。LST 按冻结读取器的 v2 归一化反变换后除以 100 得到摄氏度。Gaussian PSF sigma=32/pi、面积平均到 7×7、bicubic 回到 224×224，预测 fine LST 相对于上采样输入的残差。

v2 标准化资产原样保留。EFDiff 另需 condition/residual 的 3σ 标准化，因此为当前全部训练成员重新拟合独立任务统计；中心裁剪、每记录等权、仅有效目标像素参与；验证和测试共用这些训练统计。HLS 先恢复物理反射率并按原模型裁剪至 [0,1]，再由冻结编码器应用官方源码中的预训练常量；这不是用验证集重新拟合。

原仓库的统计脚本和读取器使用了不同的温度过滤/Fmask规则。本适配统一使用项目冻结的有效性规则，不再增加会改变既定训练成员的过滤；热红外缺测以整幅有效像素均值填补以完成 PSF，但缺测像素不进入损失和指标；光学无效像素置零。每个记录损失先对有效像素取均值，再按记录平均。此差异明确属于项目掩码约定的适配。

不使用上游 `LSTDataset` 的目录划分与缓存：它把目录 `val` 归入测试，并在训练 tiles 中抽取少量记录作验证。本任务严格使用固定 v2 索引，训练期间仅用完整独立 val 的 EMA RMSE 选模，test 仅做启动前读取完整性检查，不计算分数或用于调参。最终测试及 SSIM/LPIPS/FID 是训练后单独评估事项；当前流水线不声称已复现论文全表。

论文附录写 diffusion 200k iterations，仓库配置写 300k；本文实验遵循用户的 5 epochs。附录表格的 test=11,938 / 153 tiles 也与正文及当前归档不同；本任务仅采用已核验本地成员。结果应称“EFDiff-x₀，HLP v2，5-epoch 适配复现”，不可直接称论文数字精确复现。

## 运行与证据

Modal 数据卷：`hls-lst-datasets-20260914`。输出卷：`efdiff-hlp-v2-runs`，运行目录 `hlp-v2-x0-5epoch-20260915`。使用 `runlongyu-target` profile；其他同名旧 profile 不一定属于目标工作区。

```powershell
$env:MODAL_PROFILE='runlongyu-target'
$env:PYTHONIOENCODING='utf-8'
.venv-modal\Scripts\python.exe -m modal run --detach --env main experiments/efdiff/modal_app.py
.venv-modal\Scripts\python.exe experiments/efdiff/status.py
.venv-modal\Scripts\python.exe experiments/efdiff/sync_evidence.py
```

已有活动任务时不要重复启动。流水线先独立完成 HLP 整个 ZIP 的 SHA256、版本资产哈希、成员隔离和真实配对读取核验，再并行拟合全训练统计与运行 H200 短程检查。短程为 131 条训练记录、3 次更新、4 条独立验证记录，专门包含 3 条的尾更新；其 32 条训练统计只用于技术检查。只有全量统计与短程检查通过，才进入正式训练。HLC 不属于此首轮范围，全数据卷的上传核验仍由独立任务负责。

`preflight.json`、`normalization.json`、`smoke/status.json`、`formal/provenance.json`、`formal/status.json`、`formal/metrics.jsonl`、`formal/validation_epoch_*.json` 记录各阶段实际状态；`latest.pt` 保存模型、预训练编码器、优化器、EMA、RNG、epoch 和记录偏移，每 250 更新及每轮保存，`best_ema.pt` 保存验证选中的去噪器。代码快照及 pip freeze 随实验保存。每 6 小时可保存并释放 GPU 后续跑，不跨 epoch 补齐尾批。

本地检查：`PYTHONPATH` 设为项目目录，执行 `.venv-paper1\Scripts\python.exe -m pytest experiments/efdiff/test_contract.py -q`。已覆盖配对及划分、确定性裁剪、掩码梯度、尾批实际权重与 5 轮更新数。

另外，原仓库在 EMA 启动前保留初始化权重；本入口在前 5,000 更新复制当前去噪器，之后才启用指数平均，避免早期验证使用随机初始化参数。重建使用未裁剪的物理 bicubic 条件加预测残差，模型条件仍执行原来的 3σ 裁剪。

2026-09-15 实测：本地 4 项检查通过；Modal HLP 完整 SHA256/划分/读取核验通过；全量 218,975 条 train 统计拟合完成；H200 上 131 条短程训练及验证推理、保存通过。首次转入正式训练时遇到 Hugging Face Xet 日志阻止卷 reload，已把 Xet 缓存移到临时目录并启用单次使用 GPU 容器。已有核验、统计和短程代码指纹均通过时，可用上述 `modal run` 命令追加 `--mode resume` 恢复流水线。实际活动任务以 `run_manifest.json` 为准；日志与轻量证据可同步到 `output/efdiff_hlp_v2_20260915/`。

正式训练已确认开始：任务 `ap-EwzYm7jH8ViQ1y3vbGMsC3`，首次观察到第 10 次更新、640 次记录呈现。此为启动证据，不是实时进度或训练完成声明。`metrics.jsonl` 中的 `loss` 是该更新的最后一个微批损失，`grad_norm` 为按全部有效 batch 记录平均后的梯度范数；验证指标按全部有效像素汇总。
