# 39 个 LST 模型：本地推理包

本目录对应 2026-10-13 下载结果表中的全部 39 个模型：SD 7 个、原 VQ 8 个、VQGAN 预训练初始化后微调 8 个、VQ-drifting 8 个、grouped g8k256 8 个。均为原实验登记的 **最终第 5 轮** 权重，不重新按测试集选轮次。

2026-10-14 已完成下载与本地核验：14.35 GB 归档及 441 个包内文件的 SHA-256 全部一致，39/39 个模型通过 CPU 严格加载。`sd_C_affine` 的 pixel10 第一个保存样例已完成 CPU FP32 推理，输出见 `smoke_sd_C_affine_cpu_20261014/`；核验记录见 `local_validation.json`。未占用 GPU，未重新进行完整验证。

这不是 Prithvi RAE 模型；RAE 目前仍是单独的设计方案。这里的输入为 **256×256 单通道 LST、有效区域 mask、可观测区域 mask 和一个温度均值标量**，不是 HLS 六波段，也不是新的 HLC 中心裁剪协议。

## 文件

- `MODEL_MANIFEST.json`：39 个模型的名称、参数文件、原始检查点路径与 SHA-256、推理权重 SHA-256、对应源码版本和预训练依赖。
- `models/<名称>/model.pt`：完整模型 state_dict，包括训练后的参数与缓冲区，保留原始 dtype；去除优化器与随机状态，不用于精确断点续训。
- `models/<名称>/config.original.json`：原始训练配置，含原服务器路径，作为历史证据保留。
- `models/<名称>/training_complete.json`：原始训练完成标记。
- `sources/`：对应训练或修复评估的冻结代码。Stage 1 的六项修复结果使用修复评估源码；其训练权重仍来自原始训练检查点。
- `assets/`：构建 SD / VQGAN 所需的原始预训练依赖。最终参数仍由每个模型自己的 `model.pt` 严格覆盖。
- `download_verification.json`：下载和逐文件完整性核验记录。
- `checks/`：本地 CPU 严格加载检查记录；不代表 GPU 推理、正式验证或新的实验结果。

原始服务器训练检查点保持原样。下载包不包含优化器状态，因此不能直接替代历史训练器的 `last.pt` 进行恢复训练。

## 使用现有本地环境

在 `C:\Users\rog\Desktop\iclr` 中运行 PowerShell。现有 `.venv-paper1` 已提供 PyTorch、diffusers、timm、einops、safetensors、NumPy 和 PyYAML。

列出模型：

```powershell
.\.venv-paper1\Scripts\python.exe outputs\torch_models_20261013\test_local.py --list
```

只检查一个模型能否严格加载，不进行推理：

```powershell
.\.venv-paper1\Scripts\python.exe outputs\torch_models_20261013\test_local.py --model sd_C_affine --check-load
```

使用之前下载的 pixel10 验证样例，在 CPU 上测试：

```powershell
.\.venv-paper1\Scripts\python.exe outputs\torch_models_20261013\test_local.py --model sd_C_affine --condition pixel10 --sample-index 0 --device cpu --output outputs\local_sd_C_affine_sample01
```

可将 `--model` 替换成任意表中名称，例如 `vq_C_asinh`、`vqgan_pretrained_C_affine`、`vq_drifting_C_affine` 或 `grouped_g8k256_C_asinh`。若 GPU 可用且你希望使用 GPU，将 `--device cpu` 改为 `--device cuda:0`；默认不会自动占用 GPU。

每次选择新的输出目录。输出 `prediction.npz` 包含直接预测和 projected 后处理预测，`prediction.png` 显示对照图；`result.json` 记录参数哈希、均值来源、单样例误差和与服务器保存预测的差异。默认本地 FP32、单样例推理与原服务器 BF16、批量验证可能有数值差异，不能把单样例误差当作完整验证集指标。

仅需 CPU 严格加载全部 39 个模型时：

```powershell
.\.venv-paper1\Scripts\python.exe outputs\torch_models_20261013\test_local.py --all --check-load
```

## 输入与均值条件

这批实验使用 `mean_source="oracle"`：提供干净参考图有效像素的真实均温，模型预测围绕该均值的残差。样例模式明确从已保存的参考图计算这个标量，重现原实验条件。

测试自有数据时使用 `--input input.npz`，其中须提供：

- `observation`：摄氏度观测，256×256 或 1×256×256。
- `reference_valid`：评估/输出的有效支持 mask，同样大小。
- `observed_valid`：实际有观测值的 mask，必须是 `reference_valid` 的子集。
- `mean_celsius`：明确提供一个均温标量。
- 可选 `temperature`：摄氏度参考图，仅供计算单样例误差，不作为模型输入。

如果实际应用只能估计均温，请显式传入估计值，并将该测试标为不同于原始 oracle 条件；入口不会悄悄改用观测均温或自动读取参考均温。

HLP 本地数据读取仍遵循 v2：显式指定 `prepared="processed_data/hlp_split_v2_20260914"`，物理摄氏度为 `ds.denormalize(sample["image"]) / 100.0`，同时使用有效 mask。不能将归一化 image 直接除以 100。

此入口仅用于本地单样例测试与加载核验，不提交训练，不启用 W&B，不改动原始配置或历史结果，也不启动全量 validation/test。
