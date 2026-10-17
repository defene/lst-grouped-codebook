# LST 每轮验证与选模（2026-10-14）

这是新的、尚未提交实验的训练版本。历史 `lst_restoration` 入口、服务器导出源码、39 个模型的权重与结果均保持原样。新代码从已下载的 Stage 2 源码建立独立副本，合入 Stage 3 grouped 适配器；逐文件来源见 `SOURCE_ORIGINS.json`。

## 已固定的协议

- HLP v2：`prepared=processed_data/hlp_split_v2_20260914`，218,975 条 train，11,684 条 val。继续使用剩余 train 的归一化，不使用 test 选模。
- 训练仍为 5 个完整 epoch、有效 batch 64、尾批 31 条，共 17,110 次更新。验证不会增加训练记录呈现量，也不触发 early stopping。
- 每个 epoch 完成并保存 checkpoint 后，完整评估 val 上的 clean 和 pixel10；各两个视图，退化种子 20260925，所有 epoch 固定不变。每轮合计 46,736 次验证前向记录。
- **唯一选模指标：pixel10 的 raw 像素加权摄氏度 RMSE，越小越好。相同分数保留更早 epoch。** clean 只作辅助指标。训练前的预训练基线不进入这 5 个 epoch 的 best 竞争。
- `val/<condition>/loss_pixel` 是每张图有效像素上 `(预测摄氏度−参考摄氏度)^2 / train_residual_std_celsius^2` 的均值，再对图像和视图等权平均；它不是 VQ/KL/drifting 总损失。
- `val/<condition>/rmse_celsius` 是所有记录、视图、有效像素上的平方误差总和除以总有效像素数，再开平方；同时记录 MAE。**loss_pixel 和 RMSE 的加权方式不同，best 按后者选。**
- 与历史任务一致，均温条件仍为 oracle；这些指标不能自动解释为使用估计均温的部署结果。

## 保存、恢复与报告

|产物|含义|
|---|---|
|`epochs/epoch_01.pt` … `epoch_05.pt`|每轮完整模型、优化器、训练游标和 RNG；已有文件不覆盖|
|`last.pt`|最新恢复点，每 100 次更新及每轮结束更新|
|`best.pt`|按固定验证指标选择的完整 checkpoint，与对应 epoch 文件逐字节相同|
|`epoch_validation/epoch_XX.json`|整轮验证完成后原子写入的指标、数据指纹和 checkpoint/代码/配置哈希|
|`selection.json`|全部已验证 epoch 的分数、最佳轮次及证据哈希|
|`final/`|第 5 轮权重的原有 10 条件完整评估|
|`best_validation_selected/`|best 权重的 10 条件完整评估；best 与 final 相同时仅写复用指向，不重复推理|

验证使用 `eval()` / `inference_mode()`，结束或失败均恢复各子模块模式及 Python、NumPy、Torch RNG。它使用现有模型，不在 GPU 上另建第二个模型。

验证失败则停止训练，不进入下一轮。若在写入 epoch checkpoint 后中断，恢复时先补完该轮验证；若指标已写完但 best/selection 更新中断，则从不可变 epoch checkpoint 和指标恢复选模。缺失更早轮次验证、检查点哈希不符、代码/配置/训练数据指纹变化均拒绝继续。

完整 checkpoint 保留会增加存储：每模型最多 5 份 epoch 文件加 last、best 两份，均含优化器。应在正式部署前按模型实际文件大小核算 scratch 空间。

## 配置与技术验证

39 个新配置位于 `runs/paper1/lst_restoration_epochval_20261014/configs/`；模型初始化、数据、训练超参沿用各自原配置，新增选模协议和独立 run ID。`execution.enabled=false`、`tracking.enabled=false`，没有提交作业或上传 W&B 数据。路径仍以服务器原数据/预训练资产路径为准。

静态检查（不会训练）：

```powershell
.\.venv-paper1\Scripts\python.exe -m experiments.paper1.lst_restoration_epochval_20261014 validate --config runs/paper1/lst_restoration_epochval_20261014/configs/sd_C_affine.json
```

CPU 合成测试（不读取真实数据、不使用 GPU）：

```powershell
.\.venv-paper1\Scripts\python.exe -m experiments.paper1.lst_restoration_epochval_20261014 preflight --config runs/paper1/lst_restoration_epochval_20261014/configs/sd_C_affine.json --part epoch-validation --output outputs/epochval_cpu_check
```

测试覆盖像素加权与等图像加权差别、完整成员/双视图、非有限值拒绝、随机状态和模型模式恢复、5 轮精确尾批、epoch 1/2/5 验证中断恢复、轮内中断恢复、损坏 best 的重建、不可变 checkpoint 校验，以及更早 best 的评估加载。

正式运行仍须重新完成绑定这版源码/配置的数据、模型、资源、预算、相应量化机制和 epoch-validation 检查。旧 preflight 报告不能用于新版本；不能从历史 final checkpoint 冒充从头训练后选出的 best。当前 CPU 合成测试不是完整模型显存检查，也不是实际数据上的新实验结果。

新 CLI 的 `run` 顺序是预训练基线（适用时）→ 5 轮训练及每轮验证 → final 完整评估 → best 完整评估（与 final 不同时）。在用户另行授权前不提交实验。

## 历史结果的限制

已下载清单中，Stage 1/2/3 的 39 个正式模型各只有 `training/last.pt`，没有每轮 checkpoint。它们继续标注“第 5 轮最终权重”，不能补标为“最低验证损失权重”。这份归档无法事后恢复前四轮的验证曲线；若将来获准重新训练，必须使用新输出目录和新 run ID。
