# HLC 合成退化：EFDiff-x₀ 与 EFDiff-ε

用户已明确指定 **HLS-LST-CrossSensor（HLC）**，并同时运行 EFDiff-x₀ 和 EFDiff-ε。当前采用 fine LST 合成退化；真实 MODIS coarse LST 不作为输入。此前 HLP 试跑已停止，权重、归一化和统计均不用于本轮。

## 本地完整测试集评估（2026-09-21）

用户要求下载本轮权重并评估完整test。`epoch1_export_modal.py`从已完成v3训练检查点导出固定epoch1 EMA，验证两模型共用编码器的所有权重，以及便携编码器batch1/4/16与原模型的一致性。`epoch1_download.py`下载两份新EMA并校验SHA256，复用逐权重一致、SHA256一致的本地Prithvi文件。推理包位于`output/efdiff_hlc_epoch1_deadline_io16_20260916/models/`，约3.29GB；完整续训检查点继续保留云端。

`full_test_local.py`使用HLC v1原始test全部 **141,524条fine-LST月份样本、13,665份HLS/月集合、123 tiles**，保留训练时的中心224×224评估裁剪与Wald×32合成退化，归一化仍来自原完整训练集。测试tile与train/val隔离检查通过。每条记录恰评估一次，尾批4条，不补齐；两模型固定epoch1 EMA，x₀原15步、ε DDIM50步，不依据test调参或选模。先在32条val上完成本地功能检查，不能将该检查成绩当完整test结果。

RTX 5090上batch16顺序执行两模型，共用该批HLS特征；使用GPU BF16 autocast，便携编码器输出float32。按每批首个原始索引固定噪声种子，协议中记录batch、软件、权重与代码哈希。每512条原子保存指标和逐记录CSV，保留固定16条样例图像；恢复时检查协议指纹并从第一份未完成块续跑。保存的是全体逐记录误差明细和16条预测样例，不是所有预测栅格。

`run_full_test_pipeline.py`为一次性本地后台进程，完成后自动生成`full_test_20260921/metrics.json`、`per_record_metrics.csv`、`REPORT.md`和四页对照图，并更新运行登记。`status.json`记录进度/预计剩余时间，`pipeline.json`记录完成或失败，`process.json`记录PID及日志路径；全部位于`output/efdiff_hlc_epoch1_deadline_io16_20260916/`。进程运行期间阻止系统自动睡眠，结束后释放；没有修改系统全局电源配置。

```powershell
# 仅在现有评估进程结束后使用本命令恢复，不要重复启动。
.venv-paper1\Scripts\python.exe experiments/efdiff_hlc/run_full_test_pipeline.py
Get-Content output/efdiff_hlc_epoch1_deadline_io16_20260916/full_test_20260921/status.json
```

## 已完成：一天内完成两模型全量1轮（2026-09-16追加要求）

2026-09-21云端核验：两模型均已完成40,763次更新、2,608,772条呈现量，最后一次准确处理4条。最终报告文件时间为2026-09-17 12:41:12 America/Chicago，距一天要求提出约13小时18分，满足期限；当前应用活动任务为0。固定512条val上的epoch1 EMA结果：x₀ RMSE **1.23242°C**、MAE **0.81824°C**；ε RMSE **1.39856°C**、MAE **0.93252°C**；bicubic RMSE **1.65114°C**。这是固定验证子集成绩，不是完整val或test成绩。训练循环累计时间分别为12小时51分与12小时30分，不含全部容器初始化间隙。报告、图表、指标及运行证据已同步至`output/efdiff_hlc_epoch1_deadline_io16_20260916/`，权重保留云端；完成验收见该目录`completion_verified.json`。下文保留启动方案与费用估计，费用估计不是最终账单。

截止时间为 **2026-09-18 04:23 UTC / 2026-09-17 23:23 America/Chicago**，从用户提出一天要求开始计时。采用每模型两张H200的DDP，两模型同时运行，共四张卡。全量train仍为2,608,772条/模型，有效batch64、每卡微批16、每卡累积2次，40,763次更新，最后4条分成两卡各2条。原global batch成员顺序保留，DDP平均梯度后按实际每卡样本数加权；不缩小训练集、不改epoch定义。每卡16个读取worker、prefetch4，每模型40个CPU物理核和128GiB主机内存。两模型使用相同分配和rank随机种子规则，但不与历史单卡噪声逐位等同。

单卡实测epsilon：H200微批16/32/64约2.040/2.010/2.002秒每更新，B200微批32/64约1.984/1.965秒；纯训练均超过22小时，缺少截止时间余量。新DDP版本先完成两模型132条样本的NCCL梯度/4条尾批/跨容器续训/验证检查，以及80更新的吞吐探测。只有实测总训练估计加3小时余量小于剩余期限，才启动主训练。测速模型和pilot归一化仅用于技术检查，不作为正式初始化或归一化。

当前入口 [deadline_fast_train.py](deadline_fast_train.py)，配置 [deadline_fast_x0.yaml](configs/deadline_fast_x0.yaml)、[deadline_fast_epsilon.yaml](configs/deadline_fast_epsilon.yaml)，预算 [deadline_fast_budget.json](deadline_fast_budget.json)，[运行登记](deadline_run_manifest.json)。[Modal应用](https://modal.com/apps/runlongyu/main/ap-EoCnQWeCx9PJAgrSvkKJlz)，卷目录 `hlc-synthetic-full-epoch1-ddp2-io16-20260916-v3`。正式模型结果在该目录的`train/x0`、`train/epsilon`，比较报告在`train/REPORT.md`。原单卡协调器已取消，完整训练集32份统计继续计算并复用，原文件保留。原单卡主训练在取消前未启动。

启动验收：32份完整train统计已合并为2,608,772条；两模型132条跨容器恢复、4条尾批及验证通过。80更新测速的后60更新平均1.08352秒/更新，外推纯训练12.27小时/模型；两模型同时运行，含存盘、验证等预计14–16小时，预计总费用350–420美元（不是费用上限，未扣账户抵用金）。期限余量检查通过，两份正式训练调用已提交。v2仅完成技术检查，因数据读取吞吐缺少余量未启动主训练；v3采用16 worker/rank、prefetch4，在不同训练区段重新测速。证据见`output/efdiff_hlc_epoch1_deadline_io16_20260916/`。

```powershell
$env:MODAL_PROFILE='runlongyu-target'
$env:PYTHONIOENCODING='utf-8'
.venv-modal\Scripts\python.exe experiments/efdiff_hlc/deadline_fast_control.py status
.venv-modal\Scripts\python.exe experiments/efdiff_hlc/deadline_fast_control.py sync
```

## 原单卡准备方案（已由上面的DDP版本接替，保留复现记录）

用户在pilot测试完成后要求尝试一个完整epoch。两个模型均从头初始化去噪器，共用原冻结预训练Prithvi，从全量HLC train读取全部 **2,608,772条有效月份样本**（250,449份HLS/月集合、2,249 tiles），每条在本轮出现一次。有效batch64、微批16，共 **40,763次更新/模型**，最后4条按实际数量加权更新，不丢弃、不补齐。此次1轮是单独的用户预算覆盖，不改变其他实验5轮默认约定。

- 实际配置：[epoch1_x0.yaml](configs/epoch1_x0.yaml)、[epoch1_epsilon.yaml](configs/epoch1_epsilon.yaml)，入口 `epoch1_train.py`，预算 [epoch1_budget.json](epoch1_budget.json)，运行登记 [epoch1_run_manifest.json](epoch1_run_manifest.json)。此前pilot权重不作为初始化。
- 继续使用fine LST合成Wald×32退化和关联HLS。训练成员是原v1完整train，不改变原数据划分。
- condition/residual归一化重新由全部训练月份中心裁剪拟合；32个统计分片全部完成、覆盖及代码指纹检查通过后合并。pilot训练统计仅供技术smoke使用，主训练禁止沿用。
- 使用原全量方案的5,000步warmup、EMA从5,000步开始且衰减0.9999。pilot的100步warmup/0.99 EMA只适用于此前短预算，因此两次实验除数据量和预算外也存在这些训练设置差异。
- 验证沿用此前固定512条val成员，固定第1轮末EMA与bicubic比较，并输出相同16条样本的图像；**不是完整val分数，不使用test**。
- 每4小时在更新边界保存并排队续训，剩余数据按相同seed/order和精确offset恢复；每500次更新及epoch边界存盘。每个模型单独串行GPU队列，避免同模型并发重复训练。达到40,763更新和2,608,772呈现量后终止。
- CPU预检、32分片统计、两模型131样本的跨容器断点恢复检查全部通过，才自动启动主训练。报告由训练完成后的CPU函数生成，无周期任务。

[Modal任务](https://modal.com/apps/runlongyu/main/ap-sUNTWX5OwSjSqKjvo4N9o0)，结果卷`efdiff-hlc-synthetic-runs`目录`hlc-synthetic-full-epoch1-20260916-v1`。主训练估计一天左右（准备另计）；根据此前吞吐量和[当前H200价格](https://modal.com/pricing)，两模型GPU费用估计210–250美元，CPU/内存/统计等另计。这是估计而非费用上限。

```powershell
$env:MODAL_PROFILE='runlongyu-target'
$env:PYTHONIOENCODING='utf-8'
# 已提交，活动作业期间不要重新部署、修改代码或重复启动。
.venv-modal\Scripts\python.exe experiments/efdiff_hlc/epoch1_control.py status
.venv-modal\Scripts\python.exe experiments/efdiff_hlc/epoch1_control.py sync
```

## 已完成：固定子集 pilot（2026-09-15）

用户在获知全量成本后，先选择固定子集短程对照。以下pilot已经完成；当前授权为上面的全量1轮。后文全量5轮方案保留为历史计划，仍然停止，不能据其命令恢复5轮训练。

- 两组实际配置：[pilot_x0.yaml](configs/pilot_x0.yaml)、[pilot_epsilon.yaml](configs/pilot_epsilon.yaml)。入口为 `pilot_train.py`，不读取后文全量 YAML。
- 固定训练 16,384 条有效 fine-LST patch-month，覆盖原 train 的全部 2,249 tiles，每 tile 6–8 条；固定验证 512 条，覆盖原 val 的全部 118 tiles，每 tile 4–5 条。样本充足的 tile 尽量等额，容量不足时重新分配；tile 内无放回抽样。seed=20260915。
- 成员与哈希见 [pilot_membership/summary.json](pilot_membership/summary.json)，同目录 CSV 含原索引、record_id、month_slot、tile_id。保留原 split，train/val 无 record_id 或 tile 交集；不访问 test。此子集不取代冻结数据集划分。
- 每模型 5 个完整子集 epoch，有效 batch 64、微批16，每轮256更新，共1,280更新、81,920次样本呈现。两模型使用相同成员、顺序与增强规则，从零初始化去噪器，使用相同冻结预训练 Prithvi 条件编码器。
- 仅用选中的16,384个训练月份中心裁剪拟合 condition/residual 统计；不使用已中止的全量统计。训练及验证共用这份统计。
- pilot 共同设置为100步学习率预热、EMA从100步开始、衰减率0.99；此前复制当前权重。各自上游 Adam 学习率和扩散目标保持不变。因此对照包含各方法原有训练目标/学习率差异，不是只改变预测参数化的消融。
- 每轮评估全部512个固定验证样本；主结果为固定第5轮EMA，附bicubic基线、有效像素RMSE/MAE。两个模型保存相同16个验证样本的摄氏度预测，自动生成四页对照图与 `REPORT.md`。不据此宣称全量或收敛复现效果。
- 技术检查单独使用131条训练样本：前2更新保存，换新GPU容器恢复后完成3条尾批和4条验证推理。检查通过才自动开始pilot；检查权重不进入pilot。

Modal：[efdiff-hlc-pilot-20260915](https://modal.com/apps/runlongyu/main/ap-7SJPD46LDHw3gs8xiWpQWu)。结果卷 `efdiff-hlc-synthetic-runs`，目录 `hlc-synthetic-pilot-16384-512-v1`。两模型各一张H200。协调器有限运行，无周期计划，失败停止，全部完成后缩容。轻量证据和图表同步到 `output/efdiff_hlc_pilot_20260915/`。

```powershell
$env:MODAL_PROFILE='runlongyu-target'
$env:PYTHONIOENCODING='utf-8'
# 已提交，不要重复启动或重新部署活动任务。
.venv-modal\Scripts\python.exe experiments/efdiff_hlc/pilot_control.py status
.venv-modal\Scripts\python.exe experiments/efdiff_hlc/pilot_control.py sync
```

本地验证：`test_contract.py` 和 `test_pilot.py` 共7项通过。成员清单已固定，活动实验期间不得重新生成或修改源代码/配置。状态及启动记录见 [pilot_run_manifest.json](pilot_run_manifest.json)。

### 2026-09-16 训练完成与本地推理入口

两模型均已完成5轮、1,280次更新、81,920次样本呈现。固定第5轮EMA的512条验证样本结果：x₀ RMSE 1.97906°C / MAE 1.37033°C；ε RMSE 1.95171°C / MAE 1.39553°C；共同bicubic RMSE 1.65114°C。该短预算下两模型尚未超过bicubic基线。不得将最佳验证轮分数替代预先约定的第五轮主结果。

协调器曾在已完成的smoke被再次调用、返回`smoke_passed`时仍要求`checkpointed`，导致状态报错。训练文件完整；报告已独立重建，原报错保存在云端 `pipeline_before_recovery_20260916.json`。本地协调器源码已增加完成状态识别；未为本次恢复重新启动训练。

`local_export_modal.py` 从已完成的检查点导出第五轮EMA去噪器和完全一致的冻结Prithvi编码器。编码器使用TorchScript、输入固定为batch1的6×224×224反射率，两个随机输入与原编码器数值对齐通过；不是新的编码器实现。完整5GiB/模型的续训检查点留在云端，供本地推理的包不含优化器状态。

`download_local_export.py` 下载到 `output/efdiff_hlc_pilot_20260915/models/`，以云端SHA256验收；`local_infer.py` 从本地HLC原数据重新读取固定验证样本，CPU运行编码器、GPU以单样本运行原版扩散网络，限制自身GPU内存分配为总显存的20%。不改变现有训练进程或依赖环境。测试结果保存于 `output/efdiff_hlc_pilot_20260915/local_test/metrics.json`，独立记录样本数、种子、精度与设备；这不是新的test-set评价，也不是云端逐位重放。

```powershell
$env:PYTHONPATH=(Get-Location).Path
.venv-paper1\Scripts\python.exe experiments/efdiff_hlc/local_infer.py --samples 16
```

## 历史全量计划：配置与数据（已停止，未开始正式训练）

- `configs/x0.yaml`：原版 EFDiff-x₀ UNet + 冻结预训练 Prithvi-EO-2.0-300M，15步 ResShift，残差 L2，Adam 5e-5。
- `configs/epsilon.yaml`：相同骨干与条件编码器，1000步线性扩散、噪声 L1，Adam 2e-5；评估50步 DDIM。
- 两份完整 YAML 是本入口实际读取的配置；不存在上游 300k steps、旧 HLP 路径或50条验证样本等残留设置。结构源码仍固定为上游 `773f67375991432d790c1e99076597524dbe0f95`。
- 读取 `open_dataset('lstsr_tb', ..., prepared='processed_data/v1')`，保留原 train/val/test 成员及归一化资产。

| split | HLS/月集合记录 | 有效 fine LST 月份样本 | tiles |
|---|---:|---:|---:|
| train | 250,449 | 2,608,772 | 2,249 |
| val | 13,694 | 141,711 | 118 |
| test | 13,665 | 141,524 | 123 |

一个有效 fine LST 月份是一条训练样本；同一 record_id 的不同月份共用其一份 HLS。月份集合可能跨年份，HLS 不保证与每个月份同期，本实验不建立时间序列模型。预检验证月份到 HLS 的 record_id 配对，以及 split 间 record_id/tile 隔离；tile 隔离不构成地理 footprint 零重叠证明。

## 退化、单位与统计

fine LST 使用冻结读取器反归一化后 `/100` 恢复摄氏度，有效像素掩码保持不变。缺测先以该完整256×256图幅有效像素均值填补，仅用于数值退化；它们不参与损失和指标。训练随机裁剪224×224并翻转，验证/测试中心裁剪；PSF sigma=32/pi、nearest边界、32倍面积平均得到7×7，再 bicubic 到224×224。

预测目标为 `fine LST - synthetic bicubic LST`。新任务的 condition/residual 3σ 统计由全部2,608,772个训练月份、中心裁剪、有效像素拟合，各月份等权。验证和测试共用训练统计。HLS 恢复反射率后截断至[0,1]，无效像素置零，再经上游预训练 Prithvi 常量标准化。

## 预算与评估

两个模型各训练5个完整epoch，有效batch64、微批16，尾批4条，每轮40,763更新、共203,815更新，13,043,860次样本呈现/模型。每轮打乱月份集合，再打乱集合内部有效月份；同一套规则和seed20260915用于两模型，每条月份恰好出现一次。这样保留全量暴露并减少重复解压NPZ，不按月份排序模拟时间依赖。尾梯度按实际样本数加权，不丢弃、补齐或跨epoch累积。

每轮固定4,096条val监测，明确属于验证子集，不作为完整val分数。正式对照固定第5轮EMA，不用早停或test选模；训练完成后对完整val/test分块评估，聚合有效像素RMSE、MAE和bicubic RMSE。x₀原生15步，ε使用50步DDIM，均保留物理bicubic基准并加预测残差。此轮不声称已经完成论文SSIM/LPIPS/FID全部指标。

短程为每模型131条train、4条val，用32条train拟合临时统计。先完成2次更新存盘，换新GPU容器恢复到第3次尾更新并推理，以核对真实断点恢复。短程权重和统计不会进入正式训练。正式断点每250更新及epoch边界持久化；每6小时保存并换容器续训。代码与YAML指纹变化会拒绝恢复。

## Modal 操作

工作区 `runlongyu/main`，profile `runlongyu-target`。数据卷 `hls-lst-datasets-20260914`；结果卷 `efdiff-hlc-synthetic-runs`；目录 `hlc-synthetic-x0-epsilon-5epoch-v1`。

使用部署应用执行有限训练队列，避免数日任务受单个24小时函数超时限制；没有cron计划或公共HTTP入口。每个模型有互斥锁，协调器最多执行两段任务后续接下一段，预算及完整评估完成即结束；无任务时GPU缩容。任务错误时停止并记录 `failure_<variant>.json`，不自动无限重试。用户停止应用后禁止自动恢复。

```powershell
$env:MODAL_PROFILE='runlongyu-target'
$env:PYTHONIOENCODING='utf-8'
.venv-modal\Scripts\python.exe -m modal deploy --env main experiments/efdiff_hlc/modal_app.py
.venv-modal\Scripts\python.exe experiments/efdiff_hlc/control.py prepare
.venv-modal\Scripts\python.exe experiments/efdiff_hlc/control.py status
# 预检、全量统计、两模型短训和恢复全部通过后启动正式训练：
.venv-modal\Scripts\python.exe experiments/efdiff_hlc/control.py start
.venv-modal\Scripts\python.exe experiments/efdiff_hlc/control.py sync
```

活动作业存在时不要重复部署或准备。轻量证据同步至 `output/efdiff_hlc_synthetic_20260915/`。原始数据/上游代码/旧HLP实验均保留。

本地验证：`PYTHONPATH` 设为项目目录，运行 `.venv-paper1\Scripts\python.exe -m pytest experiments/efdiff_hlc/test_contract.py -q`。已通过月份配对、全量无重复顺序、恢复偏移、合成coarse一致性、掩码梯度、尾批权重及两模型配置一致性检查。
