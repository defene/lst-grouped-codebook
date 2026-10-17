# LST 离散表征与图像复原实验框架

> 后续实施以 [最终实验设计 v2](FINAL_PLAN_V2_ZH.md) 为准：前25组诊断、主体76组、扩展12组，最多88组。[v2清单](../lst_restoration_final_v2_20260927/manifest.json) 仅为关闭执行的逻辑设计，缺失入口仍须先接通；下文及 `table-plan` 的174/166组计划属于历史实现，不是v2可直接运行的配置。

**本版目标：在明确的温度侧信息和观测支持条件下，用紧凑离散表征恢复受传感器噪声、随机缺失和云形遮盖污染的 LST，并比较质量、码率与计算成本。**

下游分类、分割、生成模型、token 预测器均不属于本轮。当前先做 HLP 的 LST；HLS 沿用项目读取器，但本运行器尚未实现多波段实验，不能把 LST 温度扰动直接套到反射率。

## 当前状态

按结果表组织的新计划见 [EXPERIMENT_ORDER_ZH.md](EXPERIMENT_ORDER_ZH.md)：6张主表、4张补表、分阶段训练与可复用job_id。`table-plan`生成独立快照，不替换下述历史174份计划。

- 已实现数据、表示、污染、量化器、模型适配、损失、断点训练、完整评估、码流、统计、报告和未来预检入口。
- 已进行源码审阅、Python 语法和配置注册检查；2026-09-27 按用户要求补做了纯 CPU 的合成数值颜色映射往返与自动微分检查。**没有执行数据数值预检、模型初始化、模型前向/反向、训练、评估、GPU 测速或权重下载。**
- 所有生成配置均为 `execution.enabled=false`。这是遵守用户“先不要跑”的本次执行状态；不是另起一轮训练。
- 模型运行正确性、实测参数量/显存、模型输出颜色的逆映射梯度和真实数据接口仍待数值验收。独立映射的局部梯度已检查，不能替代模型验收。SatMAE++ 两尺度适配源码已接入、运行未验证；官方预训练模型仅接受可验证的本地权重。
- 历史目录及已有监督进程不在此框架控制范围；本轮不改写或重启历史实验。

## 数据流

新增幂变换/asinh 与共同分段颜色路径的公式、实测映射误差和适用限制见 [COLOUR_MAPPING_ZH.md](COLOUR_MAPPING_ZH.md)。这是合成标量检查，不是 HLP 重建结果。

```mermaid
flowchart LR
    A[冻结 HLP v2 LST] --> B[摄氏度与参考有效掩码]
    B --> C[物理域合成污染]
    C --> D[规定均值条件与残差/绝对温度]
    D --> E[R / G / C / 幂变换颜色 / asinh颜色 / 仿射控制]
    E --> F[统一观测接口与编码器]
    F --> Q[VQ / FSQ / LFQ / BSQ 或连续瓶颈]
    Q --> G[解码器]
    G --> H[逆表示映射与温度恢复]
    B --> I[仅供损失和评估的参考目标]
    H --> J[物理误差 / 码率 / 速度 / 成本]
    I --> J
```

编码器不接收目标温度、干净残差或隐藏的合成噪声支持。完整参考均值仅在预先声明的 oracle 条件下作为共同的一个 float32 标量提供；visible 对照只从带噪可见观测计算均值。

## 模型接入

2026-09-27 更新：新增地理模型使用网络框架从零训练，不进行 DEM/NDVI 权重迁移。详细适配和未完成门槛见 [GEO_BASELINES_ZH.md](GEO_BASELINES_ZH.md)。

|类型|本版实现|比较方式与限制|
|---|---|---|
|离散主线|VQ、FSQ、LFQ、BSQ|同 VAR 风格卷积骨干，默认 16×16 索引网格、1024 状态；公式适配，不宣称复现每篇论文全配方|
|暂缓|连续 AE、VAR 多尺度|已移出本轮配置计划，保留历史实现|
|新增地理 VAE|EO-VAE-LST|从零训练，固定 1/3 通道，连续 Gaussian 瓶颈；不使用光学预训练权重|
|新增地理离散模型|TerraMind-FSQ-LST|从零训练，1024 状态、16×16 索引；保留扩散解码，完整采样成本单列|
|地球空间结构|Prithvi 二维 MAE|真实 visible-token dropping/mask-token 解码，无时间输入；从零训练；复原目标适配|
|地球空间多尺度 MAE|SatMAE++-LST|128→256两尺度重建、从零训练；patch16独立对照，运行待验收|
|自然图像预训练|SD、SDXL、FLUX.1 KL-VAE|零微调与全部参数微调；本地官方权重摘要；SD/SDXL 不当成独立架构创新|

FLUX.1 入口仅支持已核实的 diffusers AutoencoderKL 转换目录，不能直接把原始 `ae.safetensors` 塞进其他架构。参数与载荷均在模型审计时实测；尚未宣称这些模型已参数对齐。机制来源和适配区别见 [model_sources.md](model_sources.md)。

## 条件与对照

核心保留三种训练条件：完全不施加人工污染、10% 随机有效像素缺失、等概率混合污染。混合在每次记录呈现时只分配一种条件，实际类别计数入日志，不把六种污染变成六倍 epoch。

评估覆盖 clean、pixel10、Gaussian、条带、云形缺失、区域偏差、联合污染，以及预先列出的更强 Gaussian/30% 云形遮盖。`cloud_bias` 留作未用于混合训练的压力条件。整块 16×16 MAE 补全另列，不与 10% 像素遮盖混成同一任务。

配置按批次组织：核心 seed17 → 核心 seed23/42 复验 → 灰度通道控制 → 地理架构从零训练 → 自然图像预训练 → 可见均值 → MAE → 仿射增益 → 原生绝对温度 → 8/12 bit 率失真。**目录里存在配置不等于批准全量运行，也不代表已作出资源预算。**

当前注册174份候选训练配置，其中首批核心24组（4量化器×2表示×3训练条件），复验48组；其余为对照/消融。全注册表3352个预训练后或从零训练后的评估单元，另有自然VAE零微调单元；12份新增地理架构配置替代已暂缓的12份 AE/VAR 配置；这是配置覆盖数，不是已提交的运行队列。SatMAE++对应4份配置已补上两尺度适配实现，仍须运行验收后才能启用。

## 使用入口

以下两条只生成/核对计划，不加载 torch、数据或模型，可用于当前设计阶段：

```powershell
.venv-paper1/Scripts/python.exe -X utf8 -m experiments.paper1.lst_restoration plan --output experiments/paper1/lst_restoration_20260927_satmae
.venv-paper1/Scripts/python.exe -X utf8 -m experiments.paper1.lst_restoration validate --config experiments/paper1/lst_restoration_20260927_satmae/configs/vq_R_mixed_seed17.json
```

未来恢复执行后，`preflight --part components|data-budget|model|all` 才会进行数值、数据或模型检查；**它会做优化器更新的合成技术测试，当前没有运行。**

```powershell
# 以下仅为未来调用说明：当前不要执行。
.venv-paper1/Scripts/python.exe -X utf8 -m experiments.paper1.lst_restoration preflight --config <config.json> --part all --device cuda:0 --output <audit-directory>
.venv-paper1/Scripts/python.exe -X utf8 -m experiments.paper1.lst_restoration assemble-preflight --config <config.json> --reports <components.json> <data_budget.json> <model.json> --output <preflight.json>
```

预检摘要绑定配置（排除执行开关）、所有实际本地源码和依赖指纹。改变表示、微批、模型或代码后旧预检不能自动通过。通过技术验收并恢复执行授权后，配置可显式启用，绑定上述报告；`run --config ... --output ...` 才依次执行自然预训练零微调评估（如适用）、5 epoch 训练、完整 val/test 评估与报告。

核心还要用 `alignment-audit --reports <四个量化器的model.json> --output <alignment.json>` 核对实测参数差异≤5%、同通道及索引载荷，再把报告路径写入 `execution.alignment_report`。这与单模型“参数数目自洽”检查不同，不能拿旧架构数值跳过。

`train`、`evaluate --checkpoint <last.pt|pretrained>` 也可分开调用。`report --job ...` 只读结果；`gallery --cell ...` 只渲染已存预测；`compare --a ... --b ... --output ...` 要求物理输入一致；`token-stability` 则只接受同一模型权重的不同污染条件。

本版不启动后台 supervisor，不自动调度所有候选作业。输出目录 OS 锁、STOP、断点指纹防止重复或误续训。模型权重依赖缺失会明确报错，运行器不会自动下载。

## 完整性与可复现

- HLP v2 固定 218,975 / 11,684 / 11,757；摄氏度只能先反归一化再除以 100。
- 每模型 5 个完整 epoch、1,094,875 次记录呈现；有效 batch64，共 17,110 更新，每轮末次31条，不丢弃/补齐尾批。
- 核心微批固定4，LFQ/BSQ 的 batch 熵基于4幅图的256个 token。梯度累积并不把非线性 batch 熵变成 batch64 熵。
- 选用最后完整 epoch，评估完整 val/test；不使用 test 选模。两个评估视图保留同一 record/tile 身份。
- 同条件跨组核对物理观测、目标和掩码摘要；均值摘要仅同均值协议比较，编码摘要仅同表示比较。
- 正式报告区分 raw / 共同范围投影、部分 / 最终结果；图片必须另行查看，生成 PNG 不会自动标记“视觉已复核”。
- 作业成本日志只追加。异常退出后未存盘的重算仍计成本，下界不包含停机空档；共享作业时长不能相加当物理单卡 GPU-hours。

断点预检目前通过复用正式 train_update 的 CPU 小模型设计验证，尚未执行；它本身也不能证明完整VAE在CUDA/AMP下的逐位续训等价。实际模型与硬件验收结果应如实单列。

详细科研口径见 [PROTOCOL_ZH.md](PROTOCOL_ZH.md)。静态检查不会替代未来数值与资源验收。

SatMAE++最新接入细节见 [SATMAE_LST_ZH.md](SATMAE_LST_ZH.md)：保留两尺度结构和MSE/L1监督形式，明确LST有效区域适配及与Prithvi的输入/损失差别。
