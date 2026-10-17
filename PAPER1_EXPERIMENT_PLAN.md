# 论文一实验执行计划：面向云与异常污染图像还原的离散表征

> 2026-09-14 划分完成：HLP v2 已建立218,975对训练、11,684对独立验证、11,757对固定测试。新读取须指定 `prepared="processed_data/hlp_split_v2_20260914"`，并使用新训练统计。见[成员、规则和读取方式](data_inspection/hlp_split_20260914/README.md)。下文“无独立验证”指旧v1及历史实验；已有训练配置尚未自动切换，新实验须接入v2并重新训练。

> 2026-09-14 术语映射：数据集统一称 HLS-LST-Pairs（HLP，旧 ID `prithvi`）与 HLS-LST-CrossSensor（HLC，旧 ID `lstsr_tb`）；温度网格称 coarse/fine，HLC 的 sequence 为月份集合。HLP 源 val 是测试集，无独立验证集；后续验证须从训练 tiles 建立。本文以下旧名称及 split 用语为历史规划，当前事实以[数据集文档](data_inspection/FINAL_DATASET_SUMMARY.md)和[读取说明](eo_data/README.md)为准。模型名 Prithvi-EO-2.0 保留。

> 2026-09-10 实施更新：用户已确认论文一主数据改为 **Prithvi 一对一 HLS–HR LST**。当前代码、配置和运行说明见 [eo_denoise/README.md](eo_denoise/README.md)。本文件较早的 LSTSR 月份展开、旧 split/test 及单位待确认段落是历史规划；当前执行沿用 Prithvi train/val、统一冻结统计和已核验的摄氏度 × 100 解码，几何增强关闭。短训验收不等于完成本表的正式实验。

> 版本：v4，2026-09-05；用户已确认 sample-level 训练：已有 mask 的离散图像还原与可靠区域自监督遮挡，缺失 sequence/月份样本忽略。状态：方向已确定，实验待实施，不是实验结果。  
> 目标：CVPR 2027，第一篇离散化方法论文。  
> 总方案：[两篇论文研究方案](ICLR_CROSS_VIEW_DISCRETE_TOKENIZATION_PLAN.md)。  
> 本计划细化第一篇的默认实施选择；方法结构和超参数为待 pilot 验证的候选，不能作为已成立的新颖性结论。

## 0. 本轮决定与论文证据链

**已确认实施路线：使用现有影像与 mask，在可靠区域额外遮挡形成自监督样本，学习离散表示并还原同分辨率图像。现有数据即可启动这条路线；完整有云/无云配对不是其前置条件。**

数据与模型接口固定为：

```text
影像 X + 原始可靠 mask M_ref
  → 在可靠区域内选自监督遮挡 M_hide
  → 输入 X_visible + M_in，其中 M_in = M_ref AND NOT M_hide
  → Encoder → 离散 token / token 恢复 → Decoder
  → 同分辨率还原影像 X_hat
```

只把可见像元和输入 mask 送入编码器；隐藏像元的原始值仅用于监督。`M_hide AND M_ref` 用于量化还原效果，`M_in` 用于约束可靠区域保持性。真实原有云孔洞允许预测，但没有参考时不计算孔洞精度。两模态分别用各自可靠 mask；跨视图辅助在此基础上做独立消融。

R1 自监督还原是当前主训练和定量验证路线，R2 实际云孔洞用于应用与局限分析；R3 未掩膜污染值恢复仅为后续可选扩展，不阻塞 R1/R2。下面模型初值仍需 pilot 决定，固定数据划分及 seed `20260905` 不变。

主要问题：**离散表征能否学习地表有效信号，从云、云影等异常污染或其掩膜缺失中还原图像，并在相同表示预算下保持光谱、温度与空间结构？**

离散化方法仍是第一篇的核心，但完整观测自重建仅用于学习/检验表示；主要恢复任务是“异常污染/缺失输入 → 同分辨率地表影像”。主 codec 不允许连续特征绕过量化瓶颈；用同模态上下文或跨视图辅助还原应分别对照。第二篇继续研究 LR→HR 的非同期跨卫星重建，不与本篇同分辨率恢复混淆。

三条证据缺一不可：

1. **保真与压缩：** 多码率重建、光谱/温度物理指标以及实际存储成本。
2. **可还原且不退化：** 云区/异常区的图像还原改善，同时保持可靠区域、边缘与温度极值；token 稳定性用于解释机制。
3. **有用：** 离散表示支撑从污染/缺失图像到地表影像的恢复，超过匹配预算的连续表示、VQ/FSQ 与必要的直接恢复基线。

不纳入本篇主实验：完整 MODIS→Landsat 重建、非同期可靠性网络、跨传感器硬/软观测约束、大型 VAR、所有 EFM 全量微调。这些留给论文二或后续扩展。自然图像几何分析是辅助实验，不以证明“EO 更低维”作为立项条件。

## 1. 当前状态与正式训练前的条件

| 项目 | 当前事实 | 正式实验需要做什么 |
|---|---|---|
| 数据文件 | 已解压，HLS/LST 各 277,808 个 NPZ，配对及文件完整性通过 | loader 从解压目录读取，不依赖已经删除的 ZIP |
| 划分 | train/val/test 已写入三个 manifest | 每次运行校验 manifest hash 和 split，不重新划分 |
| 数值分布 | 已有 train 2,048、val/test 各 256 个 sequence 的抽样报告 | 参数拟合只读取 train；已有 test 描述不作为调参依据 |
| 温度单位 | 原始 int16，scale/offset 未确认 | 从生产代码或明确元数据确认；不能把整数外观当作摄氏度证据 |
| 云污染值与还原参考 | HLS 配置为 clear-only；64 组定向抽查的云/影/邻近区全部已填 nodata | 区分原始污染值、云区缺失、参考可用性；当前不能默认存在真实污染/无污染对 |
| 归一化 | 已有抽样候选，尚非最终参数 | 全 train 按约定权重流式拟合，冻结版本 |
| 历史模型 | 总方案记录 VQ-VAE 结果；当前工作区无模型工程/权重 | 找回代码，或统一重建基线；正式结果符合新划分 |
| 空间独立性 | 无相同 tile ID 跨 split | 检查相邻 tile 的 footprint 与重复像元；发现问题先记录并修订协议，不擅自重抽 seed |
| 算力 | 原方案假设四卡 H200，本轮未实测 | 在实际训练环境测吞吐、显存和数据瓶颈 |

可以先做原始单位 loader smoke test，但未确认单位前不报告 °C 物理误差。历史 checkpoint 若见过新 val，只能做探索；不能用于严格留出主表。当前 HLS 还原可先按已确认反射率单位推进。

## 2. 固定数据协议

数据根目录：`data_inspection/extracted_metadata/lstsr_tb_final_dataset_20260719/`。

| Split | Sequence | Tile | 可用 HR LST 月份样本 |
|---|---:|---:|---:|
| train | 250,449 | 2,249 | 2,608,772 |
| val | 13,694 | 118 | 141,711 |
| test | 13,665 | 123 | 141,524 |

- 划分 seed 固定为 **20260905**。使用 `splits/` 中现有名单，不重新生成 train/val/test。
- 当前 phase4 manifest SHA256：`7110aca1ec7ef1d55b3098686e98139d9230577734d45863468a5622af0c253c`。
- 模型训练种子固定为 `[17, 29, 43]`，首轮使用 `17`；这三个数字不改变数据划分。每次重跑复用同一 seed，不随机挑选结果最好的一次。
- 子集、月份抽样、损坏、bootstrap 用 `SHA256("20260905|namespace|sample_id|...")` 派生整数；禁用进程随机化的 Python `hash()`。记录库版本、生成算法和所有子集名单。
- 有效 LST 月份同时满足 manifest availability 与有效 mask；不从有日期但未通过质量门槛的槽位补样本。
- 两个损坏视图和两模态采用一致的空间翻转；稳定性比较在相同空间坐标进行。

### 2.1 训练抽样单位

**采用 sample-level 训练。一个 paired sample = `(sequence_id, month)` 对应的单张 HR LST + 静态 HLS + 各自 mask。** 12 槽位 NPZ 只是存储容器，不作为12帧模型输入，也不要求一年月份完整。

先按 manifest 的可用月份展开 sample index，再在有效 sample 列表上均匀 shuffle。原 train 对应2,608,772个可用月份样本；同一 HLS 可随该位置不同月份重复进入不同 sample，这是当前明确采用的样本权重。废止先均匀抽 sequence 再抽月份的两级 sampler。

- 缺失整条 sequence、所需模态文件不存在、该月份不可用或目标完全没有可靠像元：不加入该任务的 sample index，并记录 sample ID 与原因。
- 一个 sequence 中某个月缺失，只跳过该月份；保留其他可用 sample。不补齐缺月、不生成不存在的完整 sequence。
- 部分像元 nodata 不等于 sample 不存在：有可用上下文/监督的 sample 保留，内部仍按像元 mask 学习与还原。
- 第一篇 paired sample 不依赖 LR LST；已有 manifest availability 沿用，不因本任务不使用 LR 就重新纳入被原协议剔除的月份。
- 在训练开始前冻结可用 sample index 和跳过记录，所有方法共享；分布式训练不得让各 rank 任意吞掉异常样本而改变步数或 batch。
- sample 继承原 sequence/tile 的 train/val/test，不对展开后的样本重新随机划分，seed仍为20260905。

所有方法共享 sample 列表、损坏种子和每模态暴露量。评估按 sample 计算并按 sequence/tile 聚合；sample-level训练不意味着同一位置不同月份在统计上独立。原始文件完整性核验无缺失；此处定义缺失时的行为，不表示已经丢弃新样本。

### 2.2 分层运行规模

以下是待生成的实验子集，尚未修改任何主划分：

| 层级 | 数据与步骤 | 用途 |
|---|---|---|
| smoke | train 内哈希选 32 个 sequence，逐一覆盖所选样本的可用月份；500 steps | 检查 mask、单位反变换、梯度和数值 |
| pilot-train | 从 train 按哈希选 256 个 tile，每 tile 最多 64 个 sequence，最多 16,384 组 | 中码率机制筛选 |
| pilot-val | 保留 val 全部 118 个 tile，每 tile 最多 16 个 sequence，每组哈希选最多 4 个可用月 | 定期评估，避免只选容易月份 |
| full | 全 train；定期 pilot-val，里程碑评估完整 val | 主表与关键消融 |
| final-test | 定型后完整 test；HLS 每 sequence 评一次，LST 遍历可用月 | 最终泛化结果 |

pilot 只判断是否值得继续，不能代替完整 val/test 结果。短跑只在同层级内比较；所有转入正式主表的模型从相同合法初始化重新训练。没有资格证明有效的短跑结果也必须保留日志。

### 2.3 归一化与 mask

参考已有 [归一化分析](data_inspection/normalization/analysis.md)，但不直接把抽样估计当作全量训练常数。

- HLS：raw × 0.0001 后逐波段 train 全局 z-score。正式paired训练参数按有效sample流计权，即静态HLS按该位置可用sample数重复加权；可用频次加权流式计算，无需重复读取同一数组。
- HR LST：确认解码后使用 train 全局固定尺度，按有效sample等权、sample内有效像元等权拟合，匹配当前sample-level目标；另外保存像元加权统计作诊断。现有HLS每sequence一次的抽样报告仍是历史描述，不直接作为新训练权重的最终参数。
- HLS/LST 分别拟合；不逐图 min-max、不逐月重新标准化、不对 test 重拟合。没有审核依据时不裁剪有效值或目标尾部。
- invalid 在归一化后填 0，并显式传入输入可用 mask；所有方法获得相同 mask。输入可用性与参考可信性使用不同 mask，云区允许输出预测，但缺参考的区域不能计有监督误差。对有参考的受控遮挡实验，遮挡区仍保留在监督和主评估分母中。
- 两模态分别求有效像元/波段均值，再等权合成损失，避免 HLS 六通道自动获得六倍权重。各样本先归约再 batch 平均，空样本跳过并记数。
- 输出头线性；光谱指数、SAM、温度误差均在反归一化后的值上计算。
- 可选归一化对照仅限 train 拟合的 median/IQR 与预先规定物理尺度；在基线 pilot 上选定并全方法共享，不单独替所提方法挑归一化。

## 3. 模型与候选机制

### 3.1 公平基础架构

先用单尺度 ResNet AE 验证机制，再检查机制是否能迁移至历史多尺度残差 VQ，避免一开始同时改变 backbone、量化器、码率与损失。

初始建议：256×256 输入；两模态独立 stem/head，encoder/decoder 同一结构；中码率 latent 为 32×16×16；VQ K=512、embedding dim=32、commitment beta=0.25。无 GAN、无 RGB perceptual loss、无绕过量化瓶颈的 encoder-decoder skip。mask 通道对所有模型一致。

pilot 起始 backbone 为 base width=64、四次 2× 下采样、每 stage 两个 ResBlock、GroupNorm/SiLU；通道倍率 `[1,1,2,2,4]`。正式 backbone 是否增宽由吞吐及 AE 保真决定；最迟 E10 前冻结，所有同组实验同步调整。历史 ch=160、约 109M 参数配置只在找回代码后列为独立容量/多尺度对照，不能把容量变化归因于机制。

在代码就绪后首先保存实际参数量、FLOPs、decoder 结构和量化器开销；上述是设计初值，不是已经测得的配置。

### 3.2 两个待验证因素

**S：污染视图与地表参考的表示一致性。** 在可靠参考区域上按云/影形状形成两个不同缺失输入，或在有经过核实的观测配对时使用污染图输入，以同一可靠地表参考监督还原。VQ 使用 soft code assignment 的对称 JS 作为候选辅助项，距离温度初值 tau=1（同一规范化距离定义）。严重遮挡下不强制不确定区域的全部 token 硬一致；以图像还原和物理保真为主。FSQ 使用量化 latent 一致性作为迁移版本并明确差别。

**C：跨视图结构约束。** 初始候选从量化后的特征经小型 projection head 得到结构表示，在共同有效位置池化为 4×4 网格，比对网格内的空间关系矩阵，而非要求 HLS 和 LST 像素值或硬 token 相同。损失通过量化器的训练梯度路径回到 encoder；projection head 在推理 codec 中移除。

关系矩阵候选为 L2 归一化网格特征间的 cosine similarity，最小化两模态关系矩阵差。仅使用原始有效覆盖≥75%的共同网格单元，固定规则应用到所有相关实验。跨年份地表关系未必一致，因此该设计只是明确可检验的起点；必须与错配配对、无跨视图项比较。

不预设这种关系约束新颖或有效。若 pilot 不成立，应先定位语义/时间不一致或监督位置问题，而不是直接堆更多 loss。shared/private 三码本是后续候选，不在首轮同时引入。

### 3.3 所有方法共享的重建目标

基础损失为 normalized-space masked MSE + 0.1 × masked finite-difference gradient loss；差分仅在两端均有效时计算。VQ 加相同量化损失，FSQ 按其定义不加 VQ codebook loss。

两损坏分支的 reconstruction loss 取平均。四组机制对照都计算两视图，并保留相同 projection head 结构，关闭某项时只令该项系数为零。这样对照不会混入数据增强或额外样本数量的差异。

候选附加项：`lambda_S ∈ {0.01,0.1}`，`lambda_C ∈ {0.01,0.1}`。先分别扫描，再组合各自候选；每种方法公开尝试次数和总计算成本。不得只为新方法无限调参。SAM/光谱指数/极值项先作为评估指标，主机制成立后再做物理损失消融。

## 4. 实验注册表与执行顺序

| ID | 优先级 | 实验 | 固定对照 / 交付 | 前置 |
|---|---|---|---|---|
| E00 | 必做 | 云污染保留、地表参考、单位与分组审计 | 污染/缺失/可靠参考分开，data_protocol、合法训练入口 | 无 |
| E01 | 必做 | train 归一化与 loader 吞吐 | norm hash、精确子集、每模态样本流 | E00 |
| E02 | 必做 | AE/VQ/FSQ smoke 与可逆编码 | 32 组 overfit、形状/梯度/roundtrip 记录 | E01 |
| E03 | 必做 | 中码率基础 pilot | 连续 AE、独立 VQ、独立 FSQ，各模态单独输出 | E02 |
| E04 | 必做 | S 一致性项 pilot | VQ00 vs VQ10，相同受控损坏 | E03 |
| E05 | 必做 | C 结构项 pilot | VQ00 vs VQ01，相同 paired 输入暴露量 | E03 |
| E06 | 必做 | S+C 联合机制 | VQ00/10/01/11 四组，全同 backbone | E04,E05 |
| E07 | 必做 | 错配与退化诊断 | 正确配对/跨 tile 错配、码本使用、细节与极值 | E06 |
| E08 | 必做 | 独立复验 | seed=29 重跑选定 pilot 对照，不再改超参 | E07 |
| E09 | 必做 | 机制判断与冻结 | 选定机制、主码率和训练配置；失败则修改假设 | E08 |
| E10 | 必做 | 三码率正式 tokenizer | VQ00、FSQ00、VQ11：每模型输出 HLS/LST 两套结果 | E09 |
| E11 | 必做 | 中码率正式消融 | 增加 VQ10、VQ01，与 E10 中的 00/11 构成四组 | E10 |
| E12 | 必做 | 云/云影异常还原、覆盖与形状迁移 | 异常区还原、可靠区保真；真实与参考可控实验分表 | E10,E11 |
| E13 | 必做 | 物理保真与坍缩诊断 | SAM/指数、温度极值/梯度、usage/entropy | E10,E11 |
| E14 | 必做 | 像素域云区缺失→离散表示→图像还原 | 输入在编码前污染/遮挡，匹配轻量恢复头 | E10 |
| E15 | 必做 | 条件重建对照 | LST-only、concat、连续 cross、离散 cross，单独计条件成本 | E09 |
| E16 | 必做 | 主要结果三种子与最终测试 | 固定配置 seed 17/29/43、tile CI、统一 test batch | E11–E15 |
| E17 | 扩展 | FSQ + 所提训练机制 | 检查机制是否依赖 VQ，明确损失适配 | E09 |
| E18 | 扩展 | 多尺度残差 VQ 迁移 | 历史 680-token anchor 与同架构无机制对照 | E09 |
| E19 | 扩展 | shared/private 与物理 loss | 一次改变一个因素，所有 token 都计码率 | E11 |
| E20 | 辅助 | 自然 RGB/HLS RGB/6-band/LST 几何分析 | matched 数量、归一化敏感性；只用 train | E01 |
| E21 | 扩展 | TerraMind/EO-VAE 适配参考 | HLS 波段/归一化/码率可比性核实后加入 | E03 |

VQ00 表示 S=0,C=0；VQ10 为 S=1,C=0；VQ01 为 S=0,C=1；VQ11 为 S=1,C=1。每个 dual-modality 运行中的两个网络输出各自 codec 结果；baseline 同样训练两支并使用同样数据，只关闭相关约束。

E16 的“三种子”至少覆盖三档码率的 VQ00/FSQ00/VQ11、中码率两个额外消融、主轻量下游与主条件比较。资源不足时先删扩展项，不能把只有一个 seed 的核心主张写成三种子验证。

## 5. 码率工作点与压缩口径

单尺度的初始三码率采用 K=512：

| 工作点 | 网格 | tokens/image | 固定长度 bits/image | bpp（每空间像元） |
|---|---:|---:|---:|---:|
| low | 8×8 | 64 | 576 | 0.0087890625 |
| medium | 16×16 | 256 | 2,304 | 0.03515625 |
| high | 32×32 | 1,024 | 9,216 | 0.140625 |

仅在 latent 端做共同的 resize-to-rate / resize-back，使三档尽量共用 encoder/decoder 容量；记录实际算力变化。HLS 的上述 bpp 是整幅六波段图共享的码率，另列每 band-pixel 的 bpp/6，不与 LST 或自然 RGB 混为同一失真定义。

FSQ 初始 levels=`[8,8,8]`，隐式词表 512；采用共同的 32-channel backbone，并显式记录量化前后投影到 3 维的结构。码率匹配不等于内部几何相同，此为量化器设计差异，不能隐去。[FSQ 原论文](https://arxiv.org/abs/2309.15505)

历史 680-token/K512 配置为 6,120 bits/image、0.0933837891 bpp，列为独立工作点；不能直接与 medium 比 RMSE 后宣称更优压缩。

正式记录三种量：

1. `R_fixed = sum_s(N_s * log2(K_s)) / (H*W)`，含全部 shared/private token；
2. 逐尺度/组经验熵，只称估计；
3. 实际编码文件字节数。先实现 train-only 静态频率表的可逆熵编码，固定概率平滑；复杂熵模型是扩展，所有方法同等预算。频率表、元数据与共享开销单独列出。

原始有效 mask、尺寸、月信息等若假设解码端已知，必须对全部方法一致并注明；同时报告这些 side information 的实际开销与包含它们的总码率。条件 HLS 更不能隐藏在码率之外。

连续 AE 是同架构容量参考，不是“零码率”。若进入存储比较，列出 latent 元素数、精度及元数据；若加量化得到 codec，必须重新测量其失真，不能引用未量化 AE 的误差。

## 6. 训练预算与 checkpoint 选择

以下为初始可复现实验预算，E01 吞吐测量后可统一修订并记录协议版本：

- optimizer：AdamW，lr=1e-4，betas=(0.9,0.95)，weight_decay=0.01；VQ embedding 的更新规则单独固定，不和 EMA 更新混用。
- batch：每个global batch为64个独立sample（每sample含一张HLS与一个月的HR LST），不是64条12帧sequence；显存不足使用累积保持global batch和样本顺序。四卡为每卡有效batch16，具体microbatch待实测。
- schedule：pilot 20,000 steps（warmup 1,000），full 100,000 steps（warmup 5,000），之后 cosine 降至起始 lr 的 0.1；gradient clip=1.0。
- BF16 若硬件支持；固定实现/随机数种子并记录非确定算子。所有方法相同增强：水平/垂直翻转，不做改变物理值的颜色扰动。
- 每 2,000 steps 评 pilot-val；正式训练每 20,000 steps 完整 val。按相同频率保存 checkpoint。
- 主比较使用相同训练步数、样本暴露量和 backbone，另列 GPU-hours；不能声称不同计算量的模型同时“精确算力匹配”。必要时补充相同 GPU-hour 的学习曲线。
- 各方法相同数量的 lr 候选 `{1e-4,3e-4}` 仅在基础 pilot 明显不收敛时启用；记录各自合法调参成本，full 前冻结。

表示预训练按 val 可靠参考自重建 nRMSE 选 checkpoint；恢复模型按固定有参考云形遮挡 val 的还原 nRMSE 选择：先分别计算遮挡区域和仍可靠区域 nRMSE，再各占50%合成选择分数，HLS/LST 分别报告。联合模型再对两模态等权平均。该规则在 E04 前冻结；无参考真实云区不进入选择分数。token 一致率不作为选择分数，同一 checkpoint 用于所有覆盖率和形状评估。

nRMSE 的分母为冻结的 train 标准差，不用 test 动态范围。若新损失大幅改变收敛速度，在训练阶段决定所有核心对照是否统一延长；不能在看过 test 后只延长某方法。

## 7. 云、云影和异常污染的图像还原协议

### 7.1 本篇“噪声”的操作定义

用户所指噪声是妨碍观测地表的云、云影、云边缘及经核实的异常污染；主要目标是还原地表影像。Gaussian、随机热像元和矩形遮挡不是主问题，仅可作为附加诊断，不能替代去云/异常还原的证据。

HLS v2 的 QA 包含云、云影、邻近区域和气溶胶等标记，标签可能重叠，邻近区域也不等于一定存在污染。以 [HLS 官方算法说明](https://hls.gsfc.nasa.gov/algorithms/) 与产品版本解释为准。`bad_fmask_bits=14` 对应本包筛除 bit1/2/3；先排除 QA fill，不能将 fill 的所有置位解释成所有异常类型。bit0 不直接作为可靠薄云标签。

积雪、水体、真实高温和有效反射率长尾不自动归为噪声；气溶胶或云邻近标记用于候选分层，需审查后决定是否排除为可靠参考。LST 的质量位和填补标志需按各自产品解释，不复用 HLS 位定义。HLS 云形状不能当成不同日期 LST 的真实云位置。

### 7.2 当前数据能支持什么

本轮审计见 [cloud_storage_audit.json](data_inspection/cloud_storage_audit.json)：选 train 中有效比例最低的64组 HLS（定向样本，不是总体估计），共4,194,304像元，其中云/邻近/云影并集76,829像元，没有一个保留全部六波段有效值；83,652个 invalid 像元均为六波段 nodata。与本包 clear-only 配置一致。

因此当前主要可执行任务是**云掩膜后缺失区域的还原**。不能将 nodata 称为原始云纹理，也不能假定剩余像元绝对无污染。真实厚云下的同刻地表在当前单次观测中不可见；模型提供的是基于先验的估计，不是已被观测验证的真实细节。

若要验证保留云/薄云/云影污染值的去除，需要找到同一来源的未掩膜数据及可信参考构造依据。manifest 中的 scene IDs 可用于后续定位，但本轮不新增下载，不改变用户的传感器范围。

### 7.3 三套互不混淆的实验

| 协议 | 输入 | 监督/评价 | 当前状态 |
|---|---|---|---|
| R1 有参考的云形缺失还原 | 在可靠参考区域移植现有 QA 云/影形状，编码前置为缺失 | 原始可靠区域作为参考，重点评价被遮区域 | 现有数据可开展；属于参考可控的缺失还原 |
| R2 实际云掩膜缺失还原 | 数据中原有 nodata 云/影孔洞及仍可见上下文 | 孔洞无真值时仅作定性与一致性分析；可靠区评价保持性 | 现有数据可开展，但不能给孔洞虚构 RMSE |
| R3 保留污染值的真实图像还原 | 未掩膜云/薄云/云影图像 | 经核实的可靠配对；不同时刻参考注明变化和错配 | 当前尚无已验证输入/参考对，不能称已具备 |

R1 不冒充 R3。不同日期影像、其他月份 LST 或插值输出不能直接当作云下真实值；如将来使用多时相参考，先控制变化、配准和时差，独立分表。模型预测不能反过来成为其自身的评估真值。

### 7.4 第一轮可执行训练方案

1. 先以 train 中经过 QA 与数值审查的可靠像元学习离散表示，并固定可靠参考 mask `M_ref`。
2. 从 train 的 Fmask 提取云、影与邻近形状，按连通区域建 mask bank；记录来源 sequence、类别、尺寸和形状 hash。不从 val/test 提取形状用于训练。
3. 将 mask bank 形状移植到可靠训练区域，构造 `M_occ`，输入 mask 为 `M_in = M_ref AND NOT M_occ`。监督仍覆盖 `M_ref`，其中 `M_occ AND M_ref` 是主要还原区域。
4. pilot 候选混合为25%可靠自重建、75%云/影形状缺失还原。形状在当前采样范围内不能达到目标面积时记录实际覆盖，不能通过无限重复小块伪称真实大云团。
5. 覆盖率先设5%、10%、20%；25%、50%作为超出本包自然高有效率条件的压力测试，标为人工扩展。区分云、影、邻近三类，不因为都进入 bad mask 就当作等价污染。
6. val/test 的参考可控评估使用各自 split 的独立 shape bank 和预先固定的遮挡 seed；禁止跨 split 复用同源连通形状。pilot 两次 realization、final 三次，统计时保持 tile cluster。
7. 未识别污染、漏检掩码的真实鲁棒性不能靠完美遮挡 mask 证明。R1 主表明确使用已知 mask，另加边缘膨胀/腐蚀与漏标测试；R3 若具备，则区分提供 QA 与不提供 QA。

薄云透射、云影衰减的数值合成目前没有经过本数据校准，不预设简单加白/乘暗就代表真实污染。可后续作为单独合成协议，但须与真实污染验证分开。

### 7.5 Loss 与还原指标的区域

恢复训练分别计算有参考遮挡区损失 `L_occ` 和参考仍可靠区 `L_keep`，各按自身有效元素归一化；起始目标为 `L_occ + L_keep + 0.1 L_gradient`，可靠自重建样本无 `L_occ` 项。模态内各样本等权，模态之间等权。S/C 项仅为辅助，不替代图像监督。

参考无效的真实云下位置不进入监督损失，但推理允许输出。图像还原评价必须同时报告遮挡区 RMSE/MAE/SAM/指数误差、可靠区变化、边界梯度与全参考区结果。token agreement 仅用于解释，不能作为去云成功的主要证据。

## 8. 指标、统计与可证伪的判断规则

### 8.1 主指标

- HLS：逐 band reflectance RMSE/bias、固定训练尺度 nRMSE、SAM、NDVI/NDWI/NBR 误差。NDWI 预先固定为 `(B3−B5)/(B3+B5)`；NDVI 用 B5/B4，NBR 用 B5/B7。指数分母绝对值低于固定 epsilon=1e-6 的参考像元排除并计数，不根据预测值动态删样本。
- LST：物理 RMSE/MAE/bias、有效邻接梯度 RMSE、每 patch-month 参考有效温度 top-5% 区域误差；可选 top-1%。极值掩码由参考固定，与各模型输出无关。
- Hotspot 辅助指标：参考 P95 作为同一样本的温度阈值，预测与参考使用同一阈值；预先规定无热点/常数场如何计分，不人为保证预测热点也占5%。
- token：clean↔corrupt 和 corrupt↔corrupt agreement、flip rate、VQ soft JS、量化 latent 距离、code usage、perplexity、逐尺度熵。硬 index agreement 只比较同一模型自己的不同输入，不比较两个独立码本的 index 是否相等。
- 同时报告独立边际基线 `sum_k p1(k)*p2(k)` 和超出随机一致率的量，识别常用少数码造成的虚高 agreement；不能用一致率一个指标认定成功。

SAM 仅在参考与预测光谱范数非零时有效；预测零向量不能通过删除而获得更低误差，须以预先固定惩罚或失败计数处理。PSNR/SSIM 是辅助，窗口有效性和 data_range 用 train 协议冻结。

### 8.2 汇总与不确定性

保留每 patch-month 的 SSE、有效数量、MAE、bias 和派生指标。主要结果用 tile 内有效像元汇总后计算 RMSE，再对 tile 等权平均；同时提供全有效像元 micro-RMSE 和 sequence 汇总，避免大 tile 或多月份位置主导唯一结论。HLS 每 sequence 只算一次。

对同一批 tile 的方法差值做 paired cluster bootstrap，重复 2,000 次，固定派生 seed；LST 将同一 tile 的所有 sequence、月份、损坏 realization 保持在同一 cluster。训练种子分别报告均值/标准差，CI 明确是地理重采样的不确定性，不把三种子×月份当独立样本。

### 8.3 Pilot 判断规则（仅作资源分配）

在 E06 首次比较前冻结以下规则。它们是本项目预设门槛，不是 CVPR 标准：

- 至少一个核心轴出现可复现收益：固定有参考云形遮挡区 primary restoration error 相对降低≥3%，同时报告同码率可靠自重建。token agreement 增加只能作为机制支持，不能单独通过还原任务判断。
- 同时约束 clean RMSE、梯度 RMSE、top-5% 温度误差和 HLS SAM：任何关键项相对恶化超过2%都要列为代价，不宣称全面改善；接近零的指标改报绝对差，禁止不稳定百分比。
- 码本使用、有效词表与输出方差不能出现与任务信息损失一致的退化；若出现坍缩证据，稳定性增益不算通过。
- seed=29 不调参复验必须方向一致；证据不明确时补 seed=43，不能删除不利种子。
- 跨视图主张须由 VQ11−VQ10 或 VQ01−VQ00 的增益及正确/错配对照支持。若只有 S 有效，则记录为同模态稳定性结果，不能声称跨视图机制成立。

final 以完整效应量、置信区间、率失真曲线和代价报告，不以刚好跨越 pilot 门槛代替科学结论。无收益也保留结果，不根据 test 重新定义主指标。

## 9. 离散表示用于图像还原

E14 主任务改为 **像素域云形缺失输入 → 离散表示 → 同分辨率影像恢复**，首先用R1获得可度量结果，再在R2真实云孔洞展示局限。HLS 和 HR LST 分别评价；LST 需有自身掩码依据，不能将静态HLS云掩码当作目标月份真值。

- 在 encoder 之前施加缺失；不能先编码完整影像再mask token，否则邻近 token 可能泄漏被遮位置的图像信息。
- 候选两阶段流程：可靠参考训练 tokenizer，然后冻结其decoder/codebook，用污染输入encoder与匹配轻量恢复头预测地表token。初始encoder来源、冻结/微调范围在各基线相同；额外网络算力单列。
- 恢复头初始4 layers、width256、8 heads、30k steps、global batch64、lr1e-4；参数为待pilot调整初值。首轮中码率比较VQ00/FSQ00/VQ11与连续AE，同样输入、mask和样本暴露量。
- 增加空间插值/简单补洞与直接连续恢复网络作为任务基线；连续网络无离散瓶颈的优势和存储代价都要如实报告，不能只挑弱基线。
- 必报可靠参考经过tokenizer的oracle自重建、污染/缺失输入直接解码、恢复头输出三项。oracle仅表示参考重建表现，不代表实际云下恢复精度。
- 主指标是图像还原与可靠区保真；token CE/accuracy和stability是辅助。跨tokenizer不能只比token accuracy宣称图像恢复更好。
- latent masked-token recovery可保留为附录的表示诊断，但不再是本篇唯一主要下游，不称为真实云去除。
- 有同期可信跨视图帮助时可做条件对照，非同期辅助需注明；完整LR→HR和跨卫星观测差异建模留给论文二。

## 10. 条件重建对照与机制排除

E15 只在中码率做有限验证：`cloud-masked image + available cross-view context → same-resolution reference image`。分别测试HLS还原与有可靠参考的HR LST还原；这是同分辨率图像恢复，不是 MODIS→Landsat。真实无参考孔洞不计算监督指标。

模型为 LST-only、concat、连续 cross-attention、离散 cross-attention；条件端编码器与 decoder 容量尽量匹配，记录额外参数和计算。全部使用相同 HLS、month 是否输入的约定和训练预算。

比较新 tokenizer 时冻结同一套条件模块或进行完全匹配的重训；报告无条件 tokenizer 与条件 decoder 两条指标。该实验可以证明条件重建改善，但不能单凭输出改善证明 token 变稳定。此处不得引入完整可靠性 gate、观测时差校正或跨卫星物理约束。

## 11. 主表与实验成本

| 论文材料 | 对应实验 | 主问题 |
|---|---|---|
| Table 1 + RD figure | E10,E13,E16 | HLS/LST 三码率的保真、物理指标与实际码率 |
| Table 2 + restoration curves | E11,E12,E13 | 云/影形状与覆盖率下的异常区还原、可靠区保真及机制 |
| Table 3 | E14 | 像素域污染/缺失输入能否通过离散表示还原图像 |
| Table 4 / appendix | E15 | 在相同 side information 下的条件重建 |
| Qualitative figure | E07,E12,E13 | 预选样本的边缘、极值、失败；统一颜色范围 |
| Appendix | E17–E21 | 量化器/尺度迁移、可选结构与已有 EO tokenizer |

正式基本运行数：E10 为 3 个方法×3 个码率×3 个 seed = 27 个 dual-modality 运行；E11 增加 2×3=6 个运行，共33个。每个 dual 运行产出 HLS/LST 两套 codec，不能误算为只有训练一支的成本。

E14 默认 VQ00/FSQ00/VQ11 在 HLS/LST 各3 seeds，共18个轻量恢复运行；AE连续恢复对照再增6个。直接恢复网络与无需训练的补洞基线另列成本，真实无参考样本不纳入有监督主表。E15 四种模型各3 seeds，共12个中码率条件运行。pilot 的模型/系数探索初步上限16个20k-step dual 运行，超出先记录原因并缩减扩展项。

GPU-hour 估算必须使用 E01/E03 的实际吞吐：`G = steps × measured_seconds_per_step × num_GPUs / 3600`。例如33个正式运行若每个用4卡、100k steps、实测0.5 s/step，单是这部分就需约1,833 GPU-hours；0.5只是演算例子，不是对本模型的测速结果。另加 pilot、轻量头、条件对照、评估和至少20%失败冗余。

若预算超限，按 E19/E20/E21 → E17/E18 → 额外下游的顺序削减。不能优先删掉四组机制对照、FSQ、公平归一化或关键三种子。只有明确重定论文 claim 后才能减少核心任务范围。

## 12. 截止日期与阶段交付

CVPR 2027 官方注册/全文/补充材料截止为 2026-11-10、11-16、11-23 AOE，核对来源：[官方 Dates](https://cvpr.thecvf.com/Conferences/2027/Dates)。

| 阶段 | 内部目标 | 必须交付 |
|---|---|---|
| 9/05–9/12 | E00–E03 | 单位与数据协议、norm/子集hash、smoke、吞吐 |
| 9/12–9/19 | E04–E09 | 四组 pilot、错配/坍缩诊断、复验与机制决策 |
| 9/20–10/10 | E10–E11 | 三码率主表草稿、正式中码率消融 |
| 10/11–10/25 | E12–E15 | robustness、物理保真、轻量下游、条件表 |
| 10/26–11/05 | E16 | 核心三种子、冻结配置、统一最终 test、CI |
| 11/06–11/10 | 注册与主表锁定 | 注册材料、结论与主证据对应 |
| 11/11–11/14 | 内部全文完成 | 初稿审阅、失败案例、复现与匿名化材料 |

正式三种子可在配置冻结后提前运行，不必等到10/26才开始。任何日期延期都先缩减扩展，不能跳过 E00 或偷偷用 test 选模型。

## 13. 运行记录、产物与复现入口

实施后每个 run 至少保存：experiment ID、代码 commit、模型/optimizer config、manifest/split/norm/subset hash、训练 seed、损坏 seed namespace、样本流算法、软件与硬件版本、总参数/decoder参数、步数/样本数/GPU-hours、checkpoint 选择依据、失败或提前终止原因。

目录约定（尚未创建训练工程）：

```text
experiments/paper1/
  registry.json
  protocol/
    data_protocol.md
    normalization.json
    subsets/
    corruption_protocol.json
    model_protocol.json
  runs/<experiment>/<model>/<rate>/seed_<seed>/
    config.json
    metrics.jsonl
    checkpoint_manifest.json
  evaluation/<frozen_version>/
    per_sample_metrics.parquet
    per_tile_metrics.parquet
    rate_distortion.csv
    paired_bootstrap.json
    qualitative_selection.json
```

本轮只创建计划与实验注册表，不创建伪装成已能运行的训练命令，不启动训练。已有 `inspect_data.py` 与解压脚本属于历史数据准备工具，部分依赖已删除 ZIP，不能作为正式训练入口。

交付给论文二的 tokenizer 必须标注训练数据成员、代码/权重hash和单位；论文二的微调另记版本。本篇结果不得依赖论文二先完成。

## 14. 当前第一批实际工作

1. 先核对云/云影污染值是否保留及可靠地表参考是否存在，固定R1/R2/R3边界；同时找到温度解码依据和历史 tokenizer 代码。
2. 只在 train 拟合全量归一化，生成固定 smoke/pilot 名单；已有 val/test 抽样不用于拟合。
3. 建统一 loader、masked metrics、VQ/FSQ/AE smoke；完成编码可逆与无旁路检查。
4. 在实际 GPU 上测吞吐，计算33个正式 tokenizer 运行及附加实验成本。
5. 先跑中码率 VQ00/10/01/11 四组，再决定是否扩展三码率及 shared/private。

基础方法参考：[VQ-VAE](https://arxiv.org/abs/1711.00937)、[FSQ](https://arxiv.org/abs/2309.15505)。EO 适配参考：[TerraMind](https://arxiv.org/abs/2504.11171)、[EO-VAE](https://arxiv.org/abs/2602.12177)。这些工作提供基线，不证明本计划提出的训练机制已经新颖或有效。
