# 两套遥感数据的统一处理流程

2026-09-14 术语同步：[数据集文档](../../data_inspection/FINAL_DATASET_SUMMARY.md)为当前名称、来源与 split 语义依据；[名称映射](dataset_catalog.json)保留兼容 ID。HLC 的 `sequence_id` 称月份集合（calendar-month bundle）；原 `hr_*` / `lr_*` 分别解释为 fine / coarse。以下 2026-09-09 实施进度属于历史记录。

2026-09-09。范围：HLS-LST-Pairs（HLP，读取 ID `prithvi`）与 HLS-LST-CrossSensor（HLC，读取 ID `lstsr_tb`）。用户已确认：**不合并训练，两份数据独立处理，但处理方法相同。** 统一接口、全量扫描、四组训练集归一化参数与读取验收已完成；保留原划分，尚未启动模型训练。当前实验先做无人工噪声的 VAE 自重建，未来噪声模块接在同一流程之后。实际完成状态见 `materialization_summary.json` 和 `MATERIALIZATION_RESULT.md`。

方法配置共用，数据成员、划分和统计参数各自独立。第一轮每份数据各训练 HLS/LST × 普通 VAE/Prithvi 架构 VAE，共 8 组。每份数据的两个 HLS 模型共用一组 HLS 统计，每份数据的两个 LST 模型共用一组 LST 统计，因此是 4 组归一化参数，而不是每个模型单独拟合。

## 1. 两个数据源，各自包含光学和温度模态

| 数据源 | 光学 HLS | fine-resolution LST | coarse-resolution LST | 样本单位 |
|---|---|---|---|---|
| HLP | `[6,256,256]`，六波段 NPY | `[256,256]`，NPY | 无 | 一次观测的一个 patch |
| HLC | 每个月份集合 一张 `[6,256,256]` | NPZ 存储 `[12,256,256]` | NPZ 存储 `[12,8,8]` | HLS 按月份集合；LST 按可用 `(sequence,month)` |

HLP 有 242,416 个 patch。HLC 有 277,808 个 月份集合/HLS 样本记录，manifest 声明 2,892,007 个可用 LST patch-month。后者 12 个日历槽位允许来自不同年份，不自动视为连续 12 帧。

上传目录是归档交付包；当前 HLC 读取根目录应为 `data_inspection/extracted_metadata/lstsr_tb_final_dataset_20260719`。它与上传 ZIP 是同一份数据的两种存储形态，不应重复计样本。当前以解压后的 manifest 为划分入口，因为它已建立 train/val/test；归档内历史 split 名称可能不同。

第一轮自重建的 LST 指fine-resolution LST。8×8 的coarse-resolution LST 保留给后续超分辨率/条件重建任务，不能通过上采样充当另一批 fine LST 参考。fine LST 是卫星反演参考产品，不能称为无误差地面真值。

## 2. 统一顺序

原始只读数据 → 数据源适配器 → 标准样本清单 → 固定 split → 原始编码解码与参考 mask → 训练集归一化 → 统一增强/可选噪声 → 模型布局适配 → 统一损失与评价。

先固定数据成员、split 与 sampler，再拟合统计量。各模型从同一配置和统计文件读取参数，禁止分别在训练脚本中实现自己的 preprocessing。

## 3. 两个适配器与一个样本接口

- HLP 适配器（历史名称 PrithviAdapter）：按实际 patch 文件清单读取 bands、Fmask、LST 与 metadata；不使用 download_progress 的分项计数充当训练索引。
- HLC 适配器（历史名称 LSTSRAdapter）：以 phase4_dataset_manifest 与 hls_condition_manifest 对应 sequence_id；LST 只展开 available_months，进一步检查有效 mask。不能仅凭日期非空加入未通过 availability 的月份。
- 独立 HLS 自重建时，HLC 每个月份集合 的 HLS 记录只取一次，不随该月份集合 的可用月份重复 6–12 次。后续 paired 任务若每个月重复使用静态 HLS，需要另设任务 profile，并显式记录其权重。
- 仅在有训练/评价所需参考像元时纳入相应模态；某一模态无效不能默默改变另一模态独立训练的数据成员。任何剔除都写入拒绝清单。

共同输出字段：

| 字段 | 约定 |
|---|---|
| dataset_id | 兼容 ID `prithvi`（HLP）或 `lstsr_tb`（HLC） |
| sample_id | 数据源命名空间内稳定唯一 ID，包含必要的 patch/month 信息 |
| modality | `hls`、`lst_hr`（fine）；未来扩展 `lst_lr`（coarse） |
| image / target | float32 `[C,256,256]`，HLS C=6，fine LST C=1 |
| reference_mask | 原始参考可信性，能广播到全部通道 |
| input_mask | 本次输入可用性，与参考 mask 分开 |
| corruption_mask | 人工污染区域；当前自重建全部为 false |
| split / tile_id | 实验有效划分与标准化后的 tile 标识；同时保留 source_split |
| provenance | 源文件、来源观测、原始时间、HLS 时间、source scale、月份、质量版本 |

时间或地理信息仅因记录在 provenance 中，并不自动成为模型输入；任何额外条件都需要在实验 profile 中对所有对照明确约定。Prithvi 架构需要时间维时只增加长度 1 的维度，不暗中改变像元值、分辨率或数据成员。

## 4. 源编码与有效性必须分别解释

| 项目 | HLP | HLC |
|---|---|---|
| HLS 波段 | Blue, Green, Red, NIR, SWIR1, SWIR2 | B2, B3, B4, B5, B6, B7；映射到同一顺序 |
| HLS 解码 | raw / 10000 | raw × 0.0001 |
| HLS 有效性 | 排除 Fmask fill=255、bit1/2/3，并结合数值填充值 | hls_valid AND raw 非 nodata，随后应用同一 Fmask fill/bit1/2/3 规则 |
| LST 原始填充 | 已观察到 -32767；需保留该源明确的填充值配置 | manifest 声明 -32768，并结合 hr_valid/lr_valid |
| LST 单位/解码 | 摄氏度 × 100 的 int16 存储编码；metadata `lst_scale=100`，已用原始 Landsat 像元抽样验证，恢复 °C 时除以 100 | fine LST 同样按摄氏度 × 100 解码，已用 manifest 指定的原始 Landsat 场景抽样验证，恢复 °C 时除以 100 |

先识别原始填充值，再转 float 和解码。原始可靠像元中雪、水、零值和真实温度极值不自动当成噪声。不新增逐图 min-max 或分位数 clipping；HLS 原来是否已裁剪属于数据源记录，不由某个模型单独补偿。

没有独立 LST QA 的来源，需要明确参考 mask 的证据范围，不能把 HLS 的云质量位直接解释为另一时刻 LST 的真实污染位置。

2026-09-09 用户补充：**LST 使用 Landsat 的温度换算公式。** 本次核对并记录的标准公式为 Landsat Collection 2 Level-2 Surface Temperature：

```text
T_K = DN * 0.00341802 + 149.0
T_C = T_K - 273.15 = DN * 0.00341802 - 124.15
```

依据：[USGS Landsat Collection 2 Surface Temperature](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature)。这里的 DN 是原始 Landsat ST 波段编码；Landsat 8/9 对应 ST_B10。用户本次未另行指定 Collection/Level，以上产品版本是本次核对公式的适用范围。

后续核验已将“可能”推进为有独立像元证据支持的单位判定：**两套数据的 fine LST 均按摄氏度 × 100 的存储编码解码。** HLP 全部 242,416 条 metadata 都记录 `lst_scale=100`；抽取两个 patch、18 个预先固定位置，与公开 Landsat ST 波段经官方公式换算的温度对照，最大绝对差 0.015749°C。HLC 抽取同一月份集合 的两个不同月份、18 个预先固定位置，17 个取到原始温度，最大绝对差 0.004857°C；另 1 个原始值未取得，不计入一致性分母。该检查验证单位与尺度，不声称恢复了导出代码中的舍入/插值细节，也不是全量像元对照。见 [单位核验记录](LST_ENCODING_VERIFICATION.md)、[原始响应与统计](lst_encoding_verification.json) 和 [可复现检查](lst_encoding_verification.ipynb)。

因此本地 NPY/NPZ 的有效存储值 `s` 使用 `T_C = s / 100.0`、`T_K = s / 100.0 + 273.15`。Landsat DN 公式只用于上游原始 ST 波段，不能重复应用于本地温度编码。先用来源 fill/mask 排除无效像元，再换算。HLC manifest 的 `scale_factor=32` 是任务数组边长比 256/8，不是温度尺度；它的 coarse LST 来源是 MODIS，不能对其套用 Landsat 原始 DN 公式。现有 coarse LST 样本量化特征也支持摄氏度 × 100，当前 VAE 基线仍只使用 fine LST。

用户在核验后已确认将此结论记为后续实验约定：两套数据的 fine LST 统一按摄氏度 × 100 解码；所有模型共用相同的单位恢复与有效 mask 规则。

已冻结的 v1 读取代码与统计文件保留原始编码空间，`image`/`target` 是 z-score，`denormalize()` 恢复存储值。因此接口恢复摄氏度的调用是 **`ds.denormalize(sample['image']) / 100.0`**，并使用 `reference_mask` 屏蔽无效位置。不能直接把已经标准化的 `image` 除以 100 当作摄氏度。冻结统计文件的 `physical_temperature_unit_confirmed=false` 是全量构建时的历史状态，当前物理单位解释以本节及核验记录为准。

因为 `(s/100 - mean_s/100)/(std_s/100) = (s-mean_s)/std_s`，两者给出相同训练输入（浮点舍入误差除外）。因此无需重跑全量预处理或改变已冻结的数据包。未来物理单位评价统一在反归一化后除以 100：MAE/RMSE 从存储单位换算为 °C 也除以 100，MSE 除以 10000；有效 mask 与指标权重保持不变。

## 5. 归一化规则共用，参数归属明确

统一采用可逆的固定 z-score：HLS 每个波段独立，LST 每个模态独立。统计只来自最终 train 的有效参考像元；验证/测试复用固定参数。填充值和缺失月份不参与统计。

若使用均匀样本采样和样本等权损失，统计口径也采用样本等权：先算每张图有效像元的均值/二阶矩，再按训练 sampler 的样本或数据源权重汇总。统计文件必须写清权重口径。

- 按用户确认的独立训练方案，以 `(dataset_scope, modality, split_version)` 保存参数；同一数据源、同一模态下的普通 VAE 与 Prithvi 架构 VAE 必须共用完全相同参数。两份数据使用相同公式，各自在 train 上拟合；不把两个来源混在一起计算统计。
- 未确认物理单位时，可以在各自来源内做原始编码空间标准化以运行独立自重建；评价必须标明编码或归一化单位，不能擅自标成摄氏度。
- 无效位置在归一化后填 0，并保留 mask。模型输出空间和反归一化函数统一；不允许某个模型自行追加 sigmoid/tanh 与值域裁剪。

当前已有的抽样统计用于诊断，不是本流程的最终训练常数。

## 6. 两份数据各自保留划分，跨来源隔离另作记录

两套数据共有 1,767 个目标 tile ID。HLP 源 train 与 HLC 源 test/val 分别重叠 87/90 个 tile；HLP 源 val（测试）与 HLC 源 train/test 分别重叠 80/4 个 tile。见 source_audit.json。

当前已确认的独立训练、各自内部评价不合并两来源，不因这些交叉重划原始数据。两份数据独立初始化模型、独立拟合统计、不交换训练样本，结果分别报告。

保留冻结 split 与成员。HLP 已从旧训练池划出147个tiles、11,684对作为独立验证；新训练218,975对、固定测试11,757对。新版本 `prepared="processed_data/hlp_split_v2_20260914"` 直接使用 train/val/test，新归一化排除了验证贡献；旧v1的源val仍表示测试。详见[划分协议](../../data_inspection/hlp_split_20260914/README.md)。不得用测试集选模型。HLC 源 train/val/test 分别为训练/验证/测试。历史日志中的 split 标签保留，报告按此映射解释；术语更新不使历史评价自动成为未参与调参的独立测试。

未来若开展跨来源预训练/微调或迁移测试，需另立协议检查全局 tile、源场景和真实地面覆盖重叠。当前两来源的结果不能宣称为地理上独立的迁移测试。

## 7. 同模型比较的冻结项

同一实验 profile 下必须一致：样本清单、split、模态和波段顺序、输入尺寸、有效性规则、归一化、增强及 seed、训练 sampler、固定验证批次、损失 mask 与指标分母。首轮人工噪声和几何增强均关闭，以先核对基础自重建；以后启用几何增强时，由公共模块同步处理 image、target 和所有 mask。雪、水、高气溶胶不额外按像元统一剔除，但保留标签供分组评价；已有来源上游筛选仍作为 provenance 记录。

将 manifest hash、split hash、normalization hash、preprocessing 版本、采样配置及噪声版本写入每个 checkpoint/run 配置。不匹配的运行不能悄悄放入同一对照表。

## 8. 后续实施验收

1. 两个适配器对真实样本给出相同接口，HLS 为 6 通道，fine LST 为 1 通道；缺月不补样本。
2. 无效像元不进入统计/损失；合成遮挡位置在有原始参考时仍进入监督。
3. 归一化可以往返恢复有效原始解码值；统计不受 val/test 内容变化影响。
4. 同 sample/epoch/seed 在不同模型与 worker 设置下产生相同增强和噪声。
5. 清单、参数和处理代码版本一致；全局训练/评价隔离检查与实验范围匹配。

上述数据读取与全量统计验收已完成。正式入口为 `eo_data.open_datasets` / `eo_data.open_dataset`。HLP 当前直接读取 `artifacts/prithvi_data_v1_20260909.zip`，旧解压目录、旧 ZIP 和独立分片已按用户授权清理；必要索引保留在 `processed_data/v1/prithvi`。HLC 复用现有 NPZ。整个新压缩包及全部压缩记录已校验，NumPy 与 Windows spawn 进程读取一致；当前环境未安装 PyTorch，PyTorch 适配器尚未在本环境执行。
