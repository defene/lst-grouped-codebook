# HLS–LST 数据集文档

文档更新：2026-09-14。HLP 新增独立验证划分 v2，数据总成员保持不变；全量规模与分布统计截至2026-09-12，全量图像与超分基线统计完成于2026-09-13。本文说明数据内容、来源、配对、划分和读取约定，原有分布统计保留在附录。

## 1. 数据集命名与概况

采用共同前缀 **HLS-LST** 和描述性后缀作为本项目的统一工作名称。它们是本文的数据标识，不表示已注册的公开数据集名称，也不暗示与某个模型绑定。

| 文档名称 | 简短描述 | 历史名称 / 读取器 ID | MGRS tiles | HLS 样本 | fine LST 样本 | 当前存储 |
|---|---|---|---:|---:|---:|---|
| **HLS-LST-Pairs（HLP）** | HLS 光学与 fine LST 的 patch 配对记录 | Prithvi Data / `prithvi` | 3,094 | 242,416 | 242,416 | 无损压缩包115.61 GiB |
| **HLS-LST-CrossSensor（HLC）** | MODIS coarse LST 与 Landsat fine LST 的 patch 配对记录，附 HLS 条件影像 | LSTSR-TB / `lstsr_tb` | 2,490 | 277,808 | 2,892,007 | 数据目录403.57 GiB |

HLP为本项目自行构建的HLS–Landsat温度配对数据。用户提供的下载脚本自述参考Prithvi-EO-2.0的tile采样思路；这解释了旧目录名，不表示数据由Prithvi模型生成或由其团队发布。`Prithvi Data`仅作为来源目录与实验兼容别名保留。文档名称调整不更改已有文件路径、读取器 ID、检查点或冻结指纹。2026-09-14 新增的 HLP 验证划分单独保存为 v2 索引与归一化版本，复用原压缩数据；旧 v1 保留供历史复现。HLC 原字段 `sequence_id` 继续作为标识符，本文称其对应对象为 **月份集合（calendar-month bundle）**。

计数单位：HLP 一条记录为一对 HLS/fine LST patch；HLC 一条月份集合含一份 HLS 和6–12个可用 coarse–fine LST 月份对。HLC 的2,892,007指可用 patch-month 对数，不把 coarse、fine 两端重复计数，也不等于独立卫星场景数。两套数据均未按跨数据集内容重复去重。存储体积为文件逻辑大小，压缩包与目录的体积口径不同。

### HLC 后续训练空间裁剪约定（2026-09-30）

后续合成退化实验统一从256×256取固定中心224×224：行列均为`[16:240]`，同一sequence各月份以及所有epoch均保持相同窗口，HLS/fine LST/mask同步，禁用随机翻转。合成coarse LST在裁剪后生成。新入口为`experiments/efdiff_hlc/center_data.py`，机器可读协议见[center_crop_protocol.json](../experiments/efdiff_hlc/center_crop_protocol.json)。原始数据及train/val/test成员不变。此前HLC扩散实验为逐月随机裁剪训练、中心裁剪验证；历史结果和代码不覆盖。新增入口已准备，不表示已启动中心裁剪重训。

## 2. Sensor / Product Provenance

以下区分“下载代码确认”“本地记录可追溯”“抽样支持”和“尚待补证”。第一套下载代码实际位于`lst-super-resolution/data_download/`，已保存[只读代码快照及来源核对](HLP_SOURCE_PROVENANCE.md)。产品官方网站用于解释产品定义；当前代码快照也不替代每个历史样本的源景记录。

| 数据集 / 模态 | 卫星与传感器 | 产品与波段 | 本地来源证据与边界 |
|---|---|---|---|
| HLP / HLS 光学 | **Landsat 8 / OLI、Landsat 9 / OLI-2**的L30产品来源；未逐样本统计两者比例 | **NASA HLSL30 v2.0**，`NASA/HLS/HLSL30/v002`，`B2`–`B7`及`Fmask`；没有使用HLSS30/Sentinel-2分支 | `common.py`固定collection，`download_patches.py`只查询HLSL30；metadata保留选中影像的`hls_image_id`。[HLS产品定义](https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSL30_v002) |
| HLP / fine LST | **Landsat 8 / TIRS、Landsat 9 / TIRS-2** | **USGS Landsat Collection 2、Tier 1、Level-2 Surface Temperature**，`LANDSAT/LC08/C02/T1_L2`与`LANDSAT/LC09/C02/T1_L2`合并后取`ST_B10` | 下载代码明确执行温度转换、按时间距离排序后mosaic并导出到HLS网格；metadata仍缺实际贡献的LST场景ID与时间。另有[单位核验](../experiments/data/LST_ENCODING_VERIFICATION.md)，18个抽样位置差异均≤0.02°C |
| HLP / coarse LST | 无独立实测来源 | 当前交付不含 coarse LST；需要实验时由 fine LST 合成退化生成 | 必须标作 synthetic coarse LST，不能称为 MODIS 配对 |
| HLC / HLS 光学 | **Landsat 8 / OLI**，由上游 `LC08` 选择条件限定 | **NASA HLSL30 v2.0**，`NASA/HLS/HLSL30/v002`，`B2`–`B7`，附 `Fmask` | 全部277,808条 HLS manifest 均指向该 collection；配置 `landsat_product_prefix=LC08`。HLSL30产品整体支持Landsat 8/9，但本数据选择范围只含Landsat 8。[HLS产品定义](https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSL30_v002) |
| HLC / fine LST | **Landsat 8 / TIRS、Landsat 9 / TIRS-2** | **USGS Landsat Collection 2、Tier 1、Level-2 Surface Temperature**；来源ID为 `LANDSAT/LC08/C02/T1_L2` 或 `LANDSAT/LC09/C02/T1_L2`，对应温度波段 `ST_B10` | 月份来源表保留 `landsat_scene_ids`、`primary_scene_id` 与采集时间。目标可由主景及同日辅助景填补组成，并非每个patch-month都只有唯一Landsat源景。[USGS产品说明](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature) |
| HLC / coarse LST | **Terra / MODIS** | **MOD11A1 Collection 6.1**，`MODIS/061/MOD11A1`，日间LST `LST_Day_1km`；附日间QA、观测时间和观测角层 | `modis_image_id` 可追溯至每日产品，数组保留 `lr_qc_day`、`lr_view_time_h_x10`、`lr_view_angle_deg`。不是 Aqua/MYD11A1，也不是由 fine LST 降采样得到。[MOD11A1产品说明](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A1) |

HLS是光学地表反射率产品来源，不是本文 fine LST 的反演来源。Landsat Surface Temperature 是卫星反演参考产品，不应称为无误差地面真值，也不是由 HLS 反射率直接提供的温度波段。

HLP现已确认collection、温度公式、目标网格、时间检索和选tile算法；剩余缺口是历史运行版本/输入清单、逐样本LST贡献场景与时间、以及未写入metadata的CRS和仿射变换。HLC温度来源有同日多景与填补信息，但本次提供的是第一套下载代码，不能用它代替HLC导出实现；HLC精确像元源景归属、全部QA/填补阈值与fine重采样核仍待补齐。

## 3. Coarse / Fine Resolution 与配对关系

### 3.1 分辨率术语

统一使用 **coarse-resolution LST** 与 **fine-resolution LST**，表示本任务中相对粗、细的输入/参考网格。原始字段 `lr_*`、`hr_*` 为兼容保留，文档分别解释为 coarse、fine。

| 内容 | 存储形状 | 网格与物理分辨率的解释 |
|---|---|---|
| HLP HLS / fine LST | `6×256×256` / `256×256` | 下载代码取选中HLSL30的`B2`原生投影与仿射变换，以同一30 m产品网格导出HLS和LST，再同位置裁剪；每patch约7.68 km×7.68 km。实际变换未存入patch metadata，不能按目录tile盲目重建 |
| HLC HLS / fine LST | `6×256×256` / 每月`256×256` | 当前月份元数据的fine网格步长均为30 m；patch覆盖约7.68 km×7.68 km |
| HLC coarse LST | 每月`8×8` | 源MOD11A1为名义1 km产品；配置倍率32，按同范围网格推算目标采样间隔为960 m。它是重投影后的coarse网格，不能描述成MODIS原生960 m观测 |

**30 m产品网格不等于30 m原生热红外观测。** Landsat 8/9热红外波段以100 m采样并重采样到30 m产品网格；本数据的32×是`256/8`的数组边长比，不是独立热信息量提高32倍。[USGS传感器与采样说明](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-landsat-archives-landsat-8-9-operational-land-imager-and)

### 3.2 HLP：光学–温度配对

连接键为原记录路径对应的 `(split, tile, obs_idx, row, col)`；同一冻结记录包含 `bands.npy`、`fmask.npy`、`lst.npy` 和 `metadata.json`。下载代码证明二者请求同一HLS网格并按同一行列裁剪，不代表独立测量的配准精度。

`metadata.datetime`来自Phase 2的HLS候选观测清单，是Phase 3检索锚点，**不是LST采集时间**。以其日历日D为基准，HLS检索`[D−3天, D+4天)`，按`CLOUD_COVERAGE`最小选一景；LST检索`[D−1天, D+2天)`，合并Landsat 8/9，按距离锚点的时间差降序排列后`mosaic()`，使较近时刻的有效像元优先，并允许其他候选填补。窗口按日期截断、右端不包含；不能写成严格同日配对，也不能把mosaic称为多景平均。[GEE日期边界](https://developers.google.com/earth-engine/apidocs/ee-imagecollection-filterdate)与[mosaic优先级](https://developers.google.com/earth-engine/apidocs/ee-imagecollection-mosaic)。

HLS查询按目标bbox的中心点`filterBounds`，没有强制场景tile ID等于目录tile。选中影像的`B2`投影用于两种模态，解释了目录tile与HLS ID可能不一致的生成机制；这仍需逐样本源网格复核。`row,col`定位的是所选HLS网格，不能仅用目录tile推断精确位置。LST导出未显式调用`resample()`，依GEE默认重投影行为解释为最近邻，不能声称实施了双线性导出。[重采样规则](https://developers.google.com/earth-engine/guides/resample)

### 3.3 HLC：真实 coarse–fine 配对与 HLS 条件

一个可用样本由 `(sequence_id, calendar_month)` 标识；该月份对应实际 `date`。其输入/参考关系为：

```text
同一可用月份：Terra/MODIS coarse LST [8,8]
           + 该月份集合共用的 HLS 光学 [6,256,256]
           → Landsat fine LST 参考 [256,256]
```

温度两端按同一日历日选择来源，全量可用月份对应的MODIS产品日期与月份来源日期一致；这不是同时刻观测保证。清单另保留传感器时间差与有效性统计。MODIS导出记录为 `nearest` 重采样到任务网格。fine/coarse各有独立有效mask；同日、同网格设计仍不消除传感器响应、观测时刻、反演误差和剩余配准误差。

每条月份集合只有**一份HLS**，通过 `sequence_id` 与温度文件及manifest连接，并非每个月独立配一份同期HLS。HLC全量HLS与目标温度日期差中位数为728天，71.83%的月份样本超过365天；这说明辅助光学的时相条件，不等于几何错位，也不单独证明误差原因。历史统计中HLP的0天中位数实际比较“选中HLS日期”与“HLS候选检索锚点”，不测量HLS与实际LST贡献景的时间差，因此不能与HLC的728天作同一物理含义的日期比较。详见[原始日期统计](lst_hls_sr_census_20260913/full_date_alignment.json)和[新代码解释](HLP_SOURCE_PROVENANCE.md)。

空间定位使用 `grid_crs`、`grid_transform` 和patch像素偏移 `x0,y0`；`patch_row/patch_col` 是裁剪网格索引。相同MGRS tile ID不保证相同裁剪范围；当前仅核验存储结构、来源元数据和索引一致性，未完成全量独立几何配准误差测量。

## 4. Calendar-month Bundle：不是连续 Time Series

HLP的基本单位是一次光学–温度配对记录。按 `(tile,row,col)` 归组只能得到候选重复位置记录，不应直接称为已配准、连续的时间序列。

HLC的 `sequence_id` 是存储键，`sequence_mode=calendar_month_bundle_independent_same_day_pairs`，`storage_grouping=twelve_calendar_month_bundle_cross_year_allowed`。轴0对应1–12月槽位；每月选择一次实际日期的温度配对，不是月平均，也不是固定年份的12个月。

例如 `T01WCP_v000_p0000` 的3月取自2021-03-30，4月取自2024-04-06，8月取自2022-08-08；月份轴不按实际时间递增。其2月虽在 `month_dates_json` 中有候选日期，却不在该patch的 `available_months` 中，因此不得把“日期非空”作为可用样本条件。

读取时先用 `available_months` 决定可用槽位，再使用fine/coarse有效mask。不要补零后当有效温度，也不要默认“上一槽位”是相邻时间。使用时间模型需显式提供实际日期/时间差、限制可用条件，并检查时序任务所需的时空一致性；当前数据不直接提供严格连续、固定间隔的时序预测基准。

## 5. Train / Validation / Test Split

| 数据集 | 文档划分 | 源split/API标签 | tiles | HLS/记录或月份集合数 | fine LST样本数 |
| --- | --- | --- | --- | --- | --- |
| HLP | 训练 | 源 train → train | 2,796 | 218,975 | 218,975 |
| HLP | 验证 | 源 train → val | 147 | 11,684 | 11,684 |
| HLP | 测试 | 源 val → test | 151 | 11,757 | 11,757 |
| HLP | 合计 | —（汇总） | 3,094 | 242,416 | 242,416 |
| HLC | 训练 | train | 2,249 | 250,449 | 2,608,772 |
| HLC | 验证 | val | 118 | 13,694 | 141,711 |
| HLC | 测试 | test | 123 | 13,665 | 141,524 |
| HLC | 合计 | —（汇总） | 2,490 | 277,808 | 2,892,007 |

**HLP 当前划分（v2，2026-09-14）：**从旧训练集的2,943个MGRS tiles中，以固定种子`20260914`，按`(SHA256("20260914:" + tile_id), tile_id)`升序排列，取前`round(0.05×2943)=147`个tiles作为独立validation。同一tile的全部HLS/fine LST配对整体移动；新训练为2,796个tiles、218,975对，验证为147个tiles、11,684对。原测试的151个tiles、11,757对成员固定。三份集合的记录ID和MGRS tile ID两两无交集，train∪val精确等于旧train；测试索引数组与旧源val逐项相同。[划分协议与读取](hlp_split_20260914/README.md)、[精确计数与校验](hlp_split_20260914/split_summary.json)、[生成脚本](../experiments/data/create_hlp_validation.py)。

新读取版本为`processed_data/hlp_split_v2_20260914`，API split直接使用`train/val/test`。旧`processed_data/v1`仍保持源`train/val`标签，其中旧`val`为测试；原记录路径和压缩包内部metadata的split继续表示来源，派生索引额外记录`source_split/experiment_split`。必须显式传入新`prepared`路径，不能把旧版本的`val`误当新验证集。

新归一化仅拟合剩余218,975条训练记录：使用旧训练集保存的未四舍五入充分统计量，减去对全部11,684条新验证记录从原压缩字节重新计算的一阶/二阶矩之和及有效像元计数，再生成mean/std；沿用原像元mask与样本等权规则。验证、测试共用新训练统计，不继续使用包含新验证成员的旧归一化。[计算明细](hlp_split_20260914/normalization_derivation.json)。这采用float64求和与相减，存在通常的浮点舍入误差。

此划分用于今后的独立训练与选模。旧模型曾在原训练池中见过新验证成员，历史实验也曾监测旧源val（测试成员）；新划分不会追溯消除这些使用记录。新实验应从零训练或使用未见过此数据的初始化，验证用于选模，固定测试用于最终评价。

**HLP 原始训练/测试来源：**现已找到Phase 1算法：先依据2019年Copernicus土地覆盖和RESOLVE 2017生态区选tile，包括类别抽样、城市/高熵区域补充与生态区补齐，再按dominant LULC分层打乱，每类取`max(1, int(n_class×0.05))`个tiles进入源val，其余进入train。当前入口传入`seed=42`。这不是全体patch随机95/5划分；同tile全部记录继承其split。当前脚本与历史导出的逐tile成员尚未用原`selected_tiles.csv`完整重放核对，最终成员以冻结索引为准。[代码与复现边界](HLP_SOURCE_PROVENANCE.md)

HLP另有Phase 4海洋/沙漠约10%下采样及位置排重代码，但该阶段只输出样本清单；未获得证明当前242,416条冻结记录采用了该清单的证据，不能把此筛选写成已应用的数据属性。当前独立验证划分已按上述v2落实；调参使用新validation，不能使用固定test选模型。

**HLC：**原持出集合于2026-09-05由源`val`更名为`test`，成员不变。随后从原训练的2,367个MGRS tiles中，按 `SHA256('20260905:' + tile_id)` 排序，取前118个tiles（`round(0.05×2367)`）作为validation；其余2,249个tiles用于train，原123个test tiles固定。一个tile下所有patch、月份集合和月份样本整体进入同一split。原持出test集合最初的选tile规则尚缺，不能将其描述为同一哈希算法生成。[划分协议](extracted_metadata/lstsr_tb_final_dataset_20260719/splits/README.md)与[精确清单/校验记录](extracted_metadata/lstsr_tb_final_dataset_20260719/splits/split_summary.json)。

两套数据各自的split之间MGRS tile ID不重叠；这不是逐像元空间零重叠证明，也不是跨年份留出。相邻tile边缘重叠、相同源景的跨tile关系需单独核验。两套数据之间共有1,767个tile ID；联合训练/跨数据集测试时不能据此假定彼此独立。

归一化参数仅由各自训练集、各自模态计算，validation/test复用。不得使用测试目标拟合温差校准、选择退化参数或调参。原始split名称保留在统计/索引文件中时，以本节的语义映射解释。

## 6. Dataset Structure / Schema

### 6.1 存储结构与关系表

```text
HLP 原逻辑记录（当前打包为可随机访问的无损 EOZ / ZIP）
  <source_split>/<tile>/obs_<id>/r<row>_c<col>/
    bands.npy | fmask.npy | lst.npy | metadata.json
  当前包：artifacts/prithvi_data_v1_20260909.zip
  当前索引：processed_data/hlp_split_v2_20260914/prithvi/index.sqlite
  历史索引：processed_data/v1/prithvi/index.sqlite（仅train/源val）

HLC 根目录：data_inspection/extracted_metadata/lstsr_tb_final_dataset_20260719/
  arrays/<tile>/<sequence_id>.npz                         # coarse/fine 温度与QA
  phase4_dataset_manifest.csv                            # 一行一个月份集合
  phase3_tile_variant_months_merged.csv                   # 一行一个variant×月份
  hls_l8_conditions_2019_2025_lc08/
    arrays/<tile>/<sequence_id>.npz                       # 单份光学条件
    hls_condition_manifest.csv                           # 一行一个光学条件记录
    hls_condition_config.json
  splits/                                               # 当前成员清单及split协议
```

| 关系表 / 字段 | 含义与连接规则 |
|---|---|
| HLP metadata | `tile, split, obs_idx, row, col, datetime, patch_size, bands, bands_dtype, bands_scale, hls_image_id, has_lst, lst_dtype, lst_scale`；记录路径唯一标识一次配对 |
| HLC `sequence_id` | 月份集合主键；temperature manifest与HLS manifest按此一对一连接；不是连续时间序列认证 |
| `tile_variant_id, month` | 温度来源表连接键；`month`为1–12；每个patch只读取自身`available_months` |
| `mgrs_tile, patch_idx, patch_row, patch_col, x0, y0` | 区域、patch标识和像素偏移；结合来源表的CRS/仿射变换定位 |
| `available_month_count, available_months, missing_months` | patch层面的可用性；`source_available`只描述上一级来源候选，不足以替代patch可用性 |
| `month_dates_json, month_landsat_times_json` | 月份对应的实际来源日期与Landsat时间；不将月份编号当连续时间坐标 |
| `landsat_scene_ids, primary_scene_id, modis_image_id` | 温度两端来源；一份fine参考可能涉及多个Landsat场景 |
| `hls_collection, hls_asset_id, hls_image_id, hls_datetime_utc` | 光学产品、完整资产标识与独立采集时间 |
| `split, archive, schema_version / *_schema_version` | 当前划分、相对数组路径和格式版本；不同manifest的`archive`相对各自目录解释 |

### 6.2 Array Schema

以下为原始存储数组；读取器输出的归一化tensor不是同一编码。HLC字段形状与dtype已读取实际样本确认，完整性沿用全量索引与已有验收；本次没有重做每个辅助QA数组的全量逐值审计。HLP当前下载代码还可选写`qa_pixel.npy`（`[256,256] uint16`），并允许QA请求失败后保存温度；该层不在已冻结四文件格式内，不能据当前代码声称它已随本地数据交付。

| 数据 | 字段 | Shape | dtype | 说明 |
|---|---|---|---|---|
| HLP | `bands.npy` | `[6,256,256]` | `int16` | 六波段光学，通道顺序见下节 |
| HLP | `fmask.npy` | `[256,256]` | `uint8` | 光学质量bitmask |
| HLP | `lst.npy` | `[256,256]` | `int16` | fine LST，填充值`-32767` |
| HLC | `hr_lst` / `hr_valid` | `[12,256,256]` | `int16` / `uint8` | fine LST及有效性；无效温度填充值`-32768` |
| HLC | `lr_lst` / `lr_valid` | `[12,8,8]` | `int16` / `uint8` | coarse LST及有效性；无效温度填充值`-32768` |
| HLC | `hr_st_qa_k_x10`, `hr_flags` | `[12,256,256]` | `uint8` | fine温度QA辅助层；`hr_flags`位定义与QA饱和/填充值约定仍需原导出说明 |
| HLC | `lr_qc_day`, `lr_view_angle_deg`, `lr_view_time_h_x10` | `[12,8,8]` | `int16` | coarse日间QA、观测角及时间辅助层；先核对源编码/填充值再解码 |
| HLC | `hls_reflectance_i16` | `[6,256,256]` | `int16` | 单份光学条件，填充值`-32768` |
| HLC | `hls_valid`, `hls_fmask` | `[256,256]` | `uint8` | 光学有效性及Fmask |
| HLC | `source_time_s`, `candidate_rank` | scalar | `int64` / `uint8` | HLS Unix秒级时间与候选排序序号 |

### 6.3 单位、波段与有效性

- 光学通道固定为 **Blue、Green、Red、NIR、SWIR1、SWIR2**；两套下载来源均对应HLSL30的`B2,B3,B4,B5,B6,B7`。反射率为存储整数乘`0.0001`，当前读取器不额外裁剪。**HLP下载端已执行`clip(0,10000)`**，因此“不额外裁剪”不能解释成上游没有裁剪；HLC不从该第一套代码推定处理方式。
- 两套fine LST采用`stored / 100.0`恢复摄氏度；HLC coarse LST沿用相同摄氏度×100编码解释。原始Landsat DN与MODIS DN各有不同缩放，不能再套用于已经转换后的这些数组。fine单位有抽样源像元核验，coarse另有来源与整数步长证据，见[编码核验记录](../experiments/data/LST_ENCODING_VERIFICATION.md)。
- HLP光学有效条件：六波段无`-9999/-32767/-32768`填值，`Fmask != 255`，且`(Fmask & 14) == 0`。HLC还与`hls_valid`取交集。bits 1–3对应云、邻云/阴影邻域、阴影；零值、雪、水与气溶胶不在当前读取器中额外全部剔除。Fmask只约束光学，不替代LST QA。
- HLP温度按`lst != -32767`；HLC分别使用`hr_valid & (hr_lst != -32768)`和`lr_valid & (lr_lst != -32768)`。任务评分使用实际所需模态的有效支持；插值还需有效coarse邻域，不能把所有数组像元当有效配对。
- HLC上游HLS最终保留规则为`valid_fraction >= 0.98`且`snow_fraction + water_fraction < 0.98`。这是记录级筛选，不意味着每个保留patch完全无雪、无水或没有无效像元。不能直接把该规则归给HLP。

HLP下载端另外按每波段缺失比例≤1%、patch的Fmask bits 0–3并集比例≤20%筛选；fine温度只保留`[-50,70]°C`内数值，并要求patch缺失≤1%。这些是下载代码中的记录级筛选，与当前读取器的像元mask分开解释。代码下载`QA_PIXEL`但未用其bit位额外去除LST云/阴影；“温度落在范围内”不等于经过完整LST质量认证。

### 6.4 读取器与任务样本的区别

现有`eo_data`入口按单模态返回归一化`image/target`及mask，当前`target`是自重建目标；LST模态展开可用fine月份。它**不是已经封装好的**`(coarse LST, HLS, fine LST)`三元组加载器。构建引导任务时，需要按上述键读取对应coarse数组和HLS，并保持split、月份与mask同步。

```python
from eo_data import open_dataset
prepared = 'processed_data/hlp_split_v2_20260914'
ds = open_dataset('prithvi', 'lst', 'val', prepared=prepared)  # HLP独立验证集
# 测试集使用 split='test'；新训练集使用 split='train'，均传同一prepared。
sample = ds[0]
temperature_c = ds.denormalize(sample['image']) / 100.0
valid = sample['reference_mask']           # 无效位置仍必须排除
```

`image`已做z-score，不能直接除以100当温度。接口与存储恢复方式详见[统一读取说明](../eo_data/README.md)；其中旧v1的`prithvi/val`为测试，新v2的`prithvi/val`为独立验证；按prepared版本区分。

## 7. Supported Tasks

| 任务 | HLP | HLC | 使用条件 |
|---|---|---|---|
| 光学引导LST空间细化：coarse LST＋HLS → fine LST | 可构造合成coarse输入 | 可使用真实MODIS或受控合成coarse输入 | 统一退化、网格和mask；真实跨传感器与合成任务分开报告 |
| 仅LST的空间细化：coarse → fine | 需合成退化 | 真实与合成均可 | fine是参考产品，不是无误差真值；倍率明确为网格倍率 |
| HLS或LST的表示学习、压缩、自编码重建 | 支持 | 支持 | 按各自训练split拟合；避免HLC重复展开月份时把同一HLS误算成独立观测 |
| 光学条件温度估计：HLS → fine LST | 可做配对回归实验 | 可做跨时相条件估计实验 | 日期不严格一致；不包装为已验证的同步温度反演基准 |
| 掩码重建、人工缺失/噪声鲁棒性 | 可构造 | 可构造 | 只在原本有效位置合成并保留未扰动参考；原生缺失区没有自动获得真值 |
| 按月份/年份/区域分析与不规则多观测建模 | 重复位置记录有限 | 可使用实际日期的月份集合 | 不宣称直接支持连续时间预测、严格连续变化检测或未来预测基准 |

没有地面实测温度、土地覆盖分类标签或严格连续时序标签时，不把相关任务写成现成监督基准。当前全量超分基线是固定32×方法的诊断，不是数据集固有难度常数；模型、训练样本量、时间条件或退化变化后需重新评价。

## 8. Evidence、版本与待补信息

当前统计以冻结索引、最终保留manifest及全量验收为准；旧合并/下载文档中的2,517个tiles等中间数量不代表当前HLC。以下资料均保留原兼容名称：

- [规模与分布原始统计](final_dataset_stats.json)：全量记录、月份、地理分组与历史各split计数；其中HLP旧源`val`对应test，旧train尚未拆出新validation；当前HLP split计数以[v2划分记录](hlp_split_20260914/split_summary.json)为准。
- [本次来源与schema核对](dataset_documentation_audit.json)：当前HLC全量manifest的collection、同日关系、网格、重采样记录，以及一个实际bundle的数组schema。
- [HLP下载代码来源核对](HLP_SOURCE_PROVENANCE.md)、[代码快照与SHA-256](provenance/hlp_download_20260913/source_manifest.json)：产品、采样、split、日期锚点、mosaic、网格与上游质量处理。
- [HLC HLS构建配置](extracted_metadata/lstsr_tb_final_dataset_20260719/hls_l8_conditions_2019_2025_lc08/hls_condition_config.json)、[HLC split协议](extracted_metadata/lstsr_tb_final_dataset_20260719/splits/README.md)。
- [LST单位核验](../experiments/data/LST_ENCODING_VERIFICATION.md)、[HLP已有来源配对检查](prithvi/source_alignment.json)。
- [全量温度复杂度统计](lst_sr_census_20260913/REPORT_ZH.md)、[coarse LST＋HLS全量评估](lst_hls_sr_census_20260913/REPORT_ZH.md)、[全量索引/光学一致性验收](lst_hls_sr_census_20260913/coverage_audit.json)。历史结果中的`synthetic_lr/native_lr/hr`字段分别按合成coarse/真实coarse/fine解释。

尚待补齐：HLP历史执行版本与选tile清单、实际LST贡献景/时间及逐patch地理变换；HLC最初持出test的选择规则与完整温度导出/QA说明；两套的全量独立配准与跨split footprint重叠核验。HLP下载算法与候选产品来源已由新代码确认，不再列为完全未知。派生数据集自己的作者署名、许可、版本发布与引用标识尚待整理，不从源产品许可或历史模型名推定。

## 附录A. 全量空间分布

按目录/清单MGRS tile编码的纬度带归组，绝对纬度带边界统一为32°和64°，不同于旧HLC按中心纬度30°/60°的分组。该统计不是精确patch边界或国家覆盖范围；MGRS规则见[NGA官方说明](https://earth-info.nga.mil/?action=coordsys&dir=coordsys)。

| 数据集 | 绝对纬度带 | tiles | HLS样本 | fine LST样本 |
| --- | --- | --- | --- | --- |
| HLP | 0°–32° | 1,391 | 110,093 | 110,093 |
| HLP | 32°–64° | 1,469 | 117,666 | 117,666 |
| HLP | ≥64° | 234 | 14,657 | 14,657 |
| HLC | 0°–32° | 1,063 | 102,855 | 1,067,426 |
| HLC | 32°–64° | 1,247 | 166,875 | 1,764,632 |
| HLC | ≥64° | 180 | 8,078 | 59,949 |

| 数据集 | 半球 | tiles | HLS样本 | fine LST样本 |
| --- | --- | --- | --- | --- |
| HLP | 北半球 | 2,428 | 191,638 | 191,638 |
| HLP | 南半球 | 666 | 50,778 | 50,778 |
| HLC | 北半球 | 1,987 | 225,859 | 2,341,943 |
| HLC | 南半球 | 503 | 51,949 | 550,064 |

HLP覆盖60个UTM分区，HLC覆盖58个。共有1,767个相同tile ID，跨数据集去重后为3,817个tile ID；相同ID不代表相同patch或时相。HLP每tile含1–92条记录，均值78.35；HLC每tile含1–196个月份集合，均值111.57。HLC清单中心纬度范围为52.86°S–78.80°N；HLP缺少可直接核验的精确坐标字段，不报告精确纬度端点。

## 附录B. 年份与日历月份分布

HLP HLS按场景ID日期统计；`metadata.datetime`经下载代码确认是HLS候选观测的检索锚点，不是LST贡献景采集时间。HLC HLS按自身采集时间，温度仅按可用月份的实际来源日期统计。

| 年份 | HLP HLS | HLP检索锚点 | HLC HLS | HLC fine LST |
| --- | --- | --- | --- | --- |
| 2014 | 16,916 | 16,916 | 0 | 0 |
| 2015 | 17,422 | 17,422 | 0 | 0 |
| 2016 | 15,480 | 15,480 | 0 | 0 |
| 2017 | 14,083 | 14,083 | 0 | 0 |
| 2018 | 14,792 | 14,792 | 0 | 0 |
| 2019 | 15,197 | 15,197 | 45,735 | 178,032 |
| 2020 | 14,779 | 14,779 | 42,688 | 163,634 |
| 2021 | 16,408 | 16,316 | 36,675 | 225,980 |
| 2022 | 29,182 | 29,274 | 48,513 | 603,165 |
| 2023 | 27,474 | 27,474 | 31,436 | 644,785 |
| 2024 | 27,651 | 27,661 | 41,572 | 554,132 |
| 2025 | 29,575 | 29,565 | 31,189 | 522,279 |
| 2026 | 3,457 | 3,457 | 0 | 0 |
| 合计 | 242,416 | 242,416 | 277,808 | 2,892,007 |

HLP记录日期范围2014-01-01至2026-02-14，2026年仅含年初；HLC HLS为2019-01-01至2025-12-29，fine LST为2019-01-01至2025-12-31。不同月份允许来自不同年份，HLS不保证与全部月份同期。

| 月份 | HLP HLS | HLP检索锚点 | HLC HLS | HLC fine LST |
| --- | --- | --- | --- | --- |
| 1月 | 17,168 | 17,252 | 8,423 | 197,646 |
| 2月 | 16,538 | 16,536 | 12,192 | 234,546 |
| 3月 | 20,220 | 20,190 | 10,113 | 251,264 |
| 4月 | 24,891 | 24,684 | 25,815 | 257,714 |
| 5月 | 21,578 | 21,664 | 44,946 | 259,266 |
| 6月 | 18,704 | 18,855 | 72,287 | 255,967 |
| 7月 | 23,880 | 23,813 | 29,109 | 236,549 |
| 8月 | 22,885 | 23,091 | 21,792 | 246,583 |
| 9月 | 20,774 | 20,291 | 16,273 | 254,489 |
| 10月 | 21,477 | 21,821 | 9,975 | 258,359 |
| 11月 | 17,910 | 17,910 | 20,166 | 235,396 |
| 12月 | 16,391 | 16,309 | 6,717 | 204,228 |
| 合计 | 242,416 | 242,416 | 277,808 | 2,892,007 |

以上均为patch样本计数，不是去重后的卫星场景数；同一场景的多个patch分别计数。

## 附录C. 重复位置记录与月份可用性

HLP按`(tile,row,col)`归组得到232,156个候选空间键，96.02%只有一次观测。该键不含完整地理变换，不能直接认定为已构造或严格配准的连续时间序列。

| HLP同一空间键记录数 | 空间键数 |
| --- | --- |
| 1 | 222,906 |
| 2 | 8,400 |
| 3 | 711 |
| 4 | 120 |
| 5 | 17 |
| 6 | 2 |

| HLC可用月份槽位数 | 月份集合数 | 占比 |
| --- | --- | --- |
| 6 | 2,664 | 0.96% |
| 7 | 2,873 | 1.03% |
| 8 | 1,204 | 0.43% |
| 9 | 61,685 | 22.20% |
| 10 | 74,480 | 26.81% |
| 11 | 72,509 | 26.10% |
| 12 | 62,393 | 22.46% |

HLC平均每个月份集合含10.41个可用月份，全部3,333,696个存储槽位中86.75%可用。12个槽位均可用的集合为62,393个（22.46%）；“槽位完整”不表示同一年连续12个月。每个月份集合固定一份HLS。

## 附录D. HLC 合成退化 pilot 固定子集（2026-09-15）

用户选择先在固定子集上短程比较 EFDiff-x₀ 与 EFDiff-ε。该子集属于实验资产，不改变 HLC 原 train/val/test 划分或本报告的全量计数。协议为 `efdiff-hlc-synthetic-pilot-16384-512-v1`；读取仍显式使用 `prepared="processed_data/v1"`。

| 用途 | 有效 fine-LST patch-month | 覆盖原 split 的 tiles | 每 tile 样本数 |
| --- | ---: | ---: | ---: |
| pilot train | 16,384 | 2,249 | 6–8 |
| pilot val | 512 | 118 | 4–5 |

使用 seed=20260915，按 tile 尽量等额分配，样本不足的 tile 受容量限制，余量按固定随机顺序分配；tile 内月份样本无放回抽样。训练与验证的 record_id、tile 均无交集。清单保留原索引、record_id、月份槽位和 tile；成员哈希见[子集记录](../experiments/efdiff_hlc/pilot_membership/summary.json)。同一月份集合的 HLS 可被多个月份复用，不保证与每个月份同期。

输入为 fine LST 经合成 Wald ×32 退化得到的 coarse LST，加关联 HLS；真实 MODIS coarse LST 不参与该实验。新的 condition/residual 统计只使用选中的16,384个训练月份拟合；验证共用训练统计，test不参与。每模型5个子集epoch、有效batch64，共1,280次更新和81,920次月份样本呈现。主对照固定第5轮EMA，并报告bicubic基线、有效像素RMSE/MAE和同样本图。该pilot不能表述成全量或已收敛的论文复现；运行状态与配置见[实验说明](../experiments/efdiff_hlc/README.md)。

### 后续全量1轮实验（2026-09-16）

pilot完成后，用户要求两模型尝试1个完整全量训练epoch。该新实验恢复使用HLC原train的全部2,608,772条有效月份样本（250,449份HLS/月集合、2,249 tiles），每条呈现1次；每模型40,763次更新，最后一次4条。去噪器从头初始化，Prithvi继续使用冻结预训练权重，新的condition/residual统计由完整train拟合。验证仍使用上述固定512条val，test不参与；此处“全量”仅指训练成员，不表示已评估完整val/test。数据版本和成员划分不变，预算及运行状态见[全量1轮登记](../experiments/efdiff_hlc/epoch1_run_manifest.json)。

用户追加要求一天内完成，截止为2026-09-18 04:23 UTC。单卡准备版本由每模型两张H200的DDP版本接替；保留完整训练成员、每条一次、有效batch64和最后4条的准确加权，数据版本与归一化拟合范围不变。当前状态以[一天期限运行登记](../experiments/efdiff_hlc/deadline_run_manifest.json)为准，原全量统计分片继续复用，测速和smoke结果不作为正式训练结果。
