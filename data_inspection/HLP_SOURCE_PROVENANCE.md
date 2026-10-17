# HLP 下载来源与实现核对

更新日期：2026-09-13。用户提供的第一套下载实现实际位于 `C:/Users/rog/Desktop/lst-super-resolution/data_download/`。本次只读取并保存代码，没有运行下载、重新认证GEE或修改冻结数据。

七个Python源文件已逐字节复制到[代码快照目录清单](provenance/hlp_download_20260913/source_manifest.json)，记录SHA-256与仓库提交`5f668490d976f211b192569dd941e9f88d6f4c3b`。该提交时间为2026-04-20；当前数据的下载进度记录早于此快照，因此不声称此提交就是全部242,416条记录的原始运行版本。它是用户指定的生成实现，可确认算法与产品族；逐样本历史追溯仍需原运行清单。

## 确认的产品来源

| 模态 / 辅助数据 | 当前下载实现 | 用途 |
|---|---|---|
| HLS光学 | `NASA/HLS/HLSL30/v002`，`B2`–`B7`、`Fmask` | Landsat来源的六波段光学条件；代码没有查询HLSS30/Sentinel-2 |
| fine LST | `LANDSAT/LC08/C02/T1_L2`与`LANDSAT/LC09/C02/T1_L2`，`ST_B10` | Landsat 8/9 Collection 2 Tier 1 Level-2温度；代码合并两类来源，不意味着每个样本都同时含两卫星 |
| LST QA | 同一候选mosaic的`QA_PIXEL` | 当前代码尝试附带下载；失败也允许保存LST，不参与该下载实现的温度逐像元云过滤 |
| 地表覆盖 | `COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019`，或本地2019年分类栅格 | tile筛选、dominant LULC分层；不是随每个patch交付的逐像元标签 |
| 生态区 | `RESOLVE/ECOREGIONS/2017`，或本地Ecoregions2017 shapefile | tile覆盖补齐；不是温度来源 |

证据：[常量](provenance/hlp_download_20260913/common.py)、[下载实现](provenance/hlp_download_20260913/download_patches.py)、[tile选择](provenance/hlp_download_20260913/select_tiles.py)。代码自述参考Prithvi采样策略，是本项目自行构建配对数据的依据，不是Prithvi官方数据集身份的依据。

## 日期选择、mosaic与空间网格

1. [Phase 2](provenance/hlp_download_20260913/build_sequences.py)按bbox中心点查询HLSL30，每季度至多取10条满足场景云量条件的候选，按日去重，必要时按排序后的日期索引`linspace`取样。[入口](provenance/hlp_download_20260913/run_pipeline.py)设置2014–2026年、每tile至多48个候选观测、场景云量`<20`。有已有CSV时会跳过重建，因此默认参数不独立证明历史执行参数。
2. `metadata.datetime`直接取Phase 2的`observations.csv`时间，记作检索锚点t，其日历日为D。实际HLS在`[D−3天,D+4天)`重新搜索，按场景`CLOUD_COVERAGE`最小选取，不按离t最近选取；`hls_image_id`记录的是此时选中的影像。
3. LST独立在`[D−1天,D+2天)`查询Landsat 8/9。代码将相对t的绝对时间差降序排列，再`mosaic()`。GEE从末景向前优先使用有效像元，因此较近时刻优先，较远候选可填补其掩码空缺；不是多景平均、不是保证单日或单源景的目标。[GEE mosaic定义](https://developers.google.com/earth-engine/apidocs/ee-imagecollection-mosaic)
4. 所有时间窗口先格式化成日期，且右端不包含。HLS与LST各自相对同一锚点搜索，不能将“±1天LST窗口”写成相对**实际所选HLS时刻**的±1天。[filterDate定义](https://developers.google.com/earth-engine/apidocs/ee-imagecollection-filterdate)
5. 下载从选中的HLSL30 `B2.projection()`取得CRS与六参数仿射变换；HLS和LST均以该grid请求`3660×3660`数组，再切成`256×256`不重叠patch。边长只取整到14个patch，边缘不足256像元的条带不输出。行列索引对应**实际HLS网格**。
6. HLS查询没有强制目录tile与scene tile相等，且没有把`proj_info`写入metadata。当前代码没有为LST指定`.resample('bilinear')`或bicubic；按GEE默认规则解释为最近邻重投影。指定同网格是生成设计，不等于已独立测量的几何配准精度。[GEE重采样规则](https://developers.google.com/earth-engine/guides/resample)

**对既有日期统计的修正：**HLP“0天中位数/91.35%同日”的分母和数值不变，但其含义是所选HLS日期与HLS候选检索锚点相比较。它不是HLS–LST采集日期差，不能与HLC实际月份温度日期的728天中位数作同义对照。LST贡献景ID、时间及逐像元来源未保存，现阶段不能从HLP metadata还原真实温度日期差分布。

## 数值处理、质量筛选与格式

- HLS服务器端执行`band × 10000 → int16`，检查每波段超出`[-500,15000]`或六波段全零的比例；超过1%丢弃patch。Fmask bits 0–3并集比例超过20%也丢弃。之后**将保存的反射率整数裁剪到`[0,10000]`**。旧样本检查的0/10000边界与此相容；当前读取器“不额外clip”并不否认上游已clip。
- LST服务器端执行`(ST_B10 × 0.00341802 + 149.0 − 273.15) × 100 → int16`，返回后除以100得到摄氏度。超出`[-50,70]°C`置NaN；tile有效比例低于1%返回无数据；patch缺失超过1%不保存。最终NaN经过填充和整数裁剪，编码为`-32767`，有效温度按摄氏度×100保存。该流程支持既有[单位核验](../experiments/data/LST_ENCODING_VERIFICATION.md)的解释，不需重新套原始DN缩放。
- `QA_PIXEL`在当前代码中单独请求，可失败；未额外依据QA bit位过滤LST。内置产品掩码、数值范围筛选、QA位过滤是不同机制，不能混称为完整云质量保证。
- 当前代码可写五个文件（含可选`qa_pixel.npy`），但本项目冻结包只记录`bands.npy/fmask.npy/lst.npy/metadata.json`四文件。不能把新代码的可选字段追加宣称为旧包的已交付字段；不得改变现有冻结schema或mask来伪造一致性。

## Tile选择、split及实际样本成员

[`select_tiles()`](provenance/hlp_download_20260913/select_tiles.py)先按每个土地覆盖类别的占比top-500抽至多100个tile，再加城市占比top-1000、高Shannon熵top-1000，补充尚未覆盖且候选tile不少于3个的生态区，剔除指定极地区域。之后按dominant LULC分组：每组shuffle，源val数量为`max(1,int(n_class×0.05))`，其余为train。入口默认seed为42；它不是逐patch全局随机95/5划分。

本项目已确定把源val解释为**测试**，无独立validation。同tile内全部记录继承split。算法现已可说明，但缺原`selected_tiles.csv`及其哈希与冻结成员的逐条复核。固定seed也不保证在改变集合迭代顺序、候选数据或代码版本后得到相同成员；优先发布现有冻结清单。

Phase 3每tile最多预留92个patch、同一`(tile,row,col)`最多10条记录；预留后仍可能因LST检查失败而不落盘。待处理观测先shuffle，并发执行及限额会影响实际保留记录；不是保证各年份均匀的采样。

[`finalize.py`](provenance/hlp_download_20260913/finalize.py)另提供同tile行列位置排重，以及训练中海洋/沙漠约10%保留，输出`train_samples.json/val_samples.json/manifest.json`，**不会删除原始patch目录**。当前冻结输入来自完整下载记录，保留的`download_progress.json`总数也为242,416；没有证据证明其采用Phase 4的子集清单。因此该阶段应列为“代码提供的后处理”，而非当前数据已经应用的筛选。它的同tile位置排重也不等于相邻tile footprint无交叠。

## 尚需保留的来源边界

已解决：第一套的产品collection、波段、温度公式、上游反射率裁剪、时间检索/mosaic机制、共用导出网格与分层split算法。

尚未解决：全部历史导出的精确代码版本、原选择/观测CSV与最终清单、实际LST贡献scene ID与采集时间、逐patch地理变换，以及独立的全量配准/QA验证。此处代码仅对应HLP，不能据此替代HLC的原始导出实现。
