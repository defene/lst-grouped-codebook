# LST 存储单位核验

2026-09-13来源补充：用户现已提供第一套HLP的下载代码，确认其使用HLSL30 v2与Landsat 8/9 Collection 2 Level-2 `ST_B10`，并在导出时转换为摄氏度×100。下文“未找到原始导出代码”描述的是本次单位抽样核验当时的材料范围；现状与实现细节见[下载来源核对](../../data_inspection/HLP_SOURCE_PROVENANCE.md)。此补充不改变原抽样结果，也不宣称已取得每个历史样本的LST贡献场景ID。第二套HLC的完整导出代码仍未由此次材料补齐。

结论：两套数据当前用于 VAE 的 HR LST 支持按 **摄氏度 × 100** 的整数编码解释，恢复摄氏度使用 `stored / 100.0`。这次结论有原始卫星像元对照证据，超出了只凭数值范围猜测的阶段。

## 独立验证方法与范围

- Prithvi 全部 242,416 条 metadata 的 `lst_scale` 都是 100。选择 HLS ID 与目标 tile 一致、行列为 r00_c00 的两个 patch，用同 tile 的已交付网格计算像元中心，不通过拟合偏移改变坐标。每个 patch 预先固定 3×3 个位置，共 18 个。
- Prithvi metadata 没有记录 LST 场景 ID。首先由 HLS 时间寻找 Landsat 场景；第二个 patch 的该场景在所有检查点都无 ST 值，于是检查同日、同卫星、同轨道的相邻 WRS 行。保留初始无值响应及后续结果。这个过程提供独立数值佐证，不等于从 metadata 直接确认了完整 LST 来源链。
- LSTSR 使用 manifest 明确给出的网格、patch 偏移和 Landsat 场景列表。检查 `T01WCP_v000_p0000` 的 4 月、8 月槽位，分别对应 2024-04-06 和 2022-08-08；每月预先固定相同 3×3 个位置。缺失场景与无效原始像元均保留在记录中。
- 从 Microsoft Planetary Computer 的公开 Landsat Collection 2 ST 波段读取原始 DN，`unscale=false`；自行应用 `DN * 0.00341802 + 149.0 - 273.15`，与本地存储值除以 100 的结果比较。公式依据 [USGS 官方说明](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature)。

| 数据 | 预选有效位置 | 能取得源 DN 的位置 | 差异 ≤0.011°C | 差异 ≤0.02°C | 最大绝对差 |
|---|---:|---:|---:|---:|---:|
| Prithvi，2 个 patch | 18 | 18 | 17 | 18 | 0.015749°C |
| LSTSR，2 个 patch-month | 18 | 17 | 17 | 17 | 0.004857°C |

0.011°C 检查约一个存储整数步长，0.02°C 用于容纳很小的插值/量化差异、判断单位是否一致。Prithvi 有一个位置超过前者，原因未从导出代码核实，不能报告为逐位一致。LSTSR 有一个位置未取得原始值；公开镜像中 `LC08_L2SP_091013_20220808_02_T1` 返回 404。原始导出代码未在当前交付工作区找到；这里是单位的抽样核验，不能替代全量来源一致性证明。

两个直观例子：

| 数据与位置 | 本地存储值 | 除以 100 | 原始 Landsat DN | 按 USGS 公式换算 |
|---|---:|---:|---:|---:|
| Prithvi `T03VXK/obs_0014/r00_c00`，像元 `[128,128]` | -2028 | -20.28°C | 30387 | -20.286626°C |
| LSTSR `T01WCP_v000_p0000`，4 月 `[128,128]` | -868 | -8.68°C | 33782 | -8.682448°C |

## 其他证据

两套数据已冻结的训练集样本等权均值分别为 2539.23749、2709.74793 个存储单位；除以 100 后分别为 25.39237°C 和 27.09748°C。这只是单位解释的辅助证据，不用于单独证明单位。

LSTSR 的 LR 来自 MODIS/061/MOD11A1，其原始温度比例是 0.02 K/DN，见 [官方数据目录](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A1)。既有 2,560 个 sequence 的确定性样本中，1,693,306 个有效 LR 像元的整数值全部是奇数，正好符合 `100 * (0.02 * MODIS_DN - 273.15) = 2 * MODIS_DN - 27315` 的量化特征。它支持 LR 同样保存为摄氏度 × 100，但当前训练范围仅为 HR；不要把 Landsat 原始 DN 公式用于 MODIS。

## 当前统一接口如何用

```python
import numpy as np
from eo_data import open_dataset

ds = open_dataset('prithvi', 'lst', 'train')  # 另一套使用 'lstsr_tb'
sample = ds[0]
temperature_c = ds.denormalize(sample['image']) / 100.0
temperature_c = np.where(sample['reference_mask'], temperature_c, np.nan)
```

`image` 已是 z-score，不能直接除以 100 当温度；无效位置也要保持 mask。当前 `denormalize()` 返回存储编码值。v1 文件和代码没有在这次单位核验中更改，冻结统计中的未知单位标志是构建时的历史状态。

物理单位恢复与训练标准化分开解释：将存储值、训练均值和标准差同时除以 100，z-score 不变。因此不需要重跑全量处理。物理温度 MAE/RMSE 为存储单位误差除以 100，MSE 除以 10000；沿用原来的有效像元与样本权重。

## 复查入口

- [检查脚本](check_lst_encoding.py)：读取原文件、已冻结统计及公开原始场景，输出逐点结果。
- [逐点结果 JSON](lst_encoding_verification.json)：含请求参数、原始响应、场景 ID、源文件 SHA-256、失败记录与分母。
- [检查 notebook](lst_encoding_verification.ipynb)：展示结果，并提供重新运行同一检查的入口。

在工作区运行 `python experiments/data/check_lst_encoding.py` 即可重做检查。公开数据服务的未来变化可能导致来源不可用，失败会保留在结果中。
