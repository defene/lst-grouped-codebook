# HLP 独立验证划分 v2

2026-09-14 完成。固定 seed `20260914`，对旧训练集的 2,943 个 MGRS tile ID 按 `(SHA256(seed:tile), tile)` 排序，前 `round(0.05×2943)=147` 个 tiles 整体进入验证集。比例按 tile 数定义，未按图像数强制配平，也未实施土地覆盖分层。

| 划分 | MGRS tiles | HLS/fine LST 配对记录 |
|---|---:|---:|
| train | 2,796 | 218,975 |
| val | 147 | 11,684 |
| test | 151 | 11,757 |
| 合计 | 3,094 | 242,416 |

三组记录 ID、tile ID 两两无交集；新 train∪val 等于原 train。test 与旧 v1 源 val 的成员、索引内容及排列完全相同。该检查是 tile ID 隔离，不证明相邻 tile footprint 在地理上零重叠。

## 读取当前版本

在项目根目录执行：

```python
from eo_data import open_dataset

prepared = 'processed_data/hlp_split_v2_20260914'
train = open_dataset('prithvi', 'hls', 'train', prepared=prepared)
val = open_dataset('prithvi', 'hls', 'val', prepared=prepared)
test = open_dataset('prithvi', 'hls', 'test', prepared=prepared)
# fine LST 使用 modality='lst'，三份集合均传同一 prepared。
```

`prepared` 指含 `prithvi/` 子目录的版本目录。新版本仅提供 HLP；HLC 继续使用 `processed_data/v1`。冻结读取器默认仍为旧 v1，不能省略新版本的 `prepared` 参数，也不要对仅含 HLP 的新目录调用 `open_datasets()`。

新索引复用 `artifacts/prithvi_data_v1_20260909.zip` 中的原始压缩记录，无需重写或复制大数据包。旧路径与旧 v1 指纹保持完整。记录 key 和原 metadata 的 split 是来源信息；`sample['split']` 和派生 metadata 的 `experiment_split` 是当前划分，`source_split` 保留历史来源。

## 归一化与使用范围

新 mean/std 仅由剩余训练成员贡献。按原样本等权规则，用旧训练集保存的一阶/二阶矩之和减去全部新验证样本重新读取原字节计算的贡献，更新样本数和有效像元数。验证与测试使用相同的新训练统计。计算使用 float64，通常的求和/相减舍入误差仍存在。

旧模型训练已见过新验证成员，因此新划分只支持今后独立训练的验证，不能把旧模型在这些成员上的分数称为独立验证。新实验应从零训练或使用未见过此数据的初始化；旧训练器/配置仍需显式接入此 prepared 版本，不能直接沿用旧验证或选模缓存。

## 证据

- [划分计数、源文件与成员 SHA-256、读取验收](split_summary.json)
- [归一化充分统计量与计算记录](normalization_derivation.json)
- [train 成员](train_records.csv)、[val 成员](val_records.csv)、[test 成员](test_records.csv)
- [train tiles](train_tiles.txt)、[val tiles](val_tiles.txt)、[test tiles](test_tiles.txt)
- [生成脚本](../../experiments/data/create_hlp_validation.py)

生成脚本拒绝覆盖已存在的新版本。全量重新读取了 11,684 条验证配对记录并验证原字节 digest、两种模态的有效像元数；三组两种模态均完成索引完整性核验和实际读取检查。划分按固定 tile 哈希确定，未根据像元值或模型效果选择成员。
