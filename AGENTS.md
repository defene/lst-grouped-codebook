# 项目数据约定

## HLC 空间裁剪（2026-09-30 用户确认）

- 后续 HLC 合成退化实验使用固定中心裁剪：256×256 输入统一取 `[16:240, 16:240]`，输出224×224；同一 sequence 的各月份以及所有 epoch 均保持不变。
- HLS、fine LST、mask 共用裁剪窗口；取消随机翻转，合成 coarse LST 从裁剪后的 fine LST 生成。
- 新训练使用 `experiments/efdiff_hlc/center_data.py` 中的 `HLCPairs` 和 `check_stats`，显式 `augment=False`；完整入口约定见 `experiments/efdiff_hlc/center_crop_protocol.json`。记录新协议与代码哈希，用新 run ID。
- 历史 `data.py`、训练器、配置和检查点保持冻结；旧结果仍标为逐月随机裁剪训练。不要把历史权重改标为中心裁剪训练，也不要直接改旧启动命令后覆盖旧 run。
- 这次变更仅准备新数据入口，不代表已重新训练。原始数据、划分及已核验的全 train 中心裁剪归一化参数不变。

本文件适用于整个项目。以下为用户于 2026-09-14 确认并落实的当前约定；后续新实验、数据迁移、报告与上传准备均应沿用。

## 名称与证据

- 数据集统一称 **HLS-LST-Pairs（HLP）**、**HLS-LST-CrossSensor（HLC）**。旧读取 ID 分别为 `prithvi`、`lstsr_tb`，保留兼容。
- 温度网格使用 coarse-resolution / fine-resolution LST；原 `lr_*` / `hr_*` 字段保留。HLC 的 `sequence_id` 对应月份集合（calendar-month bundle），不默认视为连续时间序列。
- Prithvi-EO-2.0 只在模型或来源说明中作为模型名使用；不能将 HLP 描述成该团队发布或模型生成的数据。
- 当前数据事实见 [数据集报告](data_inspection/FINAL_DATASET_SUMMARY.md)；机器可读版本与路径见 [dataset_catalog.json](experiments/data/dataset_catalog.json)。旧计划、日志和归档说明按其历史版本解释。

## HLP 当前默认实验划分

- 当前版本为 `hlp-tile-validation-v2-20260914`。新实验显式传入 `prepared="processed_data/hlp_split_v2_20260914"`，API split 使用 `train` / `val` / `test`。
- train：218,975 对、2,796 tiles；val：11,684 对、147 tiles；test：11,757 对、151 tiles。每对包含一份 HLS 和一份 fine LST。
- 从旧训练池 2,943 个 tiles 中按 `(SHA256("20260914:" + tile_id), tile_id)` 排序，前 `round(0.05×2943)=147` 个整体划入 val。同一 tile 的全部记录属于同一 split。
- 三组记录 ID 和 tile ID 两两无交集；新 train∪val 等于旧 train；test 成员与旧源 val 完全一致。tile ID 隔离不等于已核验地理 footprint 零重叠。
- 成员及哈希以 [split_summary.json](data_inspection/hlp_split_20260914/split_summary.json) 和同目录清单为准；读取与生成说明见 [划分协议](data_inspection/hlp_split_20260914/README.md)。不要在新任务中重新随机划分。

```python
from eo_data import open_dataset
ds = open_dataset("prithvi", "lst", "val",
                  prepared="processed_data/hlp_split_v2_20260914")
```

## 训练、兼容与迁移

- 使用新版本的剩余训练集归一化；validation/test 共用这些参数。不得继续使用包含新验证成员贡献的旧 HLP 归一化，不得用 test 调参或选模。
- 冻结读取器的程序默认仍指向旧 v1，旧 HLP `val` 是测试集。项目的新实验默认约定为 v2，必须在调用处落实，不能省略 `prepared` 后误用旧数据。
- 现有训练器、配置、QA 形状库及缓存尚未自动切换到 v2。启动新 HLP 训练前检查数据入口、选模集合、归一化和缓存指纹，按 v2 接入；不能把历史启动命令当成已完成切换。
- 旧模型见过原训练池中的新验证成员。新划分用于今后从零训练或未见过本数据的初始化；不把旧模型的该集合分数称为独立验证，不抹除旧测试监测记录。
- HLC 继续使用 `prepared="processed_data/v1"`，train/val/test 对应训练/验证/测试。HLP v2 目录只含 HLP，不对该目录调用要求两份数据齐备的 `open_datasets()`。
- 保留原压缩包、旧 v1、读取 ID、源记录 key 和历史检查点供复现。修改划分或预处理时建立新版本并同步报告、生成模板、catalog 与校验记录，避免覆盖冻结版本。
- 迁移或上传 HLP 时，携带原压缩包、新 v2 索引与归一化、划分清单、读取代码和当前报告。仅传原 ZIP 会缺少本次独立验证划分。
- LST 摄氏度读取为 `ds.denormalize(sample["image"]) / 100.0`，同时使用有效 mask；归一化 image 不能直接除以 100。fine LST 为卫星反演参考产品。

## 正式实验训练预算（2026-09-14 暂定）

- 后续正式对照默认统一训练 **5 个完整 epoch**。1 个 epoch 指该实验训练集合的所有记录各遍历一次，不是固定墙钟时长，也不是沿用历史固定步数。
- 同一对照中的模型使用相同数据版本、训练成员、有效 batch 和样本顺序规则；显存不同可调整微批与梯度累积，保持有效 batch 一致。
- HLP v2 每个 epoch 为 218,975 条记录，5 个 epoch 为 **1,094,875 次记录呈现/模型**。同一记录生成多个噪声视图不额外计为多个 epoch；另外报告视图呈现量。
- 不丢弃或重复补齐尾批；每个 epoch 结束时完成剩余梯度累积，按实际样本数加权，不跨 epoch 补满更新。有效 batch=64 时，每轮 3,422 次更新（最后一次31条），5轮共 **17,110 次更新**。其他有效 batch 按 `5 * ceil(N_train / effective_batch)` 推导，不硬编码此示例步数。
- 技术检查、短程筛选单独注明预算，不能与5轮正式结果混作等预算对照。历史实验、冻结配置和检查点保持原样。
- 本约定不代表训练器已接入 v2 或已启动新训练；启动前仍需完成上述数据入口、归一化、验证/测试与缓存指纹检查，并落实准确的 epoch 边界。
- 机器可读预算见 [training_budget.json](experiments/paper1/training_budget.json)。后续用户调整预算时同步更新。
