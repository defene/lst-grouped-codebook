# 两套数据的统一入口

名称与语义以[数据集文档](../data_inspection/FINAL_DATASET_SUMMARY.md)为准；[dataset_catalog.json](../experiments/data/dataset_catalog.json)提供机器可读映射。

| 工作名称 | 兼容读取 ID | 文档划分 → API split |
|---|---|---|
| HLS-LST-Pairs（HLP）v2 | `prithvi` | train → `train`；validation → `val`；test → `test`；须指定新 prepared |
| HLS-LST-CrossSensor（HLC） | `lstsr_tb` | train → `train`；validation → `val`；test → `test` |

HLP/HLC 是文档名称，冻结 API 仍接收上表的旧 ID。`lst` / `lst_hr` 指 fine-resolution LST；HLC 原字段 `lr_*` 指 coarse-resolution LST。HLC 一条 `sequence_id` 对应月份集合（calendar-month bundle），月份可跨年，不能将其默认作为连续时间序列。Landsat fine LST 是卫星反演参考产品。现有入口返回单模态自重建样本，尚未封装 coarse LST＋HLS → fine LST 三元组。

HLP 新版读取：`open_dataset("prithvi", "lst", "val", prepared="processed_data/hlp_split_v2_20260914")` 返回独立验证集；同一 prepared 下 `train/test` 分别读取新训练/固定测试。划分为218,975/11,684/11,757对，详见[划分协议](../data_inspection/hlp_split_20260914/README.md)。HLC 继续用 v1。下面未传 prepared 的旧调用示例只用于历史复现；冻结默认仍是 v1，旧 HLP `val` 表示测试。
+
+新 HLP 版本的归一化已排除验证成员的贡献，验证与测试共用新训练参数。旧模型见过新验证成员，不能作为独立验证模型直接沿用。

两份数据独立使用；处理方法共用。当前版本是无人工噪声、无几何增强的自重建数据。

## 一键处理与读取

在项目根目录运行：

```powershell
.\prepare_data.ps1 -Workers 4
python examples/read_eo_data.py
```

预处理命令会处理全量数据并验证结果。中断后重复执行同一命令可从已提交的 512 条记录检查点继续；已完成的数据保持冻结。输出位于 `processed_data/v1/`。开发子集有独立目录，默认训练接口拒绝使用。

```python
from eo_data import open_datasets, open_dataset

data = open_datasets(split="train")
hlp_hls = data["prithvi"]["hls"]
hlp_lst = data["prithvi"]["lst"]
hlc_hls = data["lstsr_tb"]["hls"]
hlc_lst = data["lstsr_tb"]["lst"]

sample = hlp_hls[0]
x = sample["image"]                 # float32 [6,256,256]; LST 为 [1,256,256]
y = sample["target"]                # 同值、独立内存
valid = sample["reference_mask"]    # bool [1,256,256]，广播到全部通道
decoded = hlp_hls.denormalize(x) # 仅在 valid 位置代表原数据
fingerprint = hlp_hls.fingerprint  # 保存到实验配置/checkpoint

validation = open_dataset("lstsr_tb", "lst", "val")
```

此接口按需读取，不会把整个数据集装入内存。使用其他位置时传 `root=项目根目录`；使用其他预处理版本时传 `prepared=该版本目录`（其下含 `prithvi/` 和 `lstsr_tb/`）。别名支持 `prithvi_data`、`lstsr_tb_final_upload_20260721`、`lst_hr`。

`get_metadata(i)` 提供来源信息；这些信息不会自动输入模型。`epoch_indices(epoch, seed)` 产生可复现的样本排列。所有样本的 `corruption_mask` 当前为 false，`input_mask` 等于 `reference_mask`。

## PyTorch 训练环境

基础依赖见 `requirements.txt`。NumPy 接口无需 PyTorch；训练环境安装 PyTorch 后可直接调用：

```python
from eo_data import make_dataloader

if __name__ == "__main__":  # Windows 多进程必须保留此保护
    loader = make_dataloader("prithvi", "hls", "train",
                             batch_size=16, num_workers=4, seed=20260909)
    for batch in loader:
        x, y = batch["image"], batch["target"]
        mask = batch["reference_mask"].expand_as(y)
        # prediction = model(x)
        # loss = (((prediction-y).square() * mask).flatten(1).sum(1)
        #         / mask.flatten(1).sum(1)).mean()
```

普通 VAE 与 Prithvi 架构 VAE 必须使用相同数据、seed、batch 配置和 mask 损失。Prithvi 如果要求时间维，只做 `x.unsqueeze(2)` 得到 `[B,C,1,H,W]`。训练脚本不要另加归一化、裁剪或尺寸变换。各 epoch 依次迭代同一个 loader；不要每个 epoch 重建并重置相同 generator。

## 固定的数据约定

- HLS 六波段顺序：Blue、Green、Red、NIR、SWIR1、SWIR2；原始整数乘 0.0001，读取器无额外 clipping。HLP 下载端已执行 clip(0,10000)，不能据读取器行为推定上游未裁剪。
- HLS 排除源填充值、Fmask 255 与 bit1/2/3，HLC 还与已有 valid mask 取交集。零值、雪、水、气溶胶位不额外剔除。
- LST 读取 fine-resolution 256×256。两来源均按“摄氏度 × 100”的 int16 存储编码解释，已用原始 Landsat 像元抽样核验并经用户确认，后续实验沿用此约定。恢复摄氏度为 `stored / 100.0`，不重复套用 Landsat 原始 DN 公式。
- 四组统计只在各自 train 中拟合：先算每个样本有效像元均值/二阶矩，再对样本等权汇总；HLS 按波段，LST 单通道。val/test 复用 train 参数。
- 无效像元在归一化后填 0，并保留 mask；全无效样本按模态排除，记录在 SQLite `rejected` 表。
- HLP v2 为独立 train/val/test；旧v1源 train/val 分别表示训练池/测试；HLC 的 train/val/test 分别表示训练/验证/测试。两份数据不合并。HLC 的 HLS 每个月份集合（原键 `sequence_id`）只计一次，LST 仅展开 manifest 的可用月份，不补齐缺月。

### LST 摄氏度读取约定

当前 `image` / `target` 已做 z-score，`denormalize()` 返回存储编码值。因此恢复温度要先反归一化，再除以 100，并使用原始有效 mask：

```python
import numpy as np

ds = open_dataset("prithvi", "lst", "train")  # lstsr_tb 使用同样方法
sample = ds[0]
temperature_c = ds.denormalize(sample["image"]) / 100.0
temperature_c = np.where(sample["reference_mask"], temperature_c, np.nan)
```

存储值、训练均值和标准差同时除以 100，z-score 保持一致，因此现有训练输入无需重算。MAE/RMSE 若在反归一化后的存储单位中计算，除以 100 得到 °C；MSE 除以 10000。冻结 v1 文件里的未知单位标志是构建时的历史状态，以本约定及[单位核验记录](../experiments/data/LST_ENCODING_VERIFICATION.md)为当前解释。压缩包内的说明保留打包时版本，迁移时一并携带这份更新说明。

## 存储与可恢复性

HLP 的四个原始文件逐样本无损编码，512 个样本组成一个 `.eoz` 分片。每个样本独立解压，SQLite 提供字节偏移，可随机读取。编码使用整数差分、字节重排及 Zstandard；NPY 头和 JSON 原始字节均保留。写入后逐条从磁盘解压并核对原始 SHA-256，分片另存完整 SHA-256。

HLC 复用已有压缩 NPZ，只新增固定索引、掩码有效像元计数和归一化参数；不复制大量 float32 数组。每个读取进程缓存最近两个源 NPZ。随机读取 LST 月份仍需解压所在的 12 个月份槽位的 fine LST 数组，这是保留现有压缩存储的代价。NPZ 大小和纳秒修改时间与预处理记录不符时拒绝训练；迁移这些源文件应保留修改时间。

`bundle.json` 是完成状态与版本证据；只有完成全量扫描、统计、分片校验和索引冻结后才生成。`progress.json` 给出处理中状态。`normalization.json` 保存最终参数；`protocol.json` 是输入规范快照，其历史 status 字段不代表当前产物状态。`verification.json` 保存读取验收结果。

```powershell
python -m eo_data verify --dataset all --samples 32
python -m eo_data restore-prithvi --destination D:\restored_prithvi
```

恢复目标必须为空；可加 `--limit 8` 验证小批恢复。恢复会写回四个原始文件及辅助文件，内容与原目录一致。恢复命令同时支持分片目录和下述单文件模式。HLC 仍依赖原 NPZ。

## HLP 单文件压缩包

当前交付文件为 `artifacts/prithvi_data_v1_20260909.zip`。它包含全部无损 EOZ 分片、冻结的索引和统计参数，以及读取代码。ZIP64 中的 EOZ 已经压缩，因此使用直接存储；读取器直接跳到样本的字节位置，无需解压整个 ZIP 或整个分片。

在当前项目里，`processed_data/v1/prithvi/storage.json` 指向该 ZIP，原来的 `open_dataset` / `open_datasets` 调用完全相同。只需保留新 ZIP 与轻量索引目录；旧解压目录、旧 ZIP 和单独的 EOZ 分片可在新包验证后清理。

迁移 ZIP 后，可用一条命令验证它并提取必需的索引（不提取大型分片）：

```powershell
python -m eo_data attach-prithvi D:\datasets\prithvi_data_v1_20260909.zip
```

迁移到没有代码的新项目时，先从 ZIP 中提取 `eo_data/` 文件夹并安装 `eo_data/requirements.txt`，再在该项目中运行上述命令。也可以完整解压整个 ZIP，直接使用其中的分片目录。ZIP 内容有逐文件 SHA-256 清单，外部 `.sha256` 文件给出整个 ZIP 的校验值。

从已完成、仍保留分片的预处理目录生成并验证新包：

```powershell
python -m eo_data archive-prithvi --destination artifacts\prithvi_data_v1_20260909.zip
```

该命令不覆盖已有压缩包，也不会自行删除原始文件。当前项目的旧副本清理由用户单独授权执行。

构建开始后不要修改 `eo_data/*.py` 或输入 `experiments/data/protocol.json`；恢复会检查代码和协议指纹。v1 的处理规则固定在 `core.py`，编辑协议文本不会自动改变算法。改变处理方法应同步实现代码并使用新输出版本，避免覆盖已用于实验的参数。读取器检查处理代码版本；同一输出目录的并发预处理会被写锁拒绝。

## 验证

```powershell
python -m pytest tests/test_eo_data.py -q
```

测试覆盖整数极值无损恢复、压缩损坏检测、mask/零值规则、样本等权统计、两种源格式的 train-only 参数、月份展开、统一输出、pickle 重建、源文件变化检测、恢复原目录、事务断点续跑、并发写锁，以及删除松散分片后的 ZIP 直接读取与迁移。正式运行另对全部 HLP 压缩记录做字节验证，对所有源样本计算实际有效像元和全训练集统计。
