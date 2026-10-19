# 地球观测架构用于 LST 复原：2026-09-27 接入说明

按最新决定，使用地球观测模型的网络框架，在 HLP 的 LST 上从零训练。无需迁就 DEM、NDVI 或光学影像的预训练权重；DEM 与 NDVI tokenizer 不重复算成两种架构。连续 AE 与 VAR 多尺度退出当前配置计划，历史实现保留。

当前完成的是适配代码与模型注册；正式配置须按最新版设计生成和验收。没有初始化模型、下载权重、运行前向/反向、训练或评估。完整官方依赖尚未绑定，数值正确性和参数量尚未实测。

## 本轮接入

|对照|保留的结构|LST 适配|比较性质|
|---|---|---|---|
|EO-VAE-LST|官方卷积 Encoder/Decoder、Gaussian posterior、KL 正则|使用官方固定通道层选项，输入输出为 1 通道 R 或 3 通道 C；从零训练|连续 VAE 架构对照；不称完整 EO-VAE 论文复现|
|TerraMind-FSQ-LST|官方 ViT 编码器、FSQ、条件扩散解码器|从零训练；输入输出 1/3 通道；FSQ 改为 [8,8,16]，共 1,024 状态|遥感离散 tokenizer 架构对照；保留扩散解码成本|
|Prithvi 空间 MAE（已有）|可见 patch 编码、mask token 解码|单帧二维 LST、从零训练|MAE 整块补全另列，不称 VAE|
|SatMAE++-LST|两尺度 MAE＋卷积上采样|128→256、从零训练，适配源码已实现|整块补全另列，数值验收未执行|

EO-VAE-LST 不给温度伪造光学波长，关闭动态波长输入输出层，使用官方源码已有的固定通道层。保留其自编码主体和概率瓶颈，但不测试多传感器能力，也不复现原论文 GAN/EQ 配方。这种适配会削弱其作为“独特遥感机制”的证据，因此结论必须限定为所列架构适配。

TerraMind 只采用图像 tokenizer，不引入多模态生成主干。其原生 DEM/NDVI 预训练配置的 FSQ levels 为 [8,8,8,6,5]；本实验没有加载这些权重，而是把从零训练版本改为 [8,8,16]。官方 FSQ 使用 sigmoid 后标量取整，与本项目通用 FSQ 的数值实现并非完全相同；相同状态数不表示量化映射相同。

## 条件、监督与预算

最新版设计中，EO-VAE在P7、TerraMind-FSQ在条件扩展P9，各为R/C* × clean/pixel10 = 4组，seed17。C*沿用一次验证选择；本两类模型不额外展开mixed、全部均温条件或全部表示。

所有模型共用 HLP v2 的训练/验证/测试成员、物理域污染、参考有效区域、均温侧信息与物理读出。每组 5 完整 epoch、1,094,875 条记录呈现、有效 batch64、17,110 次更新；不改变旧冻结实验。clean 指不施加人工污染，不表示参考产品绝对无噪声。

EO-VAE 训练采样后验，评估用后验均值；KL 按潜变量元素平均。采用 raw posterior → raw decoder 路径，不混入内部 BN/latent 打包接口。

TerraMind 编码器只看受污染观测与规定的 mask/均温条件。训练扩散解码器时使用随机时间步的加噪目标，预测 x0，再计算共同的物理重建损失；这是扩散监督，不是把干净目标送入编码器。评估从公开固定种子的高斯噪声开始，运行 50 步 DDIM，仅由离散索引和均温条件解码；不访问目标。训练随机时间步的损失不能当作完整复原分数。

自然图像组 SD/SDXL/FLUX.1 仍按之前约定保留预训练及微调路线；此次“从零训练”调整针对新增地理架构，不覆盖自然图像预训练对照。

## 公平性与效率

- TerraMind 与离散主线的名义载荷对齐：16×16 网格、256 索引、每索引 10 bit，另加均温 32 bit，共 2,592 bit。实际码流另计头、padding、支持 mask，共享权重另列。
- EO-VAE 是连续瓶颈，不把其潜变量数量冒称与离散索引等码率。
- 参数量对齐尚未完成。EO-VAE 当前为 ch128、[1,2,4,4]、每级2块、latent32；TerraMind 为官方 ViT-B 编码器及 UNet 扩散解码器。它们只是待验收的结构起点，不能直接标为等参数正式对照。正式比较前需实测并根据训练资源缩放/分档，保持结构机制、披露差异，不用闲置参数凑数。
- 扩散模型与单次解码模型分别记录训练更新成本、完整复原耗时、显存、参数和索引码率。只比参数量或编码耗时不足以证明“更高效率”。
- 同一数据预算不等于同一 FLOPs/GPU-hours；报告同时给质量—成本关系。当前没有运行证据，不给预估性能结论。

## 源码与可执行状态

实现见 `geo_models.py`；调用官方类，不用普通模型替代。训练器区分扩散训练接口与评估采样接口；评估继续走完整 val/test、固定污染视图、输入摘要、索引回解及成本记录。

`geo_sources/` 存放审阅用源码文本、许可证和 SHA256 清单，不作为可执行包导入。未来需将固定版本官方 checkout 路径写入 `model.source_root`，将其完整 `eo_vae/` 或 `terratorch/` 包所有 Python 文件的相对路径—SHA256 字典写入 `model.source_sha256`。构造器校验完整本地清单和已审阅入口摘要；缺依赖明确报错，不自动安装、下载或替换模型。

源码绑定不是运行验收。仍需模型梯度、索引回读、完整采样、尾批续训、正式微批显存和参数对齐检查。所有新配置保持 `execution.enabled=false`。

## 一手来源

- [EO-VAE 论文](https://arxiv.org/abs/2602.12177)、[官方实现](https://github.com/nilsleh/eo-vae)。审阅版本 `8e65a27546d6ad5c7b0b349024a27702d49f9333`。
- [TerraMind 官方项目](https://github.com/IBM/terramind)、[TerraTorch tokenizer 实现](https://github.com/terrastackai/terratorch/tree/0f3873f08058d991b638db8c5b71478777beb2ac/terratorch/models/backbones/terramind/tokenizer)。采用后者版本 `0f3873f08058d991b638db8c5b71478777beb2ac` 的 DiVAE 和 FSQ 接口。

唯一实验范围见 [最终设计](FINAL_PLAN_V2_ZH.md) 与 [逻辑清单](../lst_restoration_final_v2_20260927/manifest.json)。

SatMAE++ 的实现来源、双尺度损失和比较限制见 [SATMAE_LST_ZH.md](SATMAE_LST_ZH.md)。
