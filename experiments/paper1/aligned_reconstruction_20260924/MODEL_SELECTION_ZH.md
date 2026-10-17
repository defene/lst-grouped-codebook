# 模型候选修订：地球空间重建与 MAE 空间补全

当前LST实验方案与模型执行资格见 [LST PROTOCOL_ZH.md](../lst_known_mean_design_20260924/PROTOCOL_ZH.md)；候选池不等于所有模型均已接入LST。

2026-09-24；候选审查阶段，未下载新权重、未启动训练。原 52/60 作业矩阵待重建。

> 最新LST主任务：用户提出假设已知每张图片完整有效区域的均温，因此当前主实验采用“给定均温、只重建残差”；此条取代下文较早的可见均值主实验安排。两条VQ路线与其他对照使用同一给定均温。详见 [KNOWN_MEAN_LST_ZH.md](KNOWN_MEAN_LST_ZH.md)。

## 用户明确修正

地球空间模型不是只有 Prithvi；选取真实已有的地球空间 VAE/离散自编码器/MAE。Prithvi 使用 MAE 的可见 token 编码、mask token 恢复和解码机制，不能继续以之前保留所有 token 的连续 AE 代替。单帧约束继续沿用。

## 有公开来源的候选

| 模型 | 类型及特点 | 当前适配判断 |
|---|---|---|
| Prithvi-EO-2.0 | MAE，时空 ViT；官方预训练 HLS | 单帧 MAE 作空间补全基线；适配 LST 通道与输出头 |
| SatMAE | 多光谱/多时间 MAE，光谱分组 | 多光谱基准候选；HLS 需重定义对应分组；LST 单波段适配后无法检验其多光谱优势 |
| SatMAE++ | 多尺度重建 MAE，卷积上采样解码 | 优先的第二个 MAE；检查是否改善边缘与不同尺度细节 |
| Scale-MAE | GSD 尺度编码、低/高频重建 | 空间尺度/频率结构的扩展基线；需要核验真实 GSD 和适配多光谱输入 |
| EO-VAE | 基于 Flux.2 自编码器，波长条件动态输入输出层，支持可变通道 | 真正以 EO-VAE 发布的多传感器重建模型；有公开权重；优先 HLS |
| TerraMind 图像 tokenizer | 基于 FSQ 的离散图像自编码器；模态专用 tokenizer | 遥感离散重建基线；先核验 S2L2A tokenizer 对 HLS 六波段的接口，不能当成普通可学习 VQ 码本 |

EO-VAE 属于 2026 年预印本候选，不能未经证据称为已经广泛采用的主流标准。其论文以连续 VAE 为基础，但遥感微调描述的目标为 Charbonnier 与 MS-SSIM；实际 KL、采样、归一化行为须读 pinned 配置/代码确认，不能凭名称默认沿用 SD 的 KL 系数。

TerraMind 指 tokenizer 编码器/解码器，不是把整个 any-to-any 生成模型作为一个等参数 VAE 放入比较。

## 首轮推荐候选池

1. Prithvi-EO-2.0 MAE：空间补全。
2. SatMAE++：多尺度 MAE 补全。
3. EO-VAE：连续地球空间自编码器。
4. TerraMind FSQ tokenizer：地球空间离散自编码器。
5. 基于 VAR VQ-VAE 的主推离散模型：原生 LST 与线性颜色 LST 两条训练路线，见下节。
6. SD sd-vae-ft-mse：自然图像预训练 KL-VAE。
7. SDXL VAE（stabilityai/sdxl-vae）：同样四通道、8倍空间下采样的自然图像预训练 VAE，对照预训练权重差异。
8. FLUX.1 VAE（官方 black-forest-labs/FLUX.1-schnell 的 ae.safetensors）：16 通道、8倍空间下采样的自然图像预训练 VAE，作为更大潜变量容量的对照。

用户要求自然图像领域选三个 VAE，现选 SD、SDXL、FLUX.1；共八项主候选。SatMAE 和 Scale-MAE 保留为扩展候选。这八项是模型候选，不是八项都已证明可以无改动处理 LST/HLS 的可执行清单。

### 主推方法：VQ-VAE 的两种数据表示（最新用户修正）

用户明确：此前称作“通用离散基线”的 VAR VQ-VAE 路线是我们主推的方法，应分成原生 LST 图像训练和线性变化后的颜色空间图像训练两类。其他地球空间与自然图像模型是比较对象；不预设主推方法必然优于对照。

| 路线 | 输入/输出表示 | 最终评价 |
|---|---|---|
| VQ-LST | 原生单通道 LST，经固定 train 标准化 | 还原为摄氏度 |
| VQ-LST-LinearRGB | 相同 LST 经固定线性标量变换与明确的 RGB 编码，输出经逆变换恢复 LST | 同一摄氏度域 |

每条路线各有无人工遮盖训练、10%随机遮盖训练两个独立模型，合计 4 个 LST 主推方法训练格。遮盖使用相同物理位置，发生在表示变换之前；像素遮盖与 MAE patch 遮盖的比较仍依前述协议分别定义，不混作同一条件。

两条路线使用相同 VQ 主干、K=1024、D=32、多尺度680索引/次编码、量化规则、训练样本顺序和5轮预算；改变输入/输出通道所需首末层参数差异必须实测列出。三通道路线不自动增加 latent/token 数，也不训练三份独立模型。其余主干采用相同初始化快照；形状不同的首末层采用固定 seed 规则。

本轮主配对首先在原始绝对温度上定义，不能一边使用绝对温度、另一边额外给予均值/残差放大。先前已确认的可见均值残差实验保留为独立因素：在 VQ-LST 与 VQ-LST-LinearRGB 两条路线同步增加相同均值旁路，另计作业，不能只给彩色路线使用。HLS 的总体实验范围保持，但本次明确的两路线先落实于 LST，HLS 颜色编码不据此自动冻结。

“线性”需在可执行配置中写出公式：标量变换可用 z=(T−a)/(b−a)，a/b 为固定且有依据的常数；严格仿射 RGB 编码可用 c=c0+z(c1−c0)，c0≠c1，逆变换为 z_hat=((c_hat−c0)·(c1−c0))/||c1−c0||²。端点尚未选定，不能宣称方案已完全冻结。若采用彩虹查表、RGB 折线路径，应明确它是分段线性/非线性颜色编码，不能将整个处理流程称为全局线性变换。

主两路线不含 asinh、逐图 min-max 或直方图均衡化；此前 asinh_gray/asinh_rgb 降为历史参考或后续独立消融。固定浮点映射及其有效域，记录越界并避免静默裁剪；拟合常数只能用 train，不能用缺失块真值决定每幅图缩放。训练的共同重建项与最终指标在逆变换后的相同标准化物理域计算，避免仅由 RGB 数值尺度改变损失权重。

推荐额外线性灰度复制控制只在需要区分“单通道→三通道”和“颜色编码”收益时加入，不算用户指定两路线中已启动的第三条。比较自然图像预训练 VAE 时，使用与 VQ 颜色路线相同的 RGB 数据及相同逆变换。

这是实验假设与模型定位，不是已取得的贡献或性能结论；本次仅更新设计，未启动训练。

用户进一步询问借鉴 Vision Banana 距离编码：已核验原文并提出“线性温度归一化＋RGB立方体边分段线性编码”的候选，具体公式及与原文非线性距离变换的区别见 [VISION_BANANA_MAPPING_ZH.md](VISION_BANANA_MAPPING_ZH.md)。该候选尚未替代性冻结为可执行映射。

### 三个自然图像预训练 VAE 的比较口径

SD 与 SDXL 官方说明采用相同自编码器架构，约 83.7M 参数，因此不称三个完全不同架构家族。256×256 输入时，SD/SDXL 的 posterior mean 各为 4×32×32；FLUX.1 为 16×32×32。后者连续 latent 元素数四倍，不能把全部重建提升归因于预训练方法。FLUX.1 参数数在固定 revision 后实例化实测，不把整套扩散模型的参数数记作 VAE 参数数。

三者只加载 VAE 编解码器，分别评估零微调和等数据预算的全参数微调，保持各自原生潜变量。SD/SDXL 的 latent scaling、FLUX.1 的 scaling/shift 遵循各自 API，避免重复变换。默认确定性评价 posterior mean；官方 FLUX 的采样开关需在接入时核验。新加入模型本轮尚未下载或训练。

官方来源：[SD VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse)、[SDXL VAE](https://huggingface.co/stabilityai/sdxl-vae)、[FLUX.1 权重](https://huggingface.co/black-forest-labs/FLUX.1-schnell)、[FLUX 官方自编码器配置](https://github.com/black-forest-labs/flux/blob/main/src/flux/util.py)。

## 空间补全协议必须重新区分

- 原生 MAE 测试：显式外部 patch mask，编码器只见未遮盖 token，解码器在正确位置插入 mask token，评价遮盖块物理量误差。其他模型使用完全相同被遮盖像素区域。
- 256×256、patch16 时有 256 个 patch；遮 26 个是 10.15625%，不是严格 10%。天然无效区域会改变有效像素覆盖比例，必须另报有效覆盖率。
- 随机 10% 像素测试单列。不得只给 MAE 用整块缺失、给其他模型用分散像素缺失，再按一个排名判优。MAE 处理部分缺失 patch 需要声明专门适配，不能冒充原生 token dropping。
- MAE 的 masked-only loss 在 0 遮盖时无有效监督，可能分母为零。无遮盖训练若改用全图重建，标作该架构的 AE 控制，不声称原论文 MAE 训练方案。
- 原生 MAE 的高比例预训练遮盖和本任务 10% 遮盖微调分开报告。直接零微调测试与适配训练后测试也分开。
- 若统一全图物理损失，只能称统一重建目标的 MAE 结构消融；保留模型原生损失时则是完整训练方法比较。不可同时声称完全复现各自原生训练方法、又所有损失完全相同。
- patch target normalization 不能在推理恢复物理值时读取缺失块真值均值/方差。按物理量可解码的监督版本处理，并记录与原默认训练目标的差别。

## 通道、预训练和参数边界

HLS 六波段要按实际 band 名、传感器和处理级别核对官方输入波段与归一化。EO-VAE 可变通道能力允许研究六波段组合，但不能由此推断 HLS 与原 S2 输入分布完全相同。

LST 是卫星反演参考温度产品，不是原始热红外辐亮度波段。EO-VAE 的波长条件和 TerraMind 的模态专用接口都需要明确新增适配；不捏造“温度波长”，也不声称已有官方 LST 预训练权重。首轮 LST 可靠的入口是 MAE 单通道适配、VAR 单通道及 SD 的灰度/颜色表示；另两种模型在适配方案核验后加入 LST 矩阵。

保留官方预训练检查点的零微调/微调实验，参数量按实测原样报告。官方 Prithvi 的大模型不能裁剪后仍宣称使用完整官方预训练模型。近似等参数的从零训练版本单开一条控制线，并标明宽度/深度修改。所有模型先审计 encoder/decoder 权重是否齐全；只有 encoder 权重不能报告完整 MAE 零微调重建性能。

FSQ 离散状态数量由各维离散 levels 的乘积决定；不能当作普通 VQ 可学习码本 K。要对齐离散载荷需同时看状态数、token 数和实际编码位数，并保留各模型机制。尚不冻结统一参数目标、模型数量和训练作业数。

## 核验来源

- [Prithvi 官方说明](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)
- [SatMAE 官方实现](https://github.com/sustainlab-group/SatMAE)
- [SatMAE++ 官方实现](https://github.com/techmn/satmae_pp)
- [Scale-MAE 官方实现](https://github.com/bair-climate-initiative/scale-mae)
- [EO-VAE 论文](https://arxiv.org/html/2602.12177v1)、[代码](https://github.com/nilsleh/eo-vae)、[公开模型](https://huggingface.co/nilsleh/eo-vae)
- [TerraMind 官方实现](https://github.com/IBM/terramind)、[论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Jakubik_TerraMind_Large-Scale_Generative_Multimodality_for_Earth_Observation_ICCV_2025_paper.pdf)
