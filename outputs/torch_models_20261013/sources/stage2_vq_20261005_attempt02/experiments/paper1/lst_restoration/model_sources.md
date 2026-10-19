# 模型接入与来源边界

本目录是新实验代码。当前仅完成源码编写、AST 与 `compile()` 检查，**没有实例化模型、前向、反向、GPU 测速、模型评估或下载权重**。下述“已实现”表示源代码存在，不表示通过运行验收。

## 同骨干离散机制

编码器与解码器直接复用项目中已固定的 `eo_denoise/var_basic.py`，其来源为 [FoundationVision/VAR](https://github.com/FoundationVision/VAR)。当前卷积配置为 ch128、每级3个残差块、ch_mult=(1,1,2,2,4)，接口潜变量32×16×16。它是对原有 VAR 模块的实验配置，不宣称该配置等于官方发布权重。

输入适配器将表示图像、天然有效 R、观测有效 O、固定尺度归一化的逐图均温拼接，经过1×1卷积回到原输入通道数。初始图像通道恒等，其余权重为0。解码器第一层卷积特征后加入零初始化的均温仿射支路。**解码器不接原图、不接观测温度、不接云雾受影响真值mask，无图像跳连**。

scratch 共用模块按 `seed + 完整模块名的哈希` 分别初始化，使同形状共同权重不受量化器或输入通道数消耗随机数顺序影响。末层权重置0；R/G偏置0，C/C_affine偏置(-1,1,0)，对应物理零残差。此初始化尚未通过梯度运行检查，不能代替颜色跨段可训练性验收。

| 实现 | 数学来源与适配 | 索引和损失 |
|---|---|---|
| VQ | gradient-updated nearest-neighbor VQ；不混入EMA/restart | K=1024、D32；码本MSE + .25 commitment；FP32 exact search按token分块 |
| FSQ | [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505) 的有界标量、round STE、偶数level偏置 | 默认levels=[8,8,16]是本研究exact-K适配；混合基数1024状态；无VQ损失 |
| LFQ | [MAGVIT-v2](https://arxiv.org/abs/2310.05737) 的binary sign机制 | 10维±1，commitment + entropy；不是原论文完整tokenizer/损失复现 |
| BSQ | [Binary Spherical Quantization](https://arxiv.org/abs/2406.07548) 的投影→L2单位球→二值化机制 | 10维±1/√10，zero映射正号；entropy；不额外套VQ commitment |
| AE | 同一骨干、32维连续latent，移除量化器 | 无索引、无KL、无VQ损失；不宣称等码率 |

LFQ/BSQ默认采用**完整1024状态的联合soft entropy**，温度.1，系数.01，形式为 `E[H(p(code|token))] − H(E[p(code|token)])`。soft assignment来自平方欧氏距离Gibbs分布；同量化器内所有code norm相同，logits可写为2×内积/温度。当前系数与温度是本研究预设适配值，且两熵项比例固定1；不是作者的完整超参。此实现不把“各bit边缘熵之和”冒称联合码熵。精确联合熵限制1～12bits，按256 tokens分块；仍需真实内存与速度验收。训练soft熵与评价hard token困惑度不同。

每个量化器有 `decode_indices()`：完整索引加上规定均温即可进入共同解码器。FSQ与binary机制的必要投影是实际可训练参数；不可通过无用参数凑齐总量。单尺度默认256×10+32=2592bit，VAR多尺度默认680×10+32=6832bit；均是**索引加均温的标称载荷**，不含mask/容器/共享权重，也不是已测真实码流。

VAR多尺度量化器使用本地 `eo_denoise/var_quant.py` 的残差卷积与 `eo_denoise/var_ops.py` 插值政策，独立返回每级索引并采用分块精确近邻。loss按原参考累计重建latent的逐尺度VQ/commitment平均；未加入下游autoregressive生成器。没有修改历史模块。

## 自然图像预训练自编码器

支持本地 diffusers `AutoencoderKL` 目录：

- [SD sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)：4×32×32 latent。
- [SDXL VAE](https://huggingface.co/stabilityai/sdxl-vae)：同样4×32×32 latent，权重来源必须单独核验。
- [FLUX.1](https://github.com/black-forest-labs/flux)：仅接入明确转换为 diffusers AutoencoderKL 的**FLUX.1**本地权重；16×32×32 latent，scale .3611、shift .1159、无quant/post-quant卷积。原始 `ae.safetensors` 单文件不自动转换、不自动下载；不伪造 `AutoencoderKLFlux` 类，也不将FLUX.2冒充FLUX.1。

所有目录要求 `pretrained_sha256` 覆盖config.json与全部safetensors，并校验每个指定文件。加载强制 `local_files_only=True,use_safetensors=True`。这些哈希校验保证文件一致性，**来源身份仍需要实验manifest记录官方仓库、revision与获取证据**，不能靠一个自行填入的哈希证明权重官方身份。

训练采样posterior，评估采用posterior mode；KL按latent元素平均。自编码器直接encode/decode，不套扩散模型接口的scaling/shift。新增mask/均温适配器恒等初始化，预训练末层保持不变，不能清零之后仍声称官方零微调。latent容量与离散token不同，所有metadata明确 `equal_rate_claim=False`。

## 遥感 MAE 独立轨道

Prithvi实现复用本地固定的 `eo_denoise/prithvi_reference.py` 中PrithviViT与MAEDecoder（来源 [Prithvi-EO-2.0](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)，项目原适配记录revision `9eb1b1102806593963daa333bcc491b1c6f8562f`）。采用Conv2d、二维位置、无时间与地理条件；默认encoder768×9层、decoder512×7层。**具体参数量没有实例化实测，不预先宣称已经对齐。**

输入为外部整块16×16 mask。被遮盖patch在任何encoder attention之前剔除；decoder重新插入可学习mask token并恢复原空间位置。不再额外随机mask。clean对应保留全部tokens的同结构AE控制；原生MAE补全loss必须由训练器按缺失patch计算，不能用普通全图loss冒称原生训练。该类返回 `mae_patch_missing` 和 `mae_visible_patch_indices`，供监督与审计。随机像素缺失与patch mask不一致会直接报错。

这是从零训练的缩放空间MAE适配，不是官方300M预训练模型，不是离散VAE，不进入随机像素/云雾主榜；其连续CLS与可见tokens和位置元数据不可当作离散载荷。

SatMAE++ 已接入固定官方版本的两尺度 MAE 和卷积上采样模块，LST适配见 [SATMAE_LST_ZH.md](SATMAE_LST_ZH.md)。EO-VAE、TerraMind 见 [GEO_BASELINES_ZH.md](GEO_BASELINES_ZH.md)；所有新增模型运行验收仍未执行，EMA/restart-VQ 未接入。

## 启动前必须通过的运行门槛

1. 每个量化器索引域、端点、偶数level、索引→latent精确回读及索引→图像一致性。
2. 前向/反向有限值、各必要参数梯度、AMP与FP32差异、颜色跨段梯度。
3. 离散解码没有观测图像依赖；MAE被删patch不进入encoder；缺失mask对齐。
4. SD/SDXL/FLUX恒等adapter与原VAE的零微调数值等价；训练sample/评价mode及KL归约。
5. 所有模型参数、峰值内存、吞吐实测，以及checkpoint恢复一致性。

本轮上述运行项均未执行。源码静态检查不构成任何重建性能或效率结论。

## 2026-09-27 补充

连续 AE 与 VAR 多尺度本轮暂缓，上文其描述仅解释保留的历史实现。新增 EO-VAE-LST、TerraMind-FSQ-LST 均采用固定版本官方类并从零训练，具体结构适配、源码摘要、依赖门槛见 [GEO_BASELINES_ZH.md](GEO_BASELINES_ZH.md)。仅自然图像组维持预训练路线。
