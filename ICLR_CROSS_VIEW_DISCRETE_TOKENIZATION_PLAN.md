# 两篇论文研究总方案：地球观测离散表征与非同期跨卫星热学重建

> 2026-09-14 划分完成：HLP v2 已建立218,975对训练、11,684对独立验证、11,757对固定测试。新读取须指定 `prepared="processed_data/hlp_split_v2_20260914"`，并使用新训练统计。见[成员、规则和读取方式](data_inspection/hlp_split_20260914/README.md)。下文“无独立验证”指旧v1及历史实验；已有训练配置尚未自动切换，新实验须接入v2并重新训练。

> 2026-09-14 术语映射：数据集统一称 HLS-LST-Pairs（HLP，旧 ID `prithvi`）与 HLS-LST-CrossSensor（HLC，旧 ID `lstsr_tb`）；温度网格称 coarse/fine，HLC 的 sequence 为月份集合。HLP 源 val 是测试集，无独立验证集；后续验证须从训练 tiles 建立。本文以下旧名称及 split 用语为历史规划，当前事实以[数据集文档](data_inspection/FINAL_DATASET_SUMMARY.md)和[读取说明](eo_data/README.md)为准。模型名 Prithvi-EO-2.0 保留。

> 2026-09-10 实施更新：论文一以用户确认的 **Prithvi 一对一 HLS–HR LST** 为主数据，可运行框架见 [eo_denoise/README.md](eo_denoise/README.md)。两份数据继续独立处理与训练，论文二的 LSTSR 多月份/跨卫星数据不混入论文一。本文历史日期、来源布局、单位待确认与模型数值不代表当前实测状态。

> 文档状态：两篇独立论文的研究与实验执行方案  
> 更新日期：2026-09-05（补充用户对噪声与图像还原的定义）  
> 当前投稿重点：论文一，CVPR 2027  
> 论文二同步开展问题审计与基线探索；投稿时间按独立结果成熟度决定，尚未约定两篇同轮投稿。  
> 沿用原文件名作为入口；本文已替代原 ICLR 单篇计划与旧截止日期。

## 0. 两篇论文的定位与衔接

用户确认分别形成两篇论文：第一篇保持原方案的离散化方法主线；第二篇研究非同期光学引导下的热学场重建与跨卫星重建。不能将两条线合并成一篇，也不能把第二篇仅定义为给第一篇 tokenizer 接一个预测头。

| 维度 | 论文一：离散表征方法 | 论文二：非同期与跨卫星重建 |
|---|---|---|
| 暂定题目 | Cross-View Discrete Tokenization for Noise-Robust Earth Observation | Reliable Thermal Reconstruction with Asynchronous Optical Guidance across Satellites |
| 核心问题 | 如何离散表达 HLS/LST，同时保留物理信息并对损坏保持稳定？ | 如何从非同期光学条件及跨卫星低分辨率热学观测恢复高分辨率温度场？ |
| 主要对象 | HLS 与 LST tokenizer、量化和表示学习 | 静态 HLS + MODIS LR LST → Landsat HR LST |
| 方法贡献 | 量化机制、跨视图结构约束、噪声稳定性；shared/private 为候选设计 | 非同期引导可靠性、跨传感器观测差异、条件重建机制 |
| 主证据 | 率失真、物理保真、token 稳定性与轻量下游可用性 | 实际重建、错误引导鲁棒性、时差/质量分层、传感器差异消融 |
| 图像恢复与预测 | 云/异常污染或掩膜缺失的同分辨率图像还原 | 非同期跨卫星LR→HR完整系统 |

论文一核心命题：**利用光学—热学观测中的互补结构学习离散地表表示，在云、云影等异常污染或其掩膜缺失下还原图像，同时保持物理保真与表示效率。** 不预设 EO 必然比自然图像更低维，也不预设共享码本一定成功。

论文二核心命题：**在光学观测非同期、热学观测跨卫星且质量不一致时，学习可靠的高分辨率温度场重建。** 可以使用论文一 tokenizer，但必须有独立的重建方法与证据。

本文 §2–3 为共享数据与历史基线；§4–10、§14–17 为论文一方法和实验；§11–13、§18 为执行与工程安排；§19 为论文二完整计划；§20–21 规定贡献边界与成果交接。

论文一的具体执行顺序、模型对照、损坏协议、主指标、算力计算及阶段判断见 [第一篇实验执行计划](PAPER1_EXPERIMENT_PLAN.md)；机器可读实验清单见 [registry.json](experiments/paper1/registry.json)。第一篇实施细节以该执行计划为准，总方案保留候选设计与两篇论文边界。

### 0.1 用户确认的“噪声”定义（优先于旧的扰动设想）

用户进一步确认：第一篇以现有 mask 驱动的离散图像还原为当前实施路线，在可靠区域额外遮挡构造自监督训练/验证样本。完整有云/无云配对不是启动条件；保留原始污染值的配对恢复属于可选扩展。输入 mask 与监督参考 mask 分开，真实云区无参考时不报告其定量精度。

第一篇的噪声指云、云影等影响地表观测的异常污染，目标是同分辨率图像还原。Gaussian等人工数值扰动仅为可选诊断，不能代替实际异常还原的主证据。离散化是方法核心，率失真和token稳定性服务于还原能力；第二篇仍以非同期跨卫星LR→HR为主。

当前包为clear-only，云相关区域可能只保留QA与nodata。数据中云下没有可靠地表参考时，不得把原始云值、nodata、其他月份或模型自己的输出当真值。详细R1有参考云形缺失、R2真实孔洞、R3保留污染值的观测还原协议见第一篇实验计划§7；三者分别标注，不相互冒充。

## 1. 投稿定位与优先级

### 1.1 CVPR 2027 官方时间

根据 [CVPR 2027 Dates and Deadlines](https://cvpr.thecvf.com/Conferences/2027/Dates)，本轮核对日期为 2026-09-05：

- Paper registration：2026-11-10 AOE；
- Paper submission：2026-11-16 AOE；
- Supplementary materials：2026-11-23 AOE。

从 9 月 5 日至全文截止约 72 天。论文一优先完成可投稿结果；不以同时提交两篇为前提挤压论文一的验证时间。论文二何时投稿，取决于独立机制与结果是否成立。

### 1.2 论文一最小完整结果包

1. HLS 和 LST 的多码率重建与物理保真结果；
2. VQ、FSQ、连续 AE 与所提方法的公平比较；
3. 跨视图约束及噪声稳定性的直接证据，排除平滑和码本坍缩；
4. 单模态 codec 与跨模态条件重建分别报告；
5. 固定 train/val/test、关键多种子结果、tile-cluster CI；
6. 一个从像素域云/异常缺失输入到同分辨率影像的恢复任务；masked-token准确率仅是辅助，不能代替图像还原。

自然图像对照保留为机制分析，不要求证明“EO 更容易离散化”。双向去噪、三码本和更多 EFM 仅在必要时扩展。完整非同期建模、跨卫星观测约束和重建系统留给论文二。

### 1.3 首轮机制判断点

计划在 2026-09-19 前完成论文一 pilot：率失真、物理保真与稳定性至少出现一种可复现的改进，并由消融说明来自表示学习机制。具体有意义的改进幅度在 pilot 协议中根据指标尺度预先确定，不把原来的任意百分比作为投稿保证。

若只有加入 HLS 后 decoder 重建变好，而 token 本身没有可验证的改善，该结果只是条件融合基线；继续修改论文一方法，不提前把完整重建系统搬进第一篇。

---

## 2. 数据定义与可支持的科学结论

### 2.1 当前数据构成

每个 sequence 包含：

- 一张静态 HLS 光学条件：6 波段，`[6,H,W]`；
- 最多 12 个日历月份的 Landsat HR LST：`[12,256,256]`；
- 对应观测日期的 MODIS LR LST，存储为 `[12,8,8]` 网格；
- HLS valid mask、Fmask/cloud mask；
- LST valid mask；
- MGRS tile、month、HLS source time、candidate rank 等已有元数据。

当前规模约为：

- 264,143 个原始 train sequence；
- 2,750,483 个原始 train image-month 样本；
- 从原始 train 按 MGRS tile 划出 val：13,694 个 sequence、141,711 个 image-month、118 个 tile；
- 划分后实际 train：250,449 个 sequence、2,608,772 个 image-month、2249 个 tile；
- manifest `split=test`：141,524 个 image-month、13,665 个 sequence、123 个 tile。

注意：2.6M 是月度 LST 样本量，不是 2.6M 个独立 HLS 场景；同一 sequence 的多个 LST 月份共享同一个静态 HLS。

### 2.2 模态的准确表述

- HLS B2–B7：光学反射率视图，可称 **optical reflectance view**；
- LST：地表温度产品，可称 **thermal surface-field view**；
- 当前数据不是严格同期的原始可见光—热红外成像对；
- HLS 是静态空间结构条件，LST 是随月份变化的动态物理场。

因此，当前最稳妥的主张是：

> **Static optical context guides noise-robust discrete reconstruction of dynamic thermal fields.**

只有 HLS→LST 和 LST→HLS 两个方向都得到显著结果后，才能主张：

> **Bidirectional cross-view co-denoising.**

### 2.2.1 本地数据核查结论与待核实项

- 每组为跨年份选取的日历月份集合，不能作为连续一年时序或月平均场；manifest 的 source-year-count 分布为 2–7 年。
- HLS 为 Landsat 8 HLSL30 六波段条件，时间范围覆盖 2019–2025 年；反射率 scale 为 `0.0001`。
- HLS 平均有效率约 99.9166%，仍有 2,639 组雪比例超过 50%；有效 mask 不等于无雪或与目标月份匹配。
- LST 为 `int16`，nodata 为 `-32768`；温度解码公式尚须从生产代码或明确元数据确认，不能仅按数值范围猜测。
- LR 元数据记录 `nearest` 重采样。8→256 是存储网格倍率，不直接等同于真实热红外空间分辨率提高 32 倍。
- MODIS 与 Landsat 是不同传感器产品，不能先验要求 `downsample(HR) = LR`。传感器时差、QA、填补来源与偏差是论文二审计重点。
- 数组已完整解压，277,808 个 LST 与 277,808 个 HLS 文件均与 manifest 对齐；CRC/大小及文件索引检查通过。此前实际数组内容抽查为 16 对，不能称为全量科学质量核验。

### 2.3 不增加的数据

- 不使用高程；
- 不私自增加未讨论的传感器或标签；
- 自然图像仅作为公开 control dataset，用于数据几何和率失真对照；
- 人工噪声由现有样本在线生成，不构成新增观测数据。

### 2.4 数据划分

训练采用用户确认的sample-level方式：从manifest可用月份展开`(sequence_id, month)`，每个paired sample只含一张HR LST、对应位置静态HLS及mask；在有效sample列表上均匀采样。缺失sequence、缺失月份或无有效监督的样本忽略并记录，不补齐12个月；同一sequence的其他有效样本保留。像元级nodata仍由mask处理，不因局部缺失就丢弃整个sample。展开后的sample继承既有tile split，不重新划分。第一篇无需LR输入。

两篇论文共同使用以下划分，不重新抽样：

| Split | Sequence | Tile | 可用月份样本 |
|---|---:|---:|---:|
| train | 250,449 | 2,249 | 2,608,772 |
| val | 13,694 | 118 | 141,711 |
| test | 13,665 | 123 | 141,524 |

当前划分已落实到三个 manifest 的 `split` 列。固定 seed 为 `20260905`，对原始 train tile 按 `SHA256(seed:tile)` 排序，取前 118 个 tile 作为 val（约 5%）。同一 tile 的所有 sequence 和月份归入同一 split；原始 test 成员不变。名单与校验记录见数据目录 `splits/`。

模型选择必须使用 manifest train 内部的 tile-disjoint validation。manifest test 继续作为 external benchmark。

由于当前 baseline 已经在 manifest test 上完整评估过，manifest test 不再是严格意义上的从未查看 hidden test。论文级最强协议为：

1. 从现在起，所有新方法和超参数只使用内部 tile-disjoint validation；
2. 新方法定型后只运行一次 manifest test；
3. 当前主划分保持固定；如需额外时间留出实验，提前记录独立实验协议，不替换已确定的主划分；
4. 所有统计以 tile 或 sequence 为单位 bootstrap，不能把数百万 image-month 当成完全独立样本。

---

### 2.5 新划分下的训练与泄漏控制

- 划分 seed 永久固定为 `20260905`，直接读取已保存的 split 和名单；模型训练多种子与划分 seed 分开记录。
- 所有拟合步骤（tokenizer、码本、归一化统计、熵模型和预测器）仅使用 train；val 用于模型选择，test 用于定型评估。
- 旧 tokenizer 的训练名单未确认，可能包含新 val；旧 checkpoint 可用于探索，论文主结果须重新按新 train 训练，或先证明训练成员不含 val/test。
- 已验证三组无相同 tile ID；尚未验证相邻 tile 的实际覆盖重叠，不能直接声称地理 footprint 完全隔离。
- 论文二不得根据论文一 test 的失败样本反向调参；分别记录 test 访问和方法冻结时间，不将重复使用的 test 称为从未查看。

---

## 3. 当前已实现的模型与结果

本节为原方案记录的历史模型与指标，本轮仅核查数据，尚未在当前工作区找到相应训练代码和 checkpoint。不能据此宣称新划分下结果已复现。

### 3.1 当前模型

当前模型位于：

- `src/lstsr_single_image/models/stage2_tokenizer_var.py`
- `configs/local_stage2_tokenizer_var_v512_100k.yaml`

它是 FoundationVision/VAR 风格的单通道 LST 多尺度残差 VQ-VAE：

```text
HR LST [B,1,256,256]
  → ResNet Encoder
  → continuous latent [B,32,16,16]
  → quant_conv
  → multi-scale residual VQ
  → post_quant_conv
  → ResNet Decoder
  → reconstructed LST [B,1,256,256]
```

主要配置：

- 参数量：108,827,905；
- `ch=160`；
- `ch_mult=[1,1,2,2,4]`；
- 每层 2 个 encoder ResBlock；
- latent：`32×16×16`；
- codebook：`512×32`；
- `beta=0.25`；
- self-attention：最低分辨率层和 bottleneck；
- 训练 100k steps，当前最佳 checkpoint 为 95k。

### 3.2 当前多尺度量化

量化尺度：

```text
1, 2, 3, 4, 5, 6, 8, 10, 13, 16
```

总 token 数：

\[
1^2+2^2+3^2+4^2+5^2+6^2+8^2+10^2+13^2+16^2=680.
\]

codebook 为 512，因此每个 token 理论上 9 bit：

\[
R_{\max}=\frac{680\times9}{256^2}\approx0.0934\ \text{bpp}.
\]

当前十个尺度共享一个 codebook，并通过 4 个残差修正卷积重建 `16×16` latent。

### 3.3 当前损失

\[
\mathcal L_{current}
=\mathcal L_{MSE}
+\mathcal L_{VQ}
+0.1\mathcal L_{gradient}.
\]

当前不存在：

- HLS 输入；
- 人工噪声增强；
- token consistency；
- perceptual/GAN loss；
- 光谱 loss；
- cross-modal loss；
- time/geo token；
- Transformer/VAR predictor。

当前数据增强只有水平和垂直翻转。

### 3.4 当前结果及其边界

manifest-test full external evaluation：

- RMSE：约 0.8681°C；
- MAE：约 0.6673°C；
- bias：约 -0.079°C；
- SSIM：约 0.6149；
- gradient RMSE：约 0.3291。

这些结果证明：

- HR LST 可以被当前多尺度离散瓶颈较好地重建；
- 当前 LST 具有较强的低码率可压缩性。

这些结果尚不能证明：

- LR→HR 超分辨率；
- 真实传感器去噪；
- HLS 离散化；
- optical–thermal cross-view；
- noise-stable token。

---

## 4. 论文一：问题形式化

令：

- \(O\)：静态六波段 HLS optical view；
- \(T_m\)：月份 \(m\) 的 HR LST thermal field；
- \(L_m\)：同月 native LR LST；
- \(\widetilde O=C_O(O;\epsilon_O)\)：受控损坏后的 HLS；
- \(\widetilde T_m=C_T(T_m;\epsilon_T)\)：受控损坏后的 LST。

shared/private 是论文一候选参数化，不是必须实现的三码本结构。若采用该设计，不强迫 HLS 与 LST 的全部 token 一致，而写为：

\[
z_O=(z_O^s,z_O^p),\qquad
z_T=(z_T^s,z_T^p),
\]

其中：

- \(z^s\)：跨模态共享的空间结构；
- \(z_O^p\)：HLS 特有的光谱信息；
- \(z_T^p\)：LST 特有的温度动态信息。

论文评估四个轴：

1. **Rate**：token 序列实际需要多少 bit；
2. **Distortion**：重建误差和物理量误差；
3. **Stability**：同一样本在不同噪声下 token 是否稳定；
4. **Utility**：token 是否支持物理预测和第二阶段生成。

不建议把四者强行合成一个任意加权的总分；应报告 Pareto frontier。

---

## 5. 拟提出方法：Cross-View Discrete Denoising Tokenizer

### 5.1 最小可实现版本：Optical-Guided Thermal Tokenizer

论文一的条件重建 pilot 基线（不单凭此结构认定离散方法贡献成立）：

```text
noisy/partial LST → LST Encoder → multi-scale VQ tokens ┐
                                                        ├→ cross-fusion → LST Decoder → clean-reference LST
static HLS         → HLS Encoder → continuous features ┘
month token -----------------------------------------------------------┘
```

定义：

\[
z_T=Q_T(E_T(\widetilde T_m)),\qquad c_O=E_O(O),
\]

\[
\widehat T_m=D_T(z_T,\operatorname{CrossAttn}(z_T,c_O),e_m).
\]

其中 \(e_m\) 为已有 month 元数据的 embedding。HLS source time 可作为消融；不使用高程。

这一版本的优势：

- 与当前数据的静态 HLS—动态 LST 结构一致；
- 探索可从历史 checkpoint 初始化，主表须满足 §2.5 的新划分约束；
- 能直接和 concat、continuous cross-attention 比较；
- 先做小规模 pilot，再决定是否扩展。

这里 HLS 只在量化后进入 decoder。固定参数时，同一 LST 输入的 token 不直接依赖 HLS；必须通过独立的训练约束或量化前交互并直接测量 token，才能证明跨视图提升表示稳定性。

### 5.2 候选扩展：Shared–Private Cross-View Tokens

完整模型包含两个 modality-specific stem/encoder、共享结构量化器和两个 private 量化器：

\[
h_O=E_O(\widetilde O),\qquad h_T=E_T(\widetilde T_m),
\]

\[
z_O^s=Q_s(P_O^s(h_O)),\qquad z_T^s=Q_s(P_T^s(h_T)),
\]

\[
z_O^p=Q_O(P_O^p(h_O)),\qquad z_T^p=Q_T(P_T^p(h_T)).
\]

解码：

\[
\widehat O=D_O(z_O^p,z_O^s,z_T^s),
\]

\[
\widehat T_m=D_T(z_T^p,z_T^s,z_O^s,e_m).
\]

默认设计：

- HLS 与 LST 使用独立输入 stem 和输出 head；
- 空间尺度日程保持一致，便于位置对齐；
- shared tokens 使用共享 codebook 或软分配对齐；
- private tokens 使用模态独立 codebook；
- 不强迫 HLS 与 LST 的所有硬 token index 相等；
- cross-attention 只交换共享结构和必要的 private context；
- modality dropout 防止模型退化成只复制自身输入。

如果共享 codebook 导致 collapse，回退为独立 codebook + contrastive/soft-assignment alignment。

### 5.3 损失函数

总损失：

\[
\mathcal L=
\lambda_T\mathcal L_T^{rec}
+\lambda_O\mathcal L_O^{rec}
+\lambda_q\mathcal L_{quant}
+\lambda_n\mathcal L_{noise-stability}
+\lambda_c\mathcal L_{cross}
+\lambda_p\mathcal L_{physical}.
\]

#### LST reconstruction

\[
\mathcal L_T^{rec}
=\mathcal L_{masked-charbonnier/MSE}
+\alpha_g\mathcal L_{gradient}.
\]

#### HLS reconstruction

\[
\mathcal L_O^{rec}
=\mathcal L_{bandwise}
+\alpha_{sam}\mathcal L_{SAM}
+\alpha_g\mathcal L_{gradient}.
\]

#### Same-modality noise stability

对同一样本产生两个独立损坏视图：

\[
x_1=C(x;\epsilon_1),\qquad x_2=C(x;\epsilon_2).
\]

鼓励其 soft code assignment 或 encoder feature 一致：

\[
\mathcal L_{noise-stability}
=D_{sym}(p(z|x_1),p(z|x_2))
+\alpha_f\|\operatorname{sg}(h_1)-h_2\|_2^2.
\]

不建议一开始直接要求所有 hard token 完全一致，否则容易牺牲细节或造成 collapse。

#### Cross-view structural alignment

只对 shared feature/token 对齐：

\[
\mathcal L_{cross}
=\mathcal L_{contrastive}(z_O^s,z_T^s)
+\mathcal L_{cross-reconstruction}.
\]

#### Physical fidelity

HLS：

- SAM；
- NDVI、NDWI、NBR 重建误差；
- 各波段 bias/nRMSE。

LST：

- 温度 bias；
- gradient error；
- top-5% 极端温度误差；
- hotspot area/recall；
- 同源场聚合保真；跨传感器 coarse-mean consistency 需先完成 §19 的观测一致性审计。

---

### 5.4 单模态 codec 与条件重建必须分开

论文一优先验证“训练时利用配对跨视图约束，推理时每个模态可独立编码解码”的表示学习设置：

\[
z_O=Q_O(E_O(\widetilde O)),\quad z_T=Q_T(E_T(\widetilde T)),\quad
\widehat O=D_O(z_O),\quad \widehat T=D_T(z_T).
\]

跨视图信息通过训练目标改善特征和量化分配；shared/private、独立码本加软对齐均为候选方案。损失同时约束保真与码本使用，不能只优化 token agreement。

§5.1–5.2 的解码端跨模态交互作为另一个条件设置：所有 baseline 获得相同条件；分别报告目标 token 码率、条件存储和总成本。不能把连续 HLS 特征视为免费传输后与单模态 codec 混排。跨视图约束应容许物理与时间不一致，不强制所有空间位置对齐为相同语义。

---

## 6. 云与异常污染的图像还原协议

### 6.1 目标与掩码

本篇噪声主要指云、云影、云边缘以及经核实的异常污染；Gaussian、热像元与矩形随机遮挡降为可选补充诊断。积雪、水体、真实极值不是默认噪声。HLS与LST各用自身产品定义与观测日期解释QA，不能跨时相照搬云标签。

分别维护输入可用mask、污染/缺失mask和参考可信mask。模型应输出还原图，真实无参考孔洞不进入有监督指标；有参考的受控遮挡区必须进入损失与主要评价，不能因输入invalid而一并排除。

### 6.2 当前数据保留情况

2026-09-05从train选择有效率最低的64组HLS作定向检查：云/邻近/影并集76,829像元均未保留六波段有效值，83,652个invalid像元均为六波段nodata。这不是全数据分布估计，但与clear-only配置一致。审计记录见 `data_inspection/cloud_storage_audit.json`。

当前可先研究云掩膜缺失还原；不能声称已经有原始有云/无云监督对。若要恢复残留薄云和阴影污染值，先确认未掩膜观测及参考来源，不能用任意Gaussian替代用户目标。

### 6.3 三个分开的实验设置

1. **R1参考可控：** 从各split独立QA形状库生成云形缺失，移植到可靠区域，在编码前遮挡；以原可靠观测验证还原，称为有参考云形缺失还原。
2. **R2实际孔洞：** 使用当前真实cloud-masked输入做还原。云下没有真值时，仅展示估计、失败案例和可靠区保持性，不编造云区定量误差。
3. **R3原始污染：** 输入保留云/影污染值的观测，需要另行核实已有文件或同传感器源场景的取得方式与参考方案。不同日期参考必须控制地表变化，不默认等于同期真值。

训练mask bank仅来自train；val/test不回流。覆盖率5/10/20%作为候选实验，25/50%为人工扩展压力测试，按实际覆盖率记录。邻近标记与云/影分层，不能将所有邻近像元都称真实污染。掩码已知的R1不能代表未知污染检测能力；另做mask误差诊断。

### 6.4 主证据与表述

主要报告异常/遮挡区还原误差、可靠区保真、边界及光谱/温度物理量。token agreement和率失真用于解释离散机制，不是还原成功的替代指标。对无法观测的真实云下内容，输出是先验条件下的估计；只有具备可信参考才能报告相应定量精度。

HLS QA语义依据：[HLS官方算法说明](https://hls.gsfc.nasa.gov/algorithms/)。详细mask、loss、数据适用边界以 [第一篇实验执行计划](PAPER1_EXPERIMENT_PLAN.md) §7–10 为准。

---

## 7. 研究问题与实验矩阵

### RQ1：EO 数据是否具有不同于自然图像的离散可表达结构？

数据：

- ImageNet-1K RGB；
- COCO RGB，作为复验或附录；
- HLS RGB；
- HLS 6-band；
- LST 1-band patch。

公平控制：

- 相同 patch 数量；
- 相同 `256×256` crop；
- HLS RGB 与自然 RGB 直接比较；
- HLS 6-band 和 LST 使用维度归一化指标；
- 同时报告物理归一化和 per-channel z-score 敏感性分析。

指标：

- RGB/CIELAB gamut occupancy，仅用于 RGB-to-RGB；
- marginal/joint entropy；
- channel correlation；
- PCA 95%/99% explained dimension；
- effective rank/participation ratio；
- local intrinsic dimension；
- patch k-means distortion vs. K；
- spatial power spectrum/high-frequency energy。

有效维度：

\[
d_{eff}=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.
\]

原则：如果数据不支持“EO 更低维”，删除该 claim，改写为“EO 具有不同的量化结构，需要领域特定 tokenizer”。

### RQ2：EO 能否在低码率下高保真离散化？

Tokenizer baselines：

1. Continuous ResNet Autoencoder；
2. 当前 VAR-aligned VQ-VAE；
3. FSQ-AE；
4. 普通单尺度 VQ-VAE；
5. TerraMind tokenizer；
6. EO-VAE，若代码/权重可复现；
7. Proposed CVDT。

码率工作点：

- low；
- medium；
- high。

通过以下参数控制：

- latent resolution；
- codebook size；
- scale schedule；
- token entropy。

必须区分理论码率、经验熵估计与实际熵编码码率：

\[
R_{max}=\frac{N_z\log_2K}{HW},
\]

\[
R_{entropy}=\frac{\sum_s N_s H(z_s)}{HW}.
\]

上述经验熵是按尺度估计的参考量，不等于实际编码文件大小。若报告 entropy-coded bpp，应实现编码并计入必要元数据，熵模型仅用 train 拟合。连续 AE/EO-VAE 必须明确量化精度和存储方式后才能进入实际码率比较；否则只作重建参考。

主要结果使用 rate–distortion curve；条件设置与单模态设置分表，并报告条件成本。

### RQ3：cross-view 是否真正改善噪声稳定性？

模型消融：

| 模型 | HLS | LST | Cross | 离散 | Shared/private |
|---|---:|---:|---:|---:|---:|
| LST-only VQ |  | ✓ |  | ✓ |  |
| HLS-only tokenizer | ✓ |  |  | ✓ |  |
| Channel concatenation | ✓ | ✓ |  | ✓ |  |
| Continuous cross-attention | ✓ | ✓ | ✓ |  |  |
| Discrete cross-attention | ✓ | ✓ | ✓ | ✓ |  |
| CVDT full | ✓ | ✓ | ✓ | ✓ | ✓ |

所有比较匹配：

- decoder 容量；
- 总参数或 trainable 参数；
- 训练步数/GPU hours；
- token bitrate；
- 有效像素 mask。

token stability：

\[
S_{hard}=\frac{1}{N}\sum_i\mathbf1[z_i(x_1)=z_i(x_2)].
\]

同时报告：

- soft assignment JS divergence；
- token flip rate；
- token transition entropy；
- latent cosine similarity；
- reconstruction under corruption；
- signal/high-frequency retention。

### RQ4：离散 token 是否保留物理与下游信息？

#### 任务 A：HLS representation → LST prediction

统一 shallow decoder/head，比较：

- Prithvi-EO-2.0；
- DOFA；
- AnySat；
- CROMA；
- SatMAE（经典可选）；
- CVDT HLS tokens。

设置：

- frozen encoder；
- matched shallow decoder；
- 1%、10%、100% 训练数据；
- 主线仅选择少量可适配的 EFM frozen probes；全量微调不是论文一必做项。

#### 任务 B：原跨卫星轻量探针（降为可选，非本篇主还原任务）

\[
(\text{LR LST},\text{HLS},\text{month})
\rightarrow \text{HR LST tokens}
\rightarrow \text{HR LST}.
\]

论文一主下游改为实验计划E14的像素域云/异常缺失→同分辨率图像还原。此处任务A/B仅作可选探针，不再要求二选一。不加入完整非同期可靠性、跨卫星观测约束或大型生成器，这些属于论文二。

报告：

- token NLL/perplexity；
- masked-token recovery；
- convergence steps/GPU hours；
- reconstructed LST metrics；
- 未见 tile/month 泛化。

### RQ5：是否跨地理、时间和质量条件泛化？

- unseen MGRS tiles；
- month/season holdout；
- HLS source-time/candidate-rank strata；
- valid/cloud fraction strata；
- 热端和冷端子集；
- 不同噪声类型的 train-test mismatch。

---

## 8. 指标体系

### 8.1 HLS

- bandwise RMSE/MAE/bias；
- normalized RMSE；
- PSNR/SSIM；
- SAM；
- NDVI/NDWI/NBR error；
- gradient/frequency preservation；
- masked missing-region error。

### 8.2 LST

- RMSE °C；
- MAE °C；
- bias °C；
- SSIM；
- gradient RMSE；
- within 1/2/5°C；
- top-5%/top-1% extreme error；
- hotspot precision/recall/IoU；
- 同源场聚合保真；不对 MODIS/Landsat 直接施加硬均值守恒。

### 8.3 Token/Compression

- tokens/image；
- theoretical bpp；
- entropy-coded empirical bpp；
- codebook used fraction；
- perplexity/effective vocabulary；
- dead code ratio；
- token stability/flip rate；
- scale-wise entropy and usage。

### 8.4 效率

- total/trainable parameters；
- FLOPs；
- peak memory；
- encoding/decoding throughput；
- training GPU hours；
- feature/token storage per image。

### 8.5 统计

- 主结果至少 3 seeds，重型 EFM full fine-tuning 可先 1 seed、最终关键模型补 3 seeds；
- tile-cluster bootstrap 95% CI；
- 同一样本 paired comparison；
- 避免只报告 image-level 极小标准误。

---

## 9. 关键消融

优先级从高到低：

1. 无 HLS / concat HLS / continuous cross-attention / discrete cross-attention；
2. 无 noise-stability loss；
3. 无 cross-view alignment；
4. shared codebook vs separate codebooks；
5. shared/private split vs single latent；
6. modality dropout on/off；
7. month token on/off；
8. single-scale vs current multi-scale residual VQ；
9. VQ vs FSQ；
10. codebook `K=256/512/4096`；
11. latent `8×8/16×16/32×32`；
12. current 680-token schedule vs精简 scale schedule；
13. gradient/SAM/physical loss on/off；
14. frozen vs jointly trained HLS encoder；
15. shared-token hard equality vs soft alignment。

论文一主文只保留能解释机制的 5–7 个消融，其余放附录。

---

## 10. 主表与主图设计

### Figure 1：问题与方法

- static optical HLS；
- dynamic monthly LST；
- shared structure/private modality tokens；
- cross-view denoising pathway。

### Figure 2：离散可表达性

- intrinsic/effective dimension；
- k-means distortion；
- HLS/LST/natural-image rate–distortion。

### Table 1：Tokenizer 主结果

- HLS 和 LST；
- low/medium/high bpp；
- AE/VQ/FSQ/TerraMind/EO-VAE/CVDT；
- reconstruction、physical fidelity、token statistics。

### Figure 3：Noise–Stability 曲线

- 云/影形状与覆盖率 vs 图像还原；
- 同一还原实验的token agreement辅助诊断；
- modality dropout/missing bands。

### Table 2：Cross-View Denoising

- unimodal；
- concat；
- continuous cross-attention；
- discrete cross-attention；
- full CVDT。

### Table 3：Representation Utility

- Prithvi、DOFA、AnySat、CROMA、CVDT；
- frozen probe/low-data；
- HLS→LST 或 LR-LST+HLS→HR-LST。

### Figure 4：定性结果

- HLS spectral reconstruction；
- LST noise residual；
- hotspot/edge preservation；
- 失败案例。

### Table 4：关键消融

- shared/private；
- consistency；
- modality dropout；
- quantizer type；
- multi-scale。

---

## 11. 两篇论文的执行安排

### 11.1 论文一：CVPR 2027 优先计划

| 日期（2026） | 任务与交付 |
|---|---|
| 9/05–9/12 | 固定数据协议、核实温度单位与旧模型训练成员、检查地理覆盖；统一 loader、mask 和训练归一化 |
| 9/12–9/19 | HLS/LST 小规模 AE/VQ/FSQ；跨视图表示约束与 token 稳定性 pilot，做机制判断 |
| 9/20–10/10 | 主方法定型、三个码率点、条件/非条件公平基线；开始论文写作 |
| 10/11–10/25 | 受控损坏、物理指标、必要消融、一个轻量下游；可选自然图像对照 |
| 10/26–11/05 | 主模型多种子、tile CI、检查失败案例；冻结方法后统一 test 评估 |
| 11/06–11/10 | 锁定主表与注册信息；11/10 AOE 前完成注册 |
| 11/11–11/14 | 完成全文、匿名化和可复现材料，11/14 为内部全文目标 |
| 11/16、11/23 | 分别为官方全文、补充材料截止日期 |

这是计划排期，不表示相关实验已经完成。不得为追赶日期省略数据单位、分组或泄漏核查。

### 11.2 论文二：独立里程碑

1. 在共享数据审计时统计 HLS–目标日期差，以及 MODIS–Landsat 同日观测时差、质量和填补分布。
2. 建立 LR-only、简单拼接和连续 cross-attention 基线，量化加入 HLS 的收益及失败区域。
3. 论文一 tokenizer 冻结后，比较连续表示、普通 VQ/FSQ 和论文一表示；论文二基线无需等待论文一完成。
4. 验证非同期可靠性与跨传感器约束各自的增益，再做完整预测器、多种子和定型测试。
5. 独立决定投稿日期；不将“存在一个 tokenizer”或“接头后能重建”当作论文二完成。

## 12. 算力与优先级

原方案假设可用四卡 H200，该硬件和实际吞吐尚未在本工作区核实。若 9/05–11/16 连续使用四卡，72 天理论上限为 `4 × 24 × 72 = 6,912 GPU-hours`，不是可承诺的实际预算。

- 先测数据读取、单卡训练和四卡训练吞吐，再按实验数量与种子形成预算；原 28 天估算不再适用。
- 论文一主方法、必要基线和三种子优先；论文二先做审计及低成本基线。
- pilot 优先一张卡一组小实验；仅对定型模型使用多卡扩展。
- 自然图像大规模训练、所有 EFM 全量微调、双向模型和大型 VAR 不作为前置任务。
- 分别记录两篇消耗及共享预训练成本，防止论文二的增益比较隐去 tokenizer 训练代价。

---

## 13. 实现组织建议

以下为模型工程建议，当前本地已存在的是数据检查与划分脚本，不代表这些模型模块已实现。论文二新增独立 `asynchronous_guidance.py`、`cross_sensor_reconstructor.py` 和 `sensor_observation_loss.py`，不把完整重建逻辑写入论文一 tokenizer。

建议新增但尚未创建的模块：

```text
src/lstsr_single_image/models/
  hls_tokenizer_var.py
  cross_view_tokenizer.py
  shared_private_quantizer.py        # 完整版可选

src/lstsr_single_image/
  corruptions.py
  token_metrics.py
  spectral_metrics.py

configs/
  hls_tokenizer_vq.yaml
  hls_tokenizer_fsq.yaml
  cross_view_concat.yaml
  cross_view_continuous_attention.yaml
  cross_view_discrete.yaml
  cross_view_shared_private.yaml      # 完整版可选

tools/
  analyze_discrete_geometry.py
  evaluate_rate_distortion.py
  evaluate_token_stability.py
  evaluate_cross_view_denoising.py
  evaluate_efm_probe.py
```

实现原则：

- 保留当前 LST VQ-VAE 作为不可变 baseline；
- 不在 baseline 文件中堆入大量 cross-view 逻辑；
- corruption 必须 deterministic-seed 可复现；
- 训练与评估使用完全相同的物理单位定义；
- 所有 config 保存 git commit、数据 manifest hash 和 normalization hash；
- 每个主表结果能够从单一命令重现。

---

## 14. 与已有工作的比较边界

### Direct tokenizer baselines

- [TerraMind](https://arxiv.org/abs/2504.11171)：多模态 EO、FSQ tokenizer、生成式解码；
- [EO-VAE](https://arxiv.org/abs/2602.12177)：多传感器 EO tokenizer；
- VQ-VAE/FSQ/continuous AE。

### Earth Foundation Models

- [Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732)：HLS-native MAE；
- [DOFA](https://arxiv.org/abs/2403.15356)：波长自适应多传感器模型；
- [AnySat](https://arxiv.org/abs/2412.14123)：多分辨率、多模态 JEPA；
- [CROMA](https://arxiv.org/abs/2311.00566)：对比 + masked reconstruction；
- [SatMAE](https://arxiv.org/abs/2207.08051)：经典多光谱/时序 MAE。

EFM 不应放在纯 codec 表中声称“全面更差”。公平表述为：

> 在相同 HLS 下游任务、适配头和训练预算下，比较连续 foundation features 与离散 tokens 的性能、鲁棒性和存储成本。

[PANGAEA](https://arxiv.org/abs/2412.04204) 已指出没有一个 GFM 在所有 EO 任务上统一最优，因此论文应报告任务和预算条件，而不是宣称全面击败 Prithvi。

---

## 15. 论文 Claim 层级

### 可以直接主张（完成相应实验后）

1. HLS 和 LST 在给定码率下可以被高保真离散表示；
2. EO 与自然图像具有不同的 patch/光谱量化结构；
3. optical context 能改善受控噪声下的 thermal tokenization；
4. cross-view discrete model 优于相同预算的 concat 和 continuous cross-attention；
5. 离散 token 保留 HLS 光谱指标和 LST 极端温度信息；
6. token 可以用于 HLS/LR-LST 条件下的 HR-LST 预测。

### 只有证据充分时主张

- EO 比自然图像“更容易”离散化；
- LST 与 HLS 可以双向相互去噪；
- 离散表征全面优于 Earth Foundation Models；
- 模型消除了真实传感器噪声；
- shared codebook 学到了语义一致的跨模态词汇。

### 不应主张

- 当前单模态 VQ-VAE 已经使用 HLS；
- 当前模型已经是去噪模型；
- 当前 reconstruction 是 LR→HR super-resolution；
- HLS 与月度 LST 是严格同期观测；
- 仅凭颜色分布就证明离散化成立。

---

## 16. 主要风险与预设转向

### 风险 1：EO 并不比自然图像更低维

转向：不主张“更容易离散化”，改为“EO 具有不同的光谱—空间量化结构，普通自然图像 tokenizer 不是最优”。

### 风险 2：HLS→LST 有效，但 LST→HLS 无效

转向：论文聚焦有证据的单方向异常图像还原，不写 mutual/bidirectional；不能将高斯去噪替代去云目标。

### 风险 3：cross-attention 提升来自更多参数

处理：参数匹配、decoder 匹配、concat baseline、frozen HLS features、相同 GPU hours。

### 风险 4：token stability 提升但高频细节被抹平

处理：同时报告 gradient/frequency/extreme metrics；用 Pareto curve 而不是单一 stability 指标。

### 风险 5：shared codebook collapse

转向：独立 codebook + soft/contrastive shared-feature alignment。

### 风险 6：只有人工噪声结果

处理：优先补齐云/异常污染来源审计，区分R1参考可控、R2实际缺失与R3原始污染。只有R1时称云形缺失还原，不写已验证真实去云。

### 风险 7：数据无法完整发布

处理：发布 manifest 构建、split、下载/预处理、corruption 和 evaluation 代码；弱化“新数据集”措辞，改为 benchmark protocol。

### 风险 8：时间不足

处理：优先 asymmetric CVDT、三张主表和一个 token 下游；双向模型、大型 VAR 和更多 EFM 延后。

---

## 17. 最终成功标准

论文一计划形成的完整证据包应满足（不是录用保证）：

1. **不是普通 VQ-VAE 应用**：有明确 cross-view/noise-stability 方法；
2. **两个 EO 模态成立**：HLS 和 LST 都有率失真与物理保真结果；
3. **cross-view 机制成立**：明显超过 concat 和连续 cross-attention；
4. **token 不只是重建**：支持一个物理下游或小型 Transformer；
5. **比较公平**：码率、参数、训练预算、decoder 和 split 都受控；
6. **结论有边界**：不把受控噪声写成真实传感器去噪；
7. **可复现**：固定 split、config、seed、manifest hash 和评估脚本；
8. **主结果稳健**：未见 tile、关键 3 seeds、tile-level CI；未见月份结论仅在另行实施月份留出协议后报告。

如果只得到“HLS/LST 可以被普通 VQ-VAE 重建”，更适合作为遥感 workshop/期刊中的实验基础；如果上述 cross-view、稳定性和下游三条同时成立，则构成论文一的完整主线。

---

## 18. 当前执行清单

已完成：全量解压与文件完整性核验、HLS/LST manifest 配对、原 val 改名 test、从原 train 固定划出内部 val；split seed 为 `20260905`。

接下来按顺序：

1. 核实温度缩放、QA/填补标记、时间字段及地理覆盖；固化共享 `data_protocol.md`。
2. 找到历史训练代码和 checkpoint 来源；主表采用符合当前划分的训练流程。
3. 论文一跑 HLS/LST AE、VQ、FSQ 小规模公平基线，记录重建、物理与 token 指标。
4. 验证训练时跨视图结构约束、noise-stability 和 shared/private 候选；将 decoder-only HLS 融合作为基线。
5. 只有机制 pilot 成立后扩展码率、种子和轻量下游；优先完成论文一。
6. 同步为论文二输出时差/传感器差异审计与 LR-only 基线，不提前把完整重建变成论文一主任务。

---

## 19. 论文二：非同期光学引导与跨卫星热学场重建

### 19.1 独立研究问题与适用边界

设静态光学条件为 \(O_{t_O}\)，MODIS 低分辨率温度观测为 \(L_{t_L}\)，Landsat 高分辨率参考为 \(T_{t_T}\)。这里月份槽位来自实际日期，不是连续时间序列。

\[
(L_{t_L},O_{t_O},\text{available time/quality metadata})
\longrightarrow \widehat T_{t_T}.
\]

- 非同期光学引导：\(t_O\) 与目标日期可能相隔不同季节或年份。
- 跨卫星重建：当前主配对为 MODIS LR → Landsat HR；即使同日，观测时刻、像元响应、质量及产品偏差也可能不同。
- 不把当前配对结果称为“泛化到任意或未见卫星”；新增传感器、标签和高程需另行讨论。
- 训练时 HR 用作监督与评估参考，不称为无噪声真值；推理时不访问 HR 像素、HR token 或目标 QA。
- 目标日期/观测计划可作为任务已知条件；只有从输入或预先确定的计划获得的元数据可进入预测器。HR QA 可以用于训练损失权重和评估分层，但不能形成推理输入泄漏。

### 19.2 候选方法：可靠引导与跨传感器条件重建

```text
MODIS LR LST + input masks/QA ──→ thermal encoder ─────────┐
                                                        ├→ conditional predictor → HR LST
static HLS + HLS masks + time ─→ optical encoder → gate ──┘
```

第一篇 tokenizer 是可替换表示组件，主表包含连续表示、普通 tokenizer 和第一篇 tokenizer。初始版本冻结 tokenizer，再以相同预算比较微调版本；不能只报告所提 tokenizer 的最佳组合。

拟验证两个独立机制：

1. **非同期引导可靠性**：利用输入特征、HLS 质量、季节与日期差估计局部光学权重；光学不可靠时降低纹理注入。仅加入日期 embedding 是必要基线，不自动构成核心贡献。
2. **跨传感器条件建模**：根据已核实的时间/空间配准与产品差异，建立条件预测或稳健观测约束。不能依赖一个可任意拟合的偏差项掩盖错误 HR 重建。

若使用离散版本：

\[
\widehat z_T=P(E_L(L),g(E_L(L),E_O(O),\Delta t,q)\odot E_O(O),\text{time}),\quad
\widehat T=D_T(\widehat z_T).
\]

其中 \(q\) 仅含推理时可得信息。通过在连续版本上加入相同 gate 和时间信息，区分表示收益与重建机制收益。

### 19.3 监督与物理约束

- 基础目标：对有效参考像素的 MSE/Charbonnier，加经过验证的梯度或极值保真项。
- 分层比较 HR primary 与 secondary-fill 区域；QA 加权版本需与不加权版本比较，并固定评估 mask。
- 在 train 上先统计空间匹配后的 MODIS/Landsat 差值、季节偏差和时间差，确定合理的空间聚合算子与约束强度。
- 跨传感器一致性优先作为诊断；只有观测模型得到支持后才尝试带容差的软约束。禁用未经验证的硬块平均守恒。
- 若加入不确定性输出，应检验覆盖率或校准，不能把方差扩大解释成重建改善。

### 19.4 实验与消融

| 实验 | 核心问题 |
|---|---|
| 插值、LR-only 重建网络 | 不使用 HLS 的能力基线 |
| HLS-only + 时间 | 光学条件与季节信息能解释多少目标变化？ |
| LR+HLS concat、连续 cross-attention | 普通融合是否已足够？ |
| 连续 / 普通 VQ或FSQ / 论文一 tokenizer × 无/有可靠性模块 | 表示与重建模块分别贡献多少？ |
| 无时间 / month / 实际日期差 / 完整时间信息 | 非同期机制是否有效？ |
| 无传感器约束 / 经审计的软约束 | 是否解决了真实产品差异？ |
| 正确 / 受控错配 / 平移 / 受损 / 缺失 HLS | 错误引导是否造成温度纹理幻觉？ |
| 日期差、雪/水覆盖、LR质量、HR填补分层 | 实际困难条件下是否仍有收益？ |

受控错配/移位实验用于压力测试，与自然发生的非同期分层分别报告。日期差分层阈值在 train/val 确定，test 不用于选择有利分组。若做月份/季节留出，单独列出训练排除规则，不能把普通跨 tile 测试称为未见月份实验。

任务匹配的相关工作包括 [SwinFuSR](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/html/Arnold_SwinFuSR_An_Image_Fusion-inspired_Model_for_RGB-guided_Thermal_Image_Super-resolution_CVPRW_2024_paper.html) 和 [SIF-CNN-SR](https://arxiv.org/abs/2502.01204)。作为候选基线需核查代码、波段适配与退化假设，不能直接将不同数据集上的发表数值拼在一起。

### 19.5 主表、指标与成功标准

- Table B1：完整跨卫星重建对比，固定输入、训练预算、目标 mask 和测试集。
- Table B2：表示 × 可靠性控制 × 传感器约束消融，量化独立贡献。
- Table B3：真实时差与质量分层、受控错误引导和缺模态鲁棒性。
- Figures：误差空间分布、极端温度与边缘、光学权重和失败案例。权重可视化只是解释材料，不能单独证明机制。
- 指标：RMSE/MAE/bias（单位核实后使用 °C）、梯度误差、极值误差、hotspot 指标、运行成本；多种子与 paired tile bootstrap。
- 成功标准：相比匹配预算的融合基线有稳定收益；非同期与传感器机制各自可消融；错误引导退化减轻且正常引导收益保留。
- 若只有更换 tokenizer 带来增益，应作为论文一的扩展证据，尚不足以独立形成论文二。

---

## 20. 两篇论文的贡献与实验边界

| 内容 | 论文一所有的主证据 | 论文二所有的主证据 |
|---|---|---|
| 离散化 | 量化、结构约束、率失真、码本行为 | 表示作为条件重建系统中的可替换组件 |
| 噪声/缺失 | 云/异常污染与其掩膜缺失的同分辨率还原；token为方法机制 | 非同期跨卫星重建中的观测差异与错误引导 |
| 跨视图 | 配对信息如何改善表示；单模态与条件 codec 分开 | 如何判断引导可信程度并避免纹理误导 |
| 跨卫星 | 交代数据来源；轻量下游背景 | MODIS/Landsat 观测差异与完整重建机制 |
| 下游 | 输入像素域先污染/缺失的图像还原，保持分辨率 | 完整跨卫星LR→HR方法及实际重建主表 |

数据、loader、固定 split、归一化协议、指标代码与 tokenizer 可共享，并透明记录来源。两篇的主要方法、主表和机制消融分别成立，不将同一组实验包装成两个独立贡献。论文一无需完成论文二才能成立；论文二在普通 tokenizer 或连续表示上的比较也必须足以评估自身机制。

自然图像“离散可表达性”对照归论文一；非同期建模和跨卫星观测约束归论文二。完整双向 shared/private tokenizer 是否采用由论文一证据决定，不因为规划了两篇就机械增加模型复杂度。

## 21. 共享成果交接与复现

论文一交接：train-only tokenizer checkpoint、模型配置、量化/反量化接口、码率定义、mask/单位说明、归一化统计、manifest 与 checkpoint hash，以及轻量预测基线。交接时冻结版本，论文二任何微调均单独标注。

论文二交接：明确的输入/目标契约、时差与产品差异审计、连续与普通 tokenizer 对照、可靠性和传感器机制的独立消融、定型预测器与测试脚本。

固定划分文件位置：

- 数据目录：`data_inspection/extracted_metadata/lstsr_tb_final_dataset_20260719/`；
- 划分说明：`splits/README.md`；
- 精确名单：`splits/{train,val,test}_sequences.csv` 和 `splits/{train,val,test}_tiles.txt`；
- 算法及哈希：`splits/split_summary.json`；
- 划分脚本：工作区根目录 `create_validation_split.py`；seed 固定为 `20260905`。

以上是研究计划。除明确列出的数据准备工作外，新增模型、训练、消融和论文结论均未完成。
