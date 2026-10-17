# 实现与验收契约

本轮已提供的是设计生成器、配置注册表、对照关系和静态验证，不是已完成全部模型的训练框架。没有隐藏launcher；build_design.py只用标准库读写方案文件。以下是将来实现各模块时的约束。

## 模块边界

| 模块 | 输入→输出 | 必须遵守 |
|---|---|---|
| DatasetAdapter | 冻结record→温度°C、R、ID、tile | 显式HLPv2；只train拟合；保留来源hash |
| MeanCondition | 温度与R/O→float32 μ、中心化值 | oracle/visible/absolute三协议；不能误用完整参考统计 |
| Corruption | ID/epoch/view/R→M、O、可见值 | 模型无关；pixel与patch分轨；mask互斥关系可验证 |
| Representation | 物理场→R/G/C/C_affine；预测→物理场 | version/domain/inverse/tail/Jacobian；不可静默clip |
| InputAdapter | 表示、R、O、μ→原模型输入通道 | 所有卷积组同类型；恒等初始化；额外参数计数 |
| Encoder | 适配输入→32×16×16共同接口或原生latent | 不越过量化器传像素特征 |
| Quantizer | latent→整数indices、量化latent、原生正则 | 状态空间、bit宽、内部维度、尺度、索引回解明确 |
| Decoder | 量化latent、规定μ条件→表示输出 | 公共μ仿射支路，禁止参考图或encoder像素跳连 |
| PhysicsReadout | 表示输出、μ→温度 | 记录投影前后；不得用reference纠偏 |
| Trainer | 冻结job→五轮检查点、呈现账本 | 尾批31按实际样本；恢复RNG/optimizer/scheduler |
| Evaluator | 固定checkpoint+完整split→逐记录充分统计 | 输入/目标/mask跨组一致；2视图不能当独立记录 |
| CodecIO | indices+μ+支持→字节流→indices+μ+支持 | 解码器不读取原数组；报告真实开销 |
| Reporter | 审计通过的结果→独立轨道表、图、成本 | 空项写未完成，不用旧协议数字填新表 |

## Sample 与输出合同

样本至少含：`record_id, tile_id, split, temperature_c, reference_valid, observed_valid, artificial_missing, mean_value_f32, mean_source, representation_id, corruption_fingerprint`。训练目标独立存放，不交给encoder；允许在数据管线用目标计算oracle均温，但字段明确标注这一任务条件。

量化接口至少含：`indices, quantized_latent, token_layout, states_per_token, bits_per_token, quantizer_regularizers`。连续AE不得伪造indices；MAE用可见token和解码布局自己的合同，不能为了统一表格冒称离散token压缩。

结果每记录每视图保存：`n_valid,n_missing,n_observed,n_edges,sum_error,sum_squared_error,spatial_squared_error_sum,gradient_squared_error_sum`和分区统计、参考/预测方差、token统计、sidebits、计时、数值异常状态。缺失0像素时missing指标为null，不写0。数据集RMSE通过充分统计汇总，不对逐图RMSE简单平均后称pooled。

## 冻结与状态机

`design_only → implementation → preflight → frozen → authorized_to_run → training → evaluating → audit → visual_review → complete`。

当前仅到design_only；静态配置通过并不自动前进。用户“先不要跑”是明确执行约束。不得为填预检清单执行模型forward/backward、GPU测速或短程训练。后续用户要求开始后，再履行冻结/接入条件，常规可逆实现工作不必反复索取批准。

历史监督进程与新方案完全分离。当前只发现另一轮grouped-VQ已有运行，不据此启动、重启、停止或改写任何作业。这里只记录本方案没有新增训练，不声称机器上没有训练。

绝对温度合同：x=T−c_train，c_train由train总体有效像素拟合；缺失填0，表示沿用F(x)，读出c_train+inverse_F(y)，没有μ加回捷径。μ仍是输入/decoder公共条件，信息与oracle残差组相同；绝对组初始化为c_train，不冒称每图与残差组有相同物理初始预测。所有训练监督最后回到T域。c_train具体值尚未在本轮拟合，不编造常数。

读出顺序为raw_inverse→训练loss→评价field-domain projection→评价temperature projection。R/G raw_inverse不clip，C内含线段投影；主损失加共同Jacobian归一化表示辅助项，权重候选0.01，另有4个weight0对照。该项及C跨段可达性未通过梯度预检前，不执行C作业。

## 技术门槛（本轮未执行）

1. 数据：清单/hash/单位/归一化与AGENTS一致；完整有效温度域核验；footprint和重复审计记录已知/未知，不强行写零泄漏。
2. 条件：oracle/visible计算只读各自支持；μ以同一个float32数值中心化/加回；人工mask的生成不依赖温度纹理。
3. 映射：域内往返、端点、线段交界；曲线中心并列逆解；C零残差偏置；NaN/Inf；投影梯度有限差分与错误分支恢复。float32最大往返误差目标≤1e−3°C，失败时查算法而非放宽到模型RMSE量级。
4. mask：R/O/M一致、填充值先物理后映射、不同模型输入信息对齐；MAE真实drop/insert/reorder且不二次遮盖。
5. 量化：FSQ所有levels组合mixed-radix index唯一可回解；LFQ/BSQ零值sign约定；BSQ近零向量稳定；VQ更新/commitment不可混EMA与gradient。
6. 瓶颈：只由indices/模型/规定侧信息即可解码同样输出；无连续latent偷渡；共享mask支持是否传输有明确账目。
7. 公平性：每表示/方法有效参数与latent实测；共同骨干参数快照；scratch零物理初态；pretrained头保持原始行为，解码缩放只执行一次。
8. 训练预算：用67条合成记录×2轮测试尾部加权和断点等价；这属于将来的训练器测试，不在本轮运行。正式不得把67条试验当完整epoch。
9. 正则/目标：每项reduction、coef、有效像素/边分母；FSQ无VQ损失、BSQ不强制commitment；mask区域无泄漏。
10. 成本：资源探针只在未来获准后进行；采样显存和会话区间；异常退出保留可证耗时下界，不编造退出精确时刻。

## 比较与统计发布门槛

每个表格cell必须有：同一协议版本、数据hash、物理观测/目标/mask hash、mean source/value hash、表示编码hash、实际checkpoint hash、训练初始化来源、完整unique record/view计数。前3类在相同任务条件下跨模型一致；mean hash仅相同均值条件一致，表示编码hash只在相同表示及均值条件下一致。不得要求R/G/C编码张量hash相同。每val11684、test11757；clean两相同视图可复用，不增加有效样本量。

在数据集已经有历史test观察的前提下，把本轮test执行放到方法冻结后仍有价值，但不恢复其全新盲测地位。独立确认集需另有成员证据。

先按seed汇总所有预定contrast，再按tile配对重采样；支持像素加权与tile宏平均两个不同estimand。不得以困惑度、token稳定性或看起来锐利代替物理误差。3个训练seed只是拟定稳健性方案，不以其数量声称充分统计功效。

## 模型版本和预算待填项

每官方repo需固定commit、许可和本地文件hash；只有URL不等于完成依赖固定。每预训练VAE记录原模型训练来源限制；模型权重下载尚未作为本轮工作执行。

当前87,225,699是旧多尺度RGB候选的meta计数；新单尺度投影、mask/mean适配器、各量化器之后必须重数。SD旧83,653,863也不含本版新适配器。严禁把它们复制成新框架“已经等参数实测”的结果。

若采用LFQ/BSQ原生entropy近似，必须固定是否按组近似联合状态、温度、系数、采样规模和精度，并公开与原实现差异；只有量化公式而没有这些参数，不能冻结正式作业。
