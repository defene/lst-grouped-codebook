# LST loss 修复候选版（2026-10-14）

状态：代码与禁用配置已准备，本地 CPU 检查已执行。没有提交服务器作业、启动真实数据训练或上传 W&B。尚不能宣称验证质量提高。

这是独立版本 `lst-restoration-lossfix-20261014`，来自此前未启动的 epochval 版本。历史源码、配置、权重和结果保持冻结。旧 epochval 版本也不修改。

## 三项机制变更

|对象|新配置|行为与限制|
|---|---|---|
|C / C_power / C_asinh|`colour_inverse=trilinear_v1`|在 RGB 立方体内对八个路径顶点的位置作三线性插值，消除硬最近段切换；在原编码路径上准确还原。立方体外前向夹取，反向沿用已明确注册的端点代理梯度。原先路径外的预测会改变，因此是新协议，不能与旧权重混标。|
|从零训练的普通 VQ|`vq_geometry=unit_rms_train_init_v1`|每个 latent token 和码向量按单位 RMS 归一化；首个训练 microbatch 初始化码本。保留可复现的离散索引解码与 STE。没有自动替换死码，不能保证训练后仍充分利用码本。|
|VQ-drifting|`drift_reconstruction_gradient=encoder_and_codes_v1`|保持前向离散码及原 drifting 能量不变，保留重建对选中码的梯度，并增加重建到 encoder 的恒等代理梯度。明确称为本项目的梯度对照版本，不是原方法的等价实现。|

VQGAN 预训练码本及原生量化几何保持不变；没有把归一化 VQ 强加给预训练权重。Grouped 量化器也保持原机制，仅对应 C 表示使用新逆映射。C_power 的非线性逆函数仍可能有大梯度，消除分支跳变不等于解决其全部优化困难。

普通 VQ 的初始化只读取训练 latent，没有验证集拟合。初始化标志和累计使用计数存入 checkpoint；eval 不改动这些状态。单位 RMS 减少未约束的量化尺度变化，但也改变表示几何，需要与原版比较。first-batch 初始化后的高利用率不是长期码本健康的证明。

## 训练目标与新日志

仍为 `pixel + 0.1*gradient + bottleneck + 1e-6*KL`，representation 权重仍为零，不靠引入新的辅助目标掩盖问题。

- 每次更新记录真实加权 `total` 和全模型裁剪前梯度范数。
- VQ 记录活跃码数、perplexity、latent/codebook RMS；新普通 VQ 另外记录归一化前 RMS 和累计活跃码。
- 第 1 次及每 100 次更新，对具备 prequantized latent 的模型分别测量加权 pixel、gradient、bottleneck 到该 latent 的梯度范数，并标记是否连通。
- 这些分项梯度范数是各 microbatch 的范数按样本数加权的均值，不是累积后参数梯度的范数；仅用于同样 batch 设置下的机制诊断。
- C 系列记录 decoder 输出超出 RGB 立方体的有效像素比例，便于判断代理梯度是否频繁被使用。

## 数据、验证和检查点

HLP v2 显式使用 `processed_data/hlp_split_v2_20260914`；218,975 train、11,684 val，使用 train-only 归一化，不使用 test 选模。维持 5 个完整 epoch、有效 batch 64、每轮尾批 31 条、总计 17,110 次更新。

每轮保存不可变 checkpoint 后完整验证 clean / pixel10，各两个固定视图；唯一选模指标为 pixel10 raw 像素加权摄氏度 RMSE，相同分数保留较早轮次。保留 5 个 epoch、last、best 和选模证据。验证失败不得进入下一轮；恢复时补完待完成验证。验证恢复模型模式和随机状态，不增加训练预算。最终轮次和验证最优分别报告。延续 oracle mean 条件，不能将结果直接视为部署效果。

## 配置与成对检查

`runs/paper1/lst_restoration_lossfix_20261014/configs/` 含 39 个候选配置；`controls/` 含 4 个机制对照，所有 execution 和 tracking 均禁用且使用新 run ID。

`CONTROLLED_CHECK_PLAN.json` 固定四组：SD_C 的旧/新逆映射、VQ_G 的旧/新量化几何、drifting_G 的旧/新梯度路径、Grouped C_power 的旧/新逆映射。每对只改变指定机制及运行身份，数据/初始化种子/预算相同。VQ 候选同时改变归一化和初始化，其结果只能归因于整套 recipe，不能分辨两者各自贡献。

这些配置仍是正式 5 轮预算；不能以它们伪装短程筛选。真实数据短程试跑尚未执行，若开展应另行建立有明确预算的技术检查入口。

## 本地复查命令

```powershell
.venv-paper1/Scripts/python.exe -m pytest tests/test_lst_lossfix_mechanisms_20261014.py tests/test_lst_lossfix_epoch_validation_20261014.py -q
.venv-paper1/Scripts/python.exe -m experiments.paper1.lst_restoration_lossfix_20261014 preflight --config runs/paper1/lst_restoration_lossfix_20261014/configs/vq_G.json --part loss-mechanisms --output outputs/lossfix_local_recheck
```

`loss-mechanisms` 和 `epoch-validation` 只运行 CPU 合成测试，不读取真实数据、不加载完整预训练模型。正式启动还必须完成与源码和配置哈希绑定的完整模型、显存、数据、预算、量化机制和验证检查；新增加的 loss_mechanisms 是强制门槛。当前测试通过不等于这些 GPU/真实数据门槛通过。

证据见 `outputs/lossfix_verification_20261014/LOCAL_VERIFICATION.json`；该文件会记录实测通过项及未执行项。父版本源码哈希保存在同目录 `parent_source_hashes.json`，祖先冻结来源保留在 `SOURCE_ORIGINS.json`。
