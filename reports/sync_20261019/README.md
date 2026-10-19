# 代码、进度与交付索引补充（2026-10-19）

本次同步不启动训练。已有研究结果与10月17日快照一致：普通VQ和纯EMA各完成5轮；EMA+maintenance仅完成1轮并停止；Grouped K256未开始正式训练，K512仍为准备/停止状态，不补标为已训练。

- [现有结果说明](../local_vq_ema_20261017/README.md)、[保存的队列状态](run_states.json)、[逐轮验证和选模回执](receipts/outputs/ema_local_full_20261015_attempt01/)、[K512回执](receipts/outputs/ema_local_k512_full_20261015_attempt01/)。
- [39模型本地推理工具说明](../../outputs/torch_models_20261013/README.md)、[模型身份清单](../../outputs/torch_models_20261013/MODEL_MANIFEST.json)、[原CPU加载核验](../../outputs/torch_models_20261013/local_validation.json)。本次补入工具和对应冻结源码/配置，未重复运行39模型加载或完整验证。
- [完整模型/OpenImages交付说明、清单与SHA-256（私有历史仓库）](https://github.com/defene/lst-research-legacy/blob/main/docs/delivery_20261019/README.md)。模型ZIP为294.68 GiB，OpenImages ZIP为66.14 GiB；ZIP本体均未上传。

39模型仍为原始第5轮推理权重，不追溯标为验证最优；需另行取得本地模型包，单独clone本仓库没有权重。实际测试使用 `--input` 提供数据；历史示例画廊不随Git上传。原说明中的oracle mean约束、来源路径和浮点精度边界保留。

10月17日清单绑定的首页和校验脚本原字节保存在 [historical_sync_20261017](../../docs/historical_sync_20261017/)。原验证入口对这两项核验历史副本，其余原清单逐项照常核验；冻结清单没有重写。
