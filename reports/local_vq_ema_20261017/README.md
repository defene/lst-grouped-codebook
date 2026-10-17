# 本地 VQ / Grouped / EMA 实验状态（2026-10-17）

全部训练已按用户要求停止；本次提交只同步源码、配置和现有报告，不启动或恢复任何实验。

## 已有正式结果

HLP v2：train 218,975，val 11,684；G 表示，LST + mask，oracle clean mean 条件。
每轮完整 val，clean/pixel10 各两个固定视图；按 pixel10 raw 有效像素加权摄氏度 RMSE 选 best，平分取早轮次。未用 test 选模。
正式预算为 5 个完整 epoch、有效 batch64（micro8），每轮3422次更新、尾批31条。

| 模型 | 已验证预算及状态 | best epoch | clean RMSE °C | pixel10 RMSE °C | pixel10 MAE °C | pixel10 已用码 |
|---|---|---:|---:|---:|---:|---:|
| vq_G_baseline | 5/5，已完成 | 5 | 1.263869 | 1.236882 | 0.761199 | 24/1024 |
| vq_G_ema | 5/5，已完成 | 5 | 0.668888 | 0.635862 | 0.397890 | 1009/1024 |
| vq_G_ema_maint | 1/5，已停止 | 1 | 0.892492 | 0.883844 | 0.647035 | 1024/1024 |

原始 VQ 与纯 EMA 均完成 5 轮，best 都是最终第5轮。EMA+K-means+死码重置仅完成第1轮验证；不能与5轮结果作等预算结论。
同为第1轮，纯EMA的pixel10 RMSE为0.789700°C，初始化+死码重置组为0.883844°C；后者同时增加两个机制，不能据此单独归因。
Grouped K256 三组未开始正式训练。K512 的 VQ 与 Grouped 各三组只完成配置与CPU检查，GPU预检和正式训练均未开始。
Grouped K512 表示8组、每组512码；内部旧路由名 grouped_g8k256 不代表实际K。它与普通VQ不是等码率比较。

## 报告入口

- [正式实验逐轮指标、选模与配置](../../outputs/ema_local_full_20261015_attempt01/)
- [K512配置与停止状态](../../outputs/ema_local_k512_full_20261015_attempt01/)
- [EMA六组短程pilot](../../outputs/ema_local_pilot_20261014_attempt01/RESULTS.txt)：128更新、2048训练样本呈现、64条验证子集，不能与完整预算结果混用。
- [39模型历史结果索引](../../outputs/torch_experiments_20261013/results_index.json)：历史最终轮次评估，不能重新标记为逐轮验证选出的最优模型。
- [39模型loss审查](../../outputs/loss_diagnostics_20261014/FINDINGS.txt)
- [Grouped与普通VQ对照](../../outputs/grouped_vs_vq_20261014/COMPARISON.txt)
- [训练低点及checkpoint存留审查](../../outputs/training_lowpoints_20261014/ALL_39_LOWPOINTS.txt)
- [本地50例报告](../../outputs/local_cases50_all39_20261014/RESULTS.txt)：子集诊断，不是完整独立验证。
- [数据集报告](../../data_inspection/FINAL_DATASET_SUMMARY.md)

## 发布边界

新增内容包含源代码、研究配置、文本/表格/图表报告和验证汇总；不包含原始数据、prepared数据、预测数组、模型权重、训练断点、完整训练日志、虚拟环境或缓存。
既有A06/A07仓库文件保持历史语义。原始导出清单保留；旧AGENTS字节另存于 docs/historical_export_20261004，verify_export.py验证这一历史副本。
仓库原有A06/A07 registry保持冻结；当前工作区完整registry另存为 docs/workspace_registry_20261017.json，里面的旧数据口径按历史版本解释。

这是审计与源码同步，不是迁移后可直接续训的包。配置保留原始Windows绝对路径和哈希，依赖数据及checkpoint不在本次上传中。
服务器实际部署需另行绑定路径、环境与数据，重新做预检；不要直接运行旧启动命令。
两个本机队列的 STOP 标记保留。未提交SSH作业，也未恢复本地训练。

## 本次上传验证

在导出仓库中运行原A06/A07相关检查及EMA/K512检查，共49项CPU测试通过，详见[JUnit记录](unit_tests.xml)。源码语法、报告JSON、文件SHA256、完整验证记录及停止标记已核验；原始661文件导出校验仍通过。这些检查不代表全部新增历史实验入口都经过运行测试，也不代表服务器GPU或续训验证。
