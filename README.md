# LST 组合码本实验：A06 / A07

新增设计文档：[Prithvi + RAE 14 层 ViT decoder 方案](experiments/paper1/prithvi_rae_20261005/README.md)（2026-10-05，仅方案与显存估算，训练器尚未实现，未提交实验；不属于下述 A06/A07 历史结果或原始导出 manifest）。

本快照包含 single1024、g2k32、g4k256、g8k16、g8k256 五个已完成配置的实验框架和结果；没有启动新训练。

- [完整中文汇总](runs/paper1/lst_grouped_summary_20261002/REPORT_ZH.md)
- [9页PDF](output/pdf/grouped_codebook_A06_A07_summary_20261002.pdf)
- [A06原始报告](runs/paper1/lst_grouped_vq_20260924/REPORT_ZH.md) / [A07原始报告](runs/paper1/lst_grouped_vq8_20260928/REPORT_ZH.md)
- [A06协议](experiments/paper1/lst_grouped_vq/PROTOCOL_ZH.md) / [A07协议](experiments/paper1/lst_grouped_vq8/PROTOCOL_ZH.md)

## 范围与主要结论

HLP v2，train 218975 / val 11684 / test 11757；仅 LST + mask。seed17，从零5完整epoch，17110更新/模型，固定epoch5权重，两视图完整评估。A07复用的A06 g4结果不重复计数，合计20个唯一评估单元。

| 配置 | bit/图含均温 | test clean RMSE °C | test noisy10 RMSE °C |
|---|---:|---:|---:|
| single1024 | 6832 | 0.948149 | 0.985072 |
| g2k32 | 6832 | 0.805381 | 0.861624 |
| g4k256 | 21792 | 0.693595 | 0.757358 |
| g8k16 | 21792 | 0.625777 | 0.699742 |
| g8k256 | 43552 | 0.623212 | 0.696104 |

同码率分组改善总体误差；缺失区收益较小。g8k256索引位数翻倍而总体误差仅小幅下降。单seed结果，不能宣称多种子稳定优势；token多样性不等于正确恢复细节。

## 框架与复现边界

`lst_grouped_vq` 是冻结训练、量化、评估核心；`lst_grouped_vq8` 是A07配置和调度；`lst_grouped_summary` 是五配置报告与新case生成器。A04/A05和少量旧调度代码是依赖，不代表额外实验纳入本报告。eo_data/eo_denoise及上游许可证原样保留。框架中的Prithvi类是公共依赖，五个模型均未使用Prithvi。

安装适合本机的PyTorch/CUDA，再安装 requirements.txt。此文件是依赖清单，不是跨平台锁定环境。源码、配置和历史manifest按字节保存；配置保留原机器绝对路径，监测工具部分针对Windows。**这是一份可审计历史快照，不是迁移后直接启动的训练包。**

只跑单元检查（不启动训练）：
```sh
python -m pytest tests/test_lst_grouped_vq.py tests/test_lst_grouped_vq8.py tests/test_lst_v2_epochs.py tests/test_lst_residual.py -q
```

未包含原始影像、prepared数据本体、A05 QA形状库或训练权重（单检查点约1.3GB）。成员清单、数据统计证据、输入摘要和检查点SHA256保留。复现预测须另行部署数据、QA库及对应epoch5检查点，并核对哈希。迁移/再训练必须另建run与配置，显式指定HLP v2、新train归一化、split-local QA和准确epoch边界；不得覆盖本快照历史结果或直接执行旧supervise启动队列。

## 原始证据

包括最终20单元metrics、逐样本误差、逐组逐尺度token频数、记录ID、每轮验证汇总、训练日志、门禁、完成审查、恢复事件记录、GPU时长和历史/新case图。体积较大的JSONL和划分CSV使用无损gzip，解压后字节SHA256与原文件相同。每轮验证的逐样本日志、权重和原始数据未打包；其他外部依赖/未附文件由历史manifest描述。源报告保留原始路径，缺少外部资产时不能直接重跑report/gallery。

`EXPORT_MANIFEST.json` 记录所有复制文件的来源与SHA256。可执行 `python verify_export.py` 验证上传副本；不加载模型、不写历史结果。历史registry只导出A06/A07两条记录。
