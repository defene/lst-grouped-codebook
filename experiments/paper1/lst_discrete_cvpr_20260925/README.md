# LST 离散表示研究：实验设计与验证框架

2026-09-25。**DESIGN ONLY：本目录没有训练入口，不会加载模型、下载权重或启动 GPU 作业。** 用户要求先完整整理方案，执行开关固定为 false。

研究问题：在同等离散载荷和训练预算下，如何保留低图内动态范围温度场的局部差异，并恢复人工缺失的观测？离散表示是主线；颜色编码是待验证假设，VQ/FSQ/LFQ/BSQ 是机制对照。

## 阅读入口

- [完整实验方案](PROTOCOL_ZH.md)：目标、假设、方法、矩阵、统计、停止条件。
- [CVPR 审查标准](REVIEW_CRITERIA_ZH.md)：官方要求与研究建议分开。
- [方法和文献依据](REFERENCES_ZH.md)：原论文和官方代码。
- [工程与结果契约](IMPLEMENTATION_ZH.md)：将来实现时必须满足的接口和验收；不冒称模型已接入。
- [已完成实验及其边界](HISTORY_ZH.md)：历史结果只作探索证据。
- [机器可读总清单](design_manifest.json)、[逐作业配置](experiments.jsonl)、[验证结果](validation.json)。所有作业均未启动。

`build_design.py` 只用 Python 标准库生成配置和检查算术/比较契约；可重复运行，不读图像、不前向、不反向、不启动子进程。命令：

```powershell
.venv-paper1/Scripts/python.exe -X utf8 experiments/paper1/lst_discrete_cvpr_20260925/build_design.py
```

本目录取代 `lst_known_mean_design_20260924` 的 24 作业草案作为**后续设计入口**，不覆盖其历史文档、模型代码或已有运行任务。现有 grouped-VQ 作业属于另一份先前启动的协议，不是本方案已执行的证据。

方案完整不等于训练实现完成。量化器官方版本、物理输出头梯度、实测参数/显存、色域决策、独立确认集均有显式门槛；通过静态配置检查不等于这些门槛已通过。
