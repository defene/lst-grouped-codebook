# LST 复原实验配置：2026-09-27

本目录是静态、未启动的174组候选配置。连续 AE/VAR 多尺度暂缓，新增 EO-VAE-LST、TerraMind-FSQ-LST 各6组，均从零训练。自然图像模型维持预训练路线。

所有 execution.enabled=false；源码语法/配置检查通过，未初始化模型或运行数值检查。地理模型官方依赖未绑定，参数量、显存和运行正确性待验收。

详见 [地理架构说明](../lst_restoration/GEO_BASELINES_ZH.md)、[统一框架](../lst_restoration/README_ZH.md) 和 static_validation.json。20260925 目录保留为旧设计快照。
