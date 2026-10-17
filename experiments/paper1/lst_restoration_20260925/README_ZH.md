# 2026-09-25 LST 复原实验配置

这是当前“离散表征＋传感器型噪声/云形遮盖＋图像复原”研究的配置目录。所有执行开关均关闭，没有启动训练。

- 实现和使用说明：[../lst_restoration/README_ZH.md](../lst_restoration/README_ZH.md)
- 科研协议：[../lst_restoration/PROTOCOL_ZH.md](../lst_restoration/PROTOCOL_ZH.md)
- 候选注册表：[manifest.json](manifest.json)
- 静态校验：[static_validation.json](static_validation.json)

这版取代同日较早的纯设计清单作为当前实现入口，但不覆盖历史实验。下游任务排除；不是“完成174组结果”，也不是自动排队174次训练。参考色域、合成噪声强度、训练配方及资源都尚待授权后的数值预检和正式冻结。
