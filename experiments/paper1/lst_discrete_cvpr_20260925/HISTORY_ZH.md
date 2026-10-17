# 已有实验与新研究的关系

这是证据索引，不是新协议的结果表。入口为 [历史总览](../../../runs/experiment_overview_20260924/REPORT_ZH.md)。新框架本轮训练数为0。

| 历史实验 | 已提供的探索证据 | 不能替代的新证据 |
|---|---|---|
| var_vs_prithvi_ae_full | 卷积多尺度VQ与全token连续Prithvi结构AE的LST/HLS重建 | 不是真正Prithvi MAE；旧v1和旧预算不能填当前主表 |
| codebook1024/full | 4096→1024和码本使用分析 | 改K不等于解决图内细节，也没有FSQ/BSQ对照 |
| lst_residual_20260913 | 四组短程温度残差/空间损失试验，后来补全量评估 | 2000步不是5完整epoch；旧训练成员含当前val |
| lst_v2_5epoch_20260915 | v2五轮的绝对温度/残差×空间损失对照 | 均温和噪声协议不同，不替代当前oracle均温＋随机像素主表 |
| lst_residual_scale_20260918 | 固定残差缩放在已有空间4倍基础上的增益很小 | 不能预设缩放或颜色必然有效 |
| lst_sd_color_20260921 | 自然预训练SD经微调可重建LST；该轮asinh灰度优于asinh RGB | 连续大latent、可见均温、QA形状缺失、非线性表示，不是新等载荷结论 |
| openimages_hlp_color_matched_20260917 | 自然图与HLP的图内/图间变化分布不同 | 分布差异不是“转RGB必然学好”的因果证据 |
| lst_grouped_vq_20260924 | 另有先前启动的grouped-VQ作业；本轮只读发现仍在训练 | 未完成状态不能填最终结果；不属于本140格，也不自动等同于FSQ |

可核文件：

- `runs/paper1/lst_v2_5epoch_20260915/REPORT_ZH.md`
- `runs/paper1/lst_residual_scale_20260918/REPORT_ZH.md`
- `runs/paper1/lst_sd_color_20260921/REPORT_ZH.md`
- `runs/openimages_hlp_color_matched_20260917/REPORT_ZH.md`
- `runs/paper1/lst_grouped_vq_20260924/status.json`（状态会变化，不能当完成证据）

新设计使用的现有train事实为残差标准差2.6950881812941048°C和A120中央单段比例90.7373%。原统计文件与hash见前一版 `lst_known_mean_design_20260924/design_manifest.json`。这里是复用已经完成的训练集统计，没有本轮重新跑全量数据实验。

目前缺少的主要证据：同骨干同载荷VQ/FSQ/LFQ/BSQ矩阵；新表示的有效逆解/梯度；同信息均值旁路；真正MAE patch补全；多seed；实际码流；独立确认集。旧代码内已有FSQ组件和smoke记录不等于上述正式对照完成。

跨版本公平性必须显式标注。已有test长期被观察；未来不得把旧test重新包装成首次未接触确认集。旧v1模型见过当前val记录，相关val数值不是独立验证。
