"""Package already computed diagnostics; does not train or evaluate models."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/loss_diagnostics_20261014'
data = json.loads((OUT / 'diagnostics.json').read_text(encoding='utf-8'))
probes = json.loads((OUT / 'mechanism_probes.json').read_text(encoding='utf-8'))
models = data['models']
by_name = {m['model']: m for m in models}
baseline = models[0]['validation']['pixel10']['observation_mean_fill_baseline']['pixel_weighted_rmse']
rows = []
for m in models:
    row = {'model': m['model']}
    row.update({f'epoch{i+1}_pixel': e['pixel']['mean'] for i,e in enumerate(m['epochs'])})
    row['epoch5_bottleneck'] = m['epochs'][-1]['bottleneck']['mean']
    row['epoch5_total'] = m['epochs'][-1]['total']['mean']
    row['clean_val_rmse_C'] = m['validation']['clean']['raw']['pixel_weighted_rmse']
    row['pixel10_val_rmse_C'] = m['validation']['pixel10']['raw']['pixel_weighted_rmse']
    row['worse_than_mean_fill'] = row['pixel10_val_rmse_C'] > baseline
    rows.append(row)
assert len(rows) == 39
assert sum(r['worse_than_mean_fill'] for r in rows) == 20
header = list(rows[0])
table = ['39 个模型：按原始配置重建 loss；所有验证分数来自第 5 轮最终权重。',
         'epoch*_pixel 为按实际样本数加权的归一化重建 MSE，不是摄氏度 RMSE。',
         f'Pixel10 保留观测像素、缺失处使用同一 oracle mean 填充的基线 RMSE = {baseline:.9f} °C。',
         'worse_than_mean_fill = True 表示最终模型 RMSE 大于该基线。', '', '\t'.join(header)]
for row in rows:
    table.append('\t'.join(f'{v:.9g}' if isinstance(v, float) else str(v) for v in row.values()))
(OUT / 'ALL_39_LOSS_SUMMARY.txt').write_text('\n'.join(table)+'\n', encoding='utf-8')

report = '''39 个模型训练 loss 审计（2026-10-14）

结论：确实存在训练失稳、码本利用不足、重建梯度路径与目标不匹配等实质性问题。
不能仅增加验证，再按原设置重跑全部 39 个模型。本次没有启动训练或提交 GPU 作业。

范围与口径
覆盖 39 个模型、每个 17,110 次更新，共 667,290 条逐步日志。
每个模型训练 5 个完整 epoch、1,094,875 次记录呈现。均值按实际 batch 样本数加权，尾批 31 条。
HLP v2 独立验证集 11,684 条、147 tiles，每条 2 个固定视图；未使用 test 调参或选模。
真实总目标 = pixel + 0.1*gradient + bottleneck + 1e-6*KL。
representation 项权重为 0，日志里即使很大，也不构成实际优化目标。
pixel 是逐图有效像素上的归一化 MSE，再对图求均值；归一化尺度为 2.6950881766°C。
最终验证 RMSE 则按全体有效像素聚合，二者不能直接混为同一数值。
全部日志字段均为有限值，但没有 NaN/Inf 不代表收敛正常。

1. SD_C：中后期出现持续的大幅失稳
第 1–5 轮 pixel 均值：2.436784、0.094613、0.229703、34.768226、0.748563。
第 4 轮比第 3 轮增大约 151 倍；12701–12800 步的 100 步均值达到 509.020。
这不是单个 batch 的孤立尖峰。最后 1000 步均值仍为 0.648773。
最终 pixel10 验证 RMSE 为 1.989577°C。
对照 SD_C_affine：0.024811、0.007319、0.006375、0.005905、0.005681，持续稳定下降；验证 0.204150°C。
SD_G、SD_G_asinh 也比较健康，不能把全部 SD 实验一概视为异常。

2. 原始 VQ：码本利用严重不足，目标数值尺度失衡
完整验证共只使用 29–42/1024 个码，即约 2.83%–4.10%。
VQ_G 第 5 轮 bottleneck 均值 30.277455，pixel 均值 0.218090，约相差 139 倍。
码本项采用 codebook MSE + 0.25*commitment MSE，重建通过 STE 回传 encoder。
应检查 latent/codebook 尺度、各项梯度范数与活跃码的演变。
损失数值大并不能单独证明其梯度占主导；历史日志没有保存各项梯度范数。

3. VQ-drifting：高码本利用率并没有带来有效重建
完整验证使用了 1024/1024 个码，但 8 个模型的 pixel10 验证 RMSE 仍为 2.25–2.68°C。
例如 G_asinh 的逐轮 pixel：0.962155、0.848024、0.850226、0.891006、0.944709，后期反而变差。
冻结实现直接返回离散 codes，没有 STE。CPU 自动微分探针确认：
重建项到 encoder latent 的梯度不存在；普通 VQ 同一探针存在该梯度。
drifting 的 bottleneck 项仍可训练 encoder，因此不是整个 encoder 完全冻结。
这属于当前实现的目标/梯度路径与重建任务的匹配问题；不能据此否定所有 drifting 方法。
drifting energy 可为负，负号本身不是错误。

4. C 温度解码的前向不连续性已在合成输入中复现
最近颜色路径段的硬选择会切换分支。编码扰动 L2 仅 0.0001306301，
温度却从 -102.857147°C 变为 +102.858994°C，跳变 205.716141°C。
相同扰动对 C_affine 仅造成 0.000720059°C 变化。
现有 surrogate backward 没有消除硬选择的前向不连续。
这是机制存在的证据，不是历史每次 loss 尖峰的因果证明；异常时刻权重已被覆盖。
VQGAN_C 也有巨大尖峰；grouped_C_power 五轮 pixel 从 1.208586 到 0.995941，学习效果很差。
但 grouped_C 及若干 C 的变体能够正常改善，不能将全部 C 系列归为同一种失败。

5. 最终结果支持“训练/目标确实有问题”，不能简单称为过拟合
Pixel10 中 10% 有效像素缺失；保留 90% 观测像素，用相同 oracle mean 填补缺失的基线为 0.862861°C。
39 个模型中 20 个比这个不训练的基线还差：SD_C、SD_C_power、全部 8 个 VQ、
全部 8 个 VQ-drifting、VQGAN_C、grouped_C_power。
核对了全部模型的验证成员、物理输入和 mean 的哈希，三者分别一致。
这是同一 oracle mean 条件下的比较，不能直接外推为实际部署表现。
多组模型在训练集上也没有学好；缺少中途验证曲线，不能证明或排除过拟合。

为什么不应按训练日志的单步最低点选模
同一训练样本顺序会导致许多模型在同一步出现低点；此前核对有 25 个模型的重建最低点在 15814 步。
单步值混入 batch 难度与尾批规模，不能代表整体验证性能。
总 loss 还受码本正则/能量项影响，最低总 loss 未必对应最低重建误差。
旧模型只保留最后权重，不能把日志低点补成真实验证最优 checkpoint。

后续处理建议（尚未执行）
优先排查 C 解码连续性与输出约束；比较 VQ 各项梯度尺度、latent/codebook 尺度；
为当前 drifting 实现做有/无重建梯度路径的受控短程比较。
以 SD_C_affine 或 SD_G_asinh 的稳定曲线作为参考，再决定正式 5 轮实验配置。
正式训练应完整验证并保留逐轮与验证最优权重，而不是再次覆盖为单个 last.pt。

文件索引
ALL_39_LOSS_SUMMARY.txt：全部模型的逐轮 pixel、末轮瓶颈/总 loss、最终验证 RMSE。
anomalies_overview.png：代表性异常与健康对照，包含原始更新值及 100 步均值。
各家族 *_loss_components.png：全部 39 个模型的加权 loss 分解曲线。
train_vs_validation.png：训练末期重建误差尺度与完整验证的对照，标明聚合口径不同。
diagnostics.json：逐轮分布、100 步块、训练窗口、原始日志相对路径与验证来源。
mechanism_probes.json：CPU 梯度及 C 解码合成探针；没有训练或优化器更新。
'''
(OUT / 'FINDINGS.txt').write_text(report, encoding='utf-8')

def item(id, title, files, definitions, caveats, columns=None, rows=None, methods=None):
    q={'id':id+'-evidence', 'source':{'label':'冻结实验日志、配置与最终验证',
       'files':[{'label':f} for f in files],
       'metricDefinitions':[{'definition':s} for s in definitions], 'caveats':caveats}}
    if columns is not None: q.update(columns=columns, rows=rows)
    if methods: q['methods']=methods
    return {'id':id, 'title':title, 'queries':[q]}

items=[item('loss-curves','39 个模型的重建曲线与 loss 分项',
    ['diagnostics.json','config.original.json','train_*.jsonl'],
    ['逐轮 pixel 为归一化重建 MSE，按实际 batch 样本数加权。',
     '实际总 loss 为 pixel + 0.1×gradient + bottleneck + 0.000001×KL；representation 权重为零。'],
    ['训练 MSE 与完整验证的像素加权 RMSE 聚合口径不同。','损失项数值大小不能证明各项梯度范数的大小。'],
    ['model','epoch1_pixel','epoch2_pixel','epoch3_pixel','epoch4_pixel','epoch5_pixel','epoch5_bottleneck'],rows,
    [{'language':'calculation','code':'epoch_mean = sum(actual_batch * logged_loss) / sum(actual_batch)\n39 models * 17110 updates = 667290 updates'}]),
 item('gradient-path','VQ-drifting 的重建梯度不回传 encoder latent',
    ['stage2_models.py','quantizers.py','probe_loss_mechanisms_20261014.py','mechanism_probes.json'],
    ['CPU 小型张量探针检查单独重建项是否能对 encoder latent 求导。'],
    ['冻结实现的 drifting 项仍对 encoder 提供梯度。','探针没有训练真实模型，尚未通过受控实验测量修改后的收益。'],
    ['quantizer','reconstruction_latent_gradient_exists','gradient_norm'],
    [{'quantizer':k,'reconstruction_latent_gradient_exists':probes[k]['reconstruction_to_encoder_latent_grad_exists'],
      'gradient_norm':probes[k]['reconstruction_to_encoder_latent_grad_norm']} for k in ['gradient_vq','drifting_vq']]),
 item('c-inverse','C 表示的温度解码存在不连续性',
    ['representations.py','probe_loss_mechanisms_20261014.py','mechanism_probes.json'],
    ['合成编码扰动的 L2 距离及对应温度跳变，用于验证逆映射是否连续。'],
    ['这是合成反例，不能将每个历史训练尖峰归因于该机制。'],
    ['encoded_l2_delta','C_temperature_jump','C_affine_temperature_jump'],[probes['C_inverse_discontinuity_example']]),
 item('validation-baseline','20/39 个最终模型差于同条件均值填补基线',
    ['results_index.json','metrics.json','diagnostics.json'],
    ['Pixel10 在 HLP v2 验证集的 11684 条记录上使用 2 个固定视图，比较原始输出的全有效像素 RMSE，单位 °C。',
     '基线保留观测像素并用相同 oracle mean 填充缺失像素；RMSE 为 0.862860985°C。'],
    ['全部分数来自第 5 轮权重，不是验证最优 checkpoint。','oracle mean 条件限制了对实际部署的外推。'],
    ['model','pixel10_val_rmse_C','worse_than_mean_fill'],rows,
    [{'language':'calculation','code':'sum(pixel10_val_rmse_C > 0.8628609848943773) = 20 of 39'}])]
(OUT / 'sources_receipt.json').write_text(json.dumps({'schemaVersion':1,'items':items},ensure_ascii=False,indent=2), encoding='utf-8')
print('Created FINDINGS.txt, ALL_39_LOSS_SUMMARY.txt, sources_receipt.json')
