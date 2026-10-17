"""Record local model verification and build matched-prediction figures."""
import json
from pathlib import Path
from pilot_report import build

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]/'output/efdiff_hlc_pilot_20260915'
local = json.loads((OUT/'local_test/metrics.json').read_text())
assert local['status']=='passed' and set(local['metrics'])=={'x0','epsilon'}
assert local['local_cloud_input_examples_matched']==16
assert all(m['records']==16 and m['all_predictions_finite'] for m in local['metrics'].values())
build(OUT/'local_test',local=True)
cloud = json.loads((OUT/'comparison.json').read_text())['metrics']
lines = ['# HLC pilot：训练完成、权重下载与本地测试', '',
         '两模型均完成 5 个子集 epoch、1,280 次更新、81,920 次月份样本呈现。以下统一使用预先约定的第 5 轮 EMA 权重。', '',
         '## 下载与本地运行验收', '',
         '- 两份去噪器 EMA 权重和共用 Prithvi 编码器已下载至 [models](models/)，总计约 3.3 GB；[SHA256 清单](models/manifest.json)与云端导出一致。',
         '- 本地重新读取 HLC 的 16 条固定验证样本，完成编码器及扩散采样，两模型均输出有限数值。参考温度、掩码与记录身份完全一致；bicubic 输入最大跨平台差异为 0.0000153°C。',
         '- CPU float32 运行冻结编码器，RTX 5090 单样本 bf16 推理；每模型 GPU 峰值约 1.55 GB。既有训练进程保留。',
         '- 这属于验证样本上的本地推理检查，不是独立 test-set 评估。样本数、随机种子、batch 和编码器精度不同于云端512条评估，不能直接用分数变化判断模型改变。', '',
         '## 云端512条验证：主结果', '',
         '| 模型 | RMSE（°C） | MAE（°C） |', '|---|---:|---:|']
for v in ('x0','epsilon'):
    m = cloud[v]
    lines.append(f"| EFDiff-{v} | {m['rmse_c']:.4f} | {m['mae_c']:.4f} |")
lines += [f"| bicubic | {cloud['x0']['bicubic_rmse_c']:.4f} | 未单独汇总 |", '',
          '本轮短训练中，两模型均未超过 bicubic 的 RMSE 基线；不能将本轮结果表述为全量或已收敛复现。', '',
          '## 本地16条验证：功能检查', '', '| 模型 | RMSE（°C） | MAE（°C） |', '|---|---:|---:|']
for v in ('x0','epsilon'):
    m = local['metrics'][v]
    lines.append(f"| EFDiff-{v} | {m['rmse_c']:.4f} | {m['mae_c']:.4f} |")
m = local['metrics']['x0']
lines += [f"| bicubic | {m['bicubic_rmse_c']:.4f} | {m['bicubic_mae_c']:.4f} |", '',
          '[本地完整指标](local_test/metrics.json) · [本地预测图与记录](local_test/REPORT.md) · [云端主结果](REPORT.md)', '',
          '![本地同样本预测对照](local_test/comparison_1.png)', '',
          '原协调器在重新进入已完成的smoke时断言失败；两模型训练文件完整。报告已经重建，云端状态恢复为complete，原错误记录保留在pipeline_before_recovery_20260916.json。完整续训检查点仍留在云端，下载的推理包不含优化器状态。']
(OUT/'LOCAL_TEST.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
manifest_path = HERE/'pilot_run_manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest.update(local_inference_status='passed on 16 matched validation examples per model',
                local_report='output/efdiff_hlc_pilot_20260915/LOCAL_TEST.md',
                local_metrics=local['metrics'])
manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('Local report and run manifest completed')
