"""Final matched-budget LST report, physical metrics, and fixed-record pictures."""
from pathlib import Path
from datetime import datetime
import argparse
import gc
import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eo_data.core import sha256_file
from eo_denoise import engine
from eo_denoise.data import PairedDataset
from experiments.paper1.lst_residual.runtime import install, ExperimentModel
from experiments.paper1.lst_residual.prepare import ARMS
from experiments.paper1.var_vs_prithvi_ae.run_suite import read, json_write

LABELS = ['原始预测', '原始预测＋空间4倍', '残差预测', '残差预测＋空间4倍']


@torch.inference_mode()
def main(output):
    install()
    torch.set_num_threads(4)
    destination = output/'report'
    destination.mkdir(exist_ok=True)
    reference_root = ROOT/'runs/paper1/codebook1024/full/report/spatial_token_audit'
    selected = read(reference_root/'selection.json')
    reference = np.load(reference_root/'1024_lst_clean_clean.npz')
    lookup = {int(r): float(w) for r, w in zip(reference['record_ids'], reference['physical_target_within'])}
    image_positions = selected['probe_positions'][:8]
    image_ids = [selected['record_ids'][i] for i in image_positions]
    results = dict(status='running', training_steps=2000, records_presented=128000, unique_training_pool=230659,
                   validation_records=256, evaluation_views=2, image_view=0, image_record_ids=image_ids,
                   training='10% known-mask missing', pretrained_weights=False, rows=[], artifacts=[])
    pictures = {}
    events = [json.loads(s) for s in (output/'pilot/stages.jsonl').read_text(encoding='utf-8').splitlines()]
    for index, (arm, _, _) in enumerate(ARMS):
        run = output/'pilot'/f'pilot_lst_{arm}_seed17'
        assert read(run/'status.json')['status'] == 'complete'
        checkpoint = engine.read_checkpoint(run/'last.pt')
        config = checkpoint['config']
        assert checkpoint['step'] == 2000
        engine.seed_everything(config['seed'])
        device = engine.device_for(config)
        model = ExperimentModel(config).to(device).eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        starts = [datetime.fromisoformat(e['utc']) for e in events if e['name'] == run.name and e['stage'] == 'train' and e['event'] == 'start']
        ends = [datetime.fromisoformat(e['utc']) for e in events if e['name'] == run.name and e['stage'] == 'train' and e['event'] == 'end' and e['exit_code'] == 0]
        # Do not label concurrent job wall time as dedicated single-model GPU hours.
        wall = (ends[-1]-starts[-1]).total_seconds()/3600 if len(starts) == len(ends) == 1 else None
        for condition, coverage in [('clean', 0), ('noisy10', .1)]:
            metrics = read(run/f'eval_{condition}/metrics.json')
            assert metrics['checkpoint_sha256'] == sha256_file(run/'last.pt')
            rows = [json.loads(s) for s in (run/f'eval_{condition}/samples.jsonl').read_text(encoding='utf-8').splitlines()]
            rows = [r for r in rows if r['region'] == 'all']
            assert len(rows) == 512
            per_image = {}
            for r in rows:
                rid = int(r['sample_id'].rsplit(':', 1)[1])
                per_image.setdefault(rid, []).append(r)
            assert set(per_image) == set(lookup)
            means = np.array([np.mean([r['bias_physical']**2 for r in rs]) for rs in per_image.values()])
            mses = np.array([np.mean([r['mse_physical'] for r in rs]) for rs in per_image.values()])
            spatial = np.maximum(mses-means, 0)
            stats = dict(model=arm, label=LABELS[index], condition=condition,
                rmse=metrics['regions']['lst/all']['rmse'], gradient_rmse=metrics['regions']['lst/all']['gradient_rmse'],
                mean_rmse=float(np.sqrt(means.mean())), spatial_rmse=float(np.sqrt(spatial.mean())),
                spatial_r2=1-float(spatial.mean())/np.mean(list(lookup.values())),
                regions=metrics['regions'], tokens=metrics['tokens']['lst'], rate=metrics['rates']['lst'],
                shared_gpu_training_wall_hours=wall, checkpoint_sha256=metrics['checkpoint_sha256'])
            results['rows'].append(stats)
            ds = PairedDataset(config, 'val', coverage=coverage)
            positions = {int(r): i for i, r in enumerate(ds.record_ids)}
            entries = [ds[positions[r]] for r in image_ids]
            pred_parts = []
            for begin in range(0, 8, 4):
                data = entries[begin:begin+4]
                image = torch.from_numpy(np.stack([r['lst']['image'][0] for r in data])).to(device)
                mask = torch.from_numpy(np.stack([r['lst']['input_mask'][0] for r in data])).to(device)
                with engine.amp_context(config, device):
                    pred = model.codecs['lst'](image, mask)['reconstruction'].float()
                pred_parts.append(pred.cpu().numpy())
            predictions = np.concatenate(pred_parts)
            std = np.asarray(ds.sources['lst'].stats['std'], np.float32)[:, None, None]
            mean = np.asarray(ds.sources['lst'].stats['mean'], np.float32)[:, None, None]
            targets = np.stack([r['lst']['target'] for r in entries])
            masks = np.stack([r['lst']['reference_mask'][0] for r in entries]).astype(bool)
            for i, rid in enumerate(image_ids):
                old = next(r['mse_physical'] for r in per_image[rid] if r['view'] == 0)
                mse = np.mean((((predictions[i]-targets[i])*std/100)[:, masks[i]]).astype(np.float64)**2)
                assert np.isclose(mse, old, rtol=.002, atol=1e-7)
            pictures[f'{arm}_{condition}'] = (predictions*std+mean)/100
            if index == 0:
                pictures['target'] = (targets*std+mean)/100
                pictures['reference_mask'] = masks
                pictures[condition+'_input'] = (np.stack([r['lst']['image'][0] for r in entries])*std+mean)/100
                pictures[condition+'_mask'] = np.stack([r['lst']['input_mask'][0, 0] for r in entries]).astype(bool)
            ds.close()
        del model, pred, image, mask
        gc.collect()
        torch.cuda.empty_cache()
    np.savez_compressed(destination/'predictions_8_records.npz', record_ids=image_ids, **pictures)
    font = 'C:/Windows/Fonts/msyh.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(), 'axes.unicode_minus': False})
    for condition in ('clean', 'noisy10'):
        for page in range(2):
            fig, axes = plt.subplots(4, 6, figsize=(16, 11), layout='constrained')
            for row, i in enumerate(range(page*4, page*4+4)):
                valid = pictures['reference_mask'][i]
                target = pictures['target'][i, 0]
                lo, hi = np.quantile(target[valid], [.01, .99])
                if hi-lo < .01:
                    lo, hi = lo-.5, hi+.5
                values = [target, pictures[condition+'_input'][i, 0]]+[pictures[f'{a}_{condition}'][i, 0] for a, _, _ in ARMS]
                masks = [valid, pictures[condition+'_mask'][i]]+[valid]*4
                for col, (value, support) in enumerate(zip(values, masks)):
                    cmap = plt.get_cmap('inferno').copy()
                    cmap.set_bad('#777777')
                    im = axes[row, col].imshow(np.ma.masked_where(~support, value), vmin=lo, vmax=hi, cmap=cmap, interpolation='nearest')
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                    if row == 0:
                        axes[row, col].set_title((['真实温度', '实际输入']+LABELS)[col], fontsize=10)
                    if col == 0:
                        axes[row, col].set_ylabel(f'记录 {image_ids[i]}', fontsize=9)
                fig.colorbar(im, ax=axes[row, :], shrink=.8, pad=.008, label='°C')
            fig.suptitle(f'LST 残差预测 · 四组均从零训练 2000 步 · {"干净输入" if condition == "clean" else "约 10% 缺失输入"}', fontsize=14)
            fig.supxlabel('每行共用真实温度 1%–99% 色标；灰色为无效或缺失。样本在训练前由固定哈希选择，未按结果挑图。', fontsize=9)
            path = destination/f'predictions_{condition}_{page+1}.png'
            fig.savefig(path, dpi=160)
            plt.close(fig)
            results['artifacts'].append(dict(path=path.name, sha256=sha256_file(path)))
    results['status'] = 'complete'
    json_write(destination/'results.json', results)
    lines = ['# LST 残差预测短程对照', '',
        '四组均从随机初始化训练 2,000 步，使用相同种子、样本顺序、10% 已知掩码缺失输入、1024 码字和 16×16 最细网格。有效 batch 64，每组 128,000 次样本呈现；训练池 230,659 条，未完成一遍全量遍历。结果用于初步筛选，不代表充分收敛。HLS 未参与。', '',
        '残差组只用可见输入计算逐图温度均值；编码器输入减去该均值，解码器预测残差，最后加回该均值。既不访问缺失区的真实温度，也不访问干净参考均值。该均值作为 32 bit 辅助量计入码率：基线 6800 bit/图，残差组 6832 bit/图，不含共同掩码、文件头和模型。', '',
        '空间4倍表示在原有 MSE 上额外加入 3 倍逐图去均值空间 MSE；VQ 损失系数与 0.1 梯度损失系数保持不变。四组构成 2×2 对照，分别检查预测方式和损失权重。', '',
        '| 模型 | 评估输入 | RMSE °C | 均值 RMSE °C | 空间 RMSE °C | 空间 R² | 梯度 RMSE °C | 码字困惑度 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results['rows']:
        lines.append(f"| {r['label']} | {'干净' if r['condition']=='clean' else '缺失'} | {r['rmse']:.4f} | {r['mean_rmse']:.4f} | {r['spatial_rmse']:.4f} | {r['spatial_r2']:.1%} | {r['gradient_rmse']:.4f} | {r['tokens']['perplexity']:.2f} |")
    lines += ['', '空间 R² 的参照是同一批 256 条验证数据的图内方差；去均值仍包含大尺度温度渐变，不能当作纯高频细节指标。两种噪声实现先在每个样本内平均，再对样本等权。负 R² 表示空间误差超过该方差基准。所有比较以相同步数的新基线为准，不把之前 10,813 步的模型当作等预算对照。', '',
        '## 用时', '', '| 模型 | 训练阶段墙钟小时（与另一作业共享 GPU） |', '|---|---:|']
    for r in results['rows'][::2]:
        lines.append(f"| {r['label']} | {r['shared_gpu_training_wall_hours']:.3f} |" if r['shared_gpu_training_wall_hours'] is not None else f"| {r['label']} | 存在恢复，待核算 |")
    lines += ['', '两组并行共享一张 RTX 5090；以上是每个作业训练阶段经过的时间，含定期验证和检查点 I/O，不能直接相加后声称是实际消耗的独占 GPU-hours，也不能当作单模型独占速度。阶段起止记录在 pilot/stages.jsonl，显存在 gpu.jsonl。', '',
        '## 配对图像', '']
    for a in results['artifacts']:
        lines += [f"![{a['path']}]({a['path']})", '']
    (destination/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    json_write(destination/'numerical_review.json', dict(status='passed', checkpoints_and_steps_verified=True,
        plot_predictions_match_evaluation=True, same_validation_membership=True, visual_review_pending=True,
        results_sha256=sha256_file(destination/'results.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
