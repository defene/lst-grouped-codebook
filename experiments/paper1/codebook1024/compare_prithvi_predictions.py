"""Add frozen continuous Prithvi AE predictions to the existing 16-pair gallery."""
from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.paper1.codebook1024.more_predictions import DEST as PREVIOUS, read, write, region_metrics
from eo_data.core import sha256_file
from eo_denoise.engine import read_checkpoint, seed_everything, device_for, amp_context, source_hashes, memory_stats
from eo_denoise.models import CrossViewModel

DEST = PREVIOUS/'prithvi_comparison'
A02 = ROOT/'runs/paper1/var_vs_prithvi_ae_full'
A03 = ROOT/'runs/paper1/codebook1024/full'
KEYS = [(1024, 'clean'), (1024, 'noisy'), ('prithvi', 'clean'), ('prithvi', 'noisy')]
LABELS = ['原始参考', '实际输入', 'VQ-VAE 1024\n干净训练', 'VQ-VAE 1024\n加噪训练',
          'Prithvi 连续 AE\n干净训练', 'Prithvi 连续 AE\n加噪训练']


@torch.inference_mode()
def infer():
    DEST.mkdir(exist_ok=True)
    if (DEST/'results.json').exists():
        raise FileExistsError('Completed inference already exists. Use render mode to reuse it.')
    start = time.perf_counter()
    original = read(PREVIOUS/'results.json')
    selection = read(PREVIOUS/'selection.json')
    assert original['status']=='complete' and original['source_hashes']==source_hashes()
    torch.set_num_threads(4)
    # All reused artifacts are read-only hard links: no duplicate image arrays on disk.
    reused = ['selection.json']+[f'{m}_{suffix}' for m in ('hls','lst') for suffix in ('inputs.npz','samples.json')]
    reused += [f'1024_{m}_{t}_predictions.npz' for m in ('hls','lst') for t in ('clean','noisy')]
    for name in reused:
        if not (DEST/name).exists():
            os.link(PREVIOUS/name, DEST/name)
        assert sha256_file(PREVIOUS/name)==sha256_file(DEST/name)
    samples = original['samples']
    ids = {s['sample_id'] for s in samples}
    result = dict(status='running', inference_only=True, samples=samples, source_hashes=source_hashes(),
                  rows=[r for r in original['rows'] if r['codebook_size']==1024],
                  models={k:v for k,v in original['models'].items() if k.startswith('1024_')},
                  reused_selection_sha256=sha256_file(PREVIOUS/'selection.json'),
                  no_codebook_in_prithvi=True, no_kl_in_prithvi=True, no_time=True, all_scratch_training=True)
    for m in ('hls','lst'):
        assert sha256_file(DEST/f'{m}_inputs.npz')==selection['input_hashes'][m]
        with np.load(DEST/f'{m}_inputs.npz') as z:
            data = {k:z[k] for k in z.files}
        std, mean = data['std'], data['mean']
        scale = 100 if m=='lst' else 1
        for training in ('clean','noisy'):
            run = A02/f'full_{m}_prithvi_{training}_seed17'
            checkpoint_sha = sha256_file(run/'last.pt')
            checkpoint = read_checkpoint(run/'last.pt')
            config = checkpoint['config']
            assert config==read(run/'config.json')
            assert config['model']['backbone']=='prithvi_eo2_continuous' and config['model']['quantizer']=='ae'
            assert config['loss']['kl']==0 and read(run/'model.json')['pretrained_weights'] is False
            seed_everything(config['seed'])
            device = device_for(config)
            model = CrossViewModel(config).to(device).eval()
            model.load_state_dict(checkpoint['model'],strict=True)
            assert not hasattr(model.codecs[m], 'quantizer')
            del checkpoint
            key = f'prithvi_{m}_{training}'
            saved, differences = {}, []
            for condition in ('clean','noisy10'):
                metric = read(run/f'eval_{condition}/metrics.json')
                comparison = read(A03/f'full_{m}_conv_{training}_seed17/eval_{condition}/metrics.json')
                assert metric['step']==10813 and metric['checkpoint_sha256']==checkpoint_sha
                assert metric['data_fingerprint']==comparison['data_fingerprint']
                outputs=[]
                for begin in range(0,16,8):
                    image=torch.from_numpy(data[condition+'_image'][begin:begin+8]).to(device)
                    mask=torch.from_numpy(data[condition+'_input_mask'][begin:begin+8]).to(device)
                    with amp_context(config,device):
                        predicted=model.codecs[m](image,mask)['reconstruction']
                    outputs.append(predicted.float().cpu().numpy())
                    del image,mask,predicted
                predictions=np.concatenate(outputs)
                assert predictions.shape==data['target'].shape and np.isfinite(predictions).all()
                saved[condition+'_prediction']=predictions
                old={}
                with (run/f'eval_{condition}/samples.jsonl').open(encoding='utf-8') as f:
                    for r in map(json.loads,f):
                        if r['sample_id'] in ids and r['view']==0 and r['region']=='all':
                            old[r['sample_id']]=r
                assert len(old)==16
                for i,sample in enumerate(samples):
                    p=(predictions[i]*std+mean)/scale
                    t=(data['target'][i]*std+mean)/scale
                    ref=data['reference_mask'][i,0].astype(bool)
                    corrupt=data[condition+'_corruption_mask'][i,0].astype(bool)&ref
                    regions={n:region_metrics(p,t,mask) for n,mask in [('all',ref),('corrupt',corrupt),('retained',ref&~corrupt)]}
                    error=((predictions[i]-data['target'][i])*std/scale)[:,ref]
                    rmse=float(np.sqrt(np.mean(error.astype(np.float64)**2)))
                    old_rmse=math.sqrt(old[sample['sample_id']]['mse_physical'])
                    relative=abs(rmse-old_rmse)/max(old_rmse,1e-12)
                    assert math.isclose(rmse,old_rmse,rel_tol=5e-3,abs_tol=1e-5),(key,condition,sample,rmse,old_rmse)
                    differences.append(relative)
                    result['rows'].append(dict(model=key,codebook_size=None,modality=m,training=training,
                        condition=condition,**sample,coverage=float(data[condition+'_actual_coverage'][i]),regions=regions,
                        final_evaluation_rmse=old_rmse,recomputed_normalized_difference_rmse=rmse))
            np.savez_compressed(DEST/f'{key}_predictions.npz',**saved)
            result['models'][key]=dict(checkpoint=str((run/'last.pt').relative_to(ROOT)),checkpoint_sha256=checkpoint_sha,
                predictions_sha256=sha256_file(DEST/f'{key}_predictions.npz'),parameters=read(run/'model.json')['parameters'],
                rate=model.codecs[m].rate(),saved_full_evaluation_max_relative_rmse_difference=max(differences),memory=memory_stats(device))
            print(json.dumps(dict(model=key,complete=True,max_relative_rmse_difference=max(differences),memory=memory_stats(device))),flush=True)
            del model,config,predictions,outputs,saved
            gc.collect();torch.cuda.empty_cache()
            write(DEST/'status.json',dict(status='running',completed_prithvi_models=len(result['models'])-4))
    result.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
    assert len(result['rows'])==256
    write(DEST/'results.json',result)
    write(DEST/'status.json',dict(status='complete',completed_prithvi_models=4,new_predictions=128))


def report():
    results=read(DEST/'results.json')
    full_rows=[]
    for m in ('hls','lst'):
        for training in ('clean','noisy'):
            for condition in ('clean','noisy10'):
                models={}
                for key,root,backbone in [('vq4096',A02,'conv'),('vq1024',A03,'conv'),('prithvi',A02,'prithvi')]:
                    path=root/f'full_{m}_{backbone}_{training}_seed17/eval_{condition}/metrics.json'
                    d=read(path)
                    assert d['evaluated_pairs']==11757
                    models[key]=dict(regions=d['regions'],metric_sha256=sha256_file(path),path=str(path.relative_to(ROOT)))
                full_rows.append(dict(modality=m,training=training,condition=condition,models=models))
    lines=['# Prithvi 连续 AE 与卷积 VQ-VAE：同一批 16 对预测', '',
        '直接复用上一批 16 个瓦片、16 对 HLS-LST 的输入、参考、噪声掩膜与裁剪。新增四个 Prithvi 最终检查点的 128 张预测，不重新训练；与 1024 VQ-VAE 合计 256 张预测。四个 Prithvi 模型也从零训练，使用 EO 2.0 空间 ViT 的连续瓶颈，无时间编码、VQ 或 KL。', '',
        '每个模态的图册有 16 页：列为原图、实际输入、VQ-VAE 1024 干净训练、VQ-VAE 1024 加噪训练、Prithvi 干净训练、Prithvi 加噪训练；行依次为干净输入全图、干净局部、缺失输入全图、缺失局部。', '',
        '**全量结果：Prithvi 的总体重建误差明显更低。但在加噪训练、缺失输入条件下，缺失区的误差与 VQ-VAE 接近，主要优势在保留区。**', '',
        '已检查的放大图中，Prithvi 保留了更多 HLS 边界和 LST 局部温度结构。例如样本12的干净输入对比，VQ-VAE 的色块平滑明显，而 Prithvi 更接近参考。缺失恢复仍有问题：样本14的 LST 缺失区域，Prithvi 加噪模型也出现块状接缝，不能把看起来更丰富的纹理一律当作正确细节。', '',
        '下表来自原全量 11,757 条验证记录，每条两个视图，不是新抽出的 16 张图。HLS 单位为反射率，LST 为 °C。', '',
        '| 模态 | 训练 | 验证输入 | VQ-VAE 4096 RMSE | VQ-VAE 1024 RMSE | Prithvi RMSE |',
        '|---|---|---|---:|---:|---:|']
    for r in full_rows:
        v=[r['models'][k]['regions'][r['modality']+'/all']['rmse'] for k in ('vq4096','vq1024','prithvi')]
        lines.append(f"| {r['modality'].upper()} | {'干净' if r['training']=='clean' else '加噪'} | {'干净' if r['condition']=='clean' else '10% 缺失'} | "+' | '.join(f'{x:.6f}' for x in v)+' |')
    lines+=['', '以下只比较加噪训练模型在 10% 缺失输入上的区域误差：', '',
        '| 模态 | 区域 | VQ-VAE 1024 RMSE | Prithvi RMSE |','|---|---|---:|---:|']
    for r in full_rows:
        if r['training']=='noisy' and r['condition']=='noisy10':
            for region,zh in [('corrupt','缺失区'),('retained','保留区')]:
                values=[r['models'][k]['regions'][r['modality']+'/'+region]['rmse'] for k in ('vq1024','prithvi')]
                lines.append(f"| {r['modality'].upper()} | {zh} | {values[0]:.6f} | {values[1]:.6f} |")
    lines+=['', 'Prithvi 只做干净训练时，遇到缺失输入的缺失区误差很大；其良好的干净重建不等于天然具备缺失恢复能力。不能把干净模型与加噪模型的输入条件混在一起比较。', '',
        '下表仅展示这 16 张图的梯度幅值比中位数。它表示预测空间变化相对参考的强弱，并不是细节正确率；大于 100% 也可能来自伪纹理、伪边缘或数值偏差。', '',
        '| 模态 | 模型 | 训练 | 干净输入 | 缺失输入 |','|---|---|---|---:|---:|']
    for m in ('hls','lst'):
        for key,t in KEYS:
            values=[np.median([r['regions']['all']['gradient_amplitude_ratio'] for r in results['rows'] if r['model']==f'{key}_{m}_{t}' and r['condition']==c]) for c in ('clean','noisy10')]
            lines.append(f"| {m.upper()} | {'Prithvi 连续 AE' if key=='prithvi' else 'VQ-VAE 1024'} | {'干净' if t=='clean' else '加噪'} | {values[0]:.1%} | {values[1]:.1%} |")
    lines+=['', '**比较限制：参数规模接近，但压缩率没有对齐。** Prithvi 保留 257×768 个连续特征（包含 CLS），按 float32 计为 6,316,032 bit；VQ-VAE 1024 是 680 个离散索引、固定载荷 6,800 bit。二者都不包含模型、掩膜或文件头，也不是实际编码文件大小。这项对比不能把优势完全归因于 ViT，或完全归因于是否使用 VQ。', '',
        '三遍训练数据、单种子、只有验证集；尚未证明充分收敛。图像锐利不等于高频细节正确，需要同时查看参考、误差与纹理。', '',
        '所有列共用同一参考图确定的显示范围及裁剪，未锐化或平滑。LST 读取除以100；HLS 显示 RGB，但误差使用六波段。', '',
        '| 样本 | 记录 | 瓦片 | HLS | LST |', '|---:|---|---|---|---|']
    for s in results['samples']:
        n,rid=s['number'],s['record_id']
        lines.append(f"| {n} | {rid} | {s['tile_id']} | [查看](images/hls_{n:02d}_{rid}.png) | [查看](images/lst_{n:02d}_{rid}.png) |")
    lines+=['', '完整全量指标及来源：full_validation_comparison.json；逐图指标、检查点和原评估一致性核对：results.json；原始预测：prithvi_*_predictions.npz。', '',
        '![样本12干净输入局部对比](paired_sample12_clean_detail.png)']
    write(DEST/'full_validation_comparison.json',full_rows)
    (DEST/'README_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def render():
    from experiments.paper1.codebook1024.render_more_predictions import main as plates
    from experiments.paper1.codebook1024.prediction_quicklook import main as quicklook
    plates(destination=DEST,model_keys=KEYS,column_labels=LABELS,write_vq_report=False)
    quicklook(destination=DEST,model_keys=KEYS,column_titles=[LABELS[0]]+LABELS[2:],
              caption='同一参考裁剪：VQ-VAE 1024 对照 Prithvi 连续 AE；参数规模接近，压缩率未对齐')
    report()


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='render':
        render()
    else:
        infer()
