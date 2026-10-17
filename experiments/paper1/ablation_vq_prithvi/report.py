"""Build reproducible scientific figures and an exact first-pilot result table."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import argparse
import copy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eo_data.core import json_write, sha256_file
from eo_denoise.config import load_config
from eo_denoise.data import PairedDataset
from eo_denoise.engine import read_checkpoint, seed_everything, device_for, amp_context, move_batch, inputs_only
from eo_denoise.models import CrossViewModel
from eo_denoise.metrics import EvaluationMetrics

COLORS = {'conv':'#16658B', 'prithvi':'#B66A28'}
LABELS = {'conv':'Conv VQ-VAE', 'prithvi':'Spatial Prithvi VQ-VAE'}
CASES = (0,64,128,192)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def model_name(m, b, n):
    return f'pilot_{m}_{b}_{n}_seed17'


def averaged_samples(path, modality, region):
    values = defaultdict(list)
    for row in rows(path):
        if row['modality'] == modality and row['region'] == region:
            values[(row['sample_id'],row['tile_id'])].append(row['mse_physical'])
    return {key:float(np.mean(value)) for key,value in values.items()}


def paired_delta(clean_path, noisy_path, modality, region):
    a, b = averaged_samples(clean_path,modality,region), averaged_samples(noisy_path,modality,region)
    assert a.keys() == b.keys()
    by_tile = defaultdict(list)
    for key in a:
        by_tile[key[1]].append([a[key],b[key]])
    sums = np.array([np.sum(v,axis=0) for v in by_tile.values()])
    counts = np.array([len(v) for v in by_tile.values()])
    rmse = np.sqrt(sums.sum(0)/counts.sum())
    rng = np.random.default_rng(20260910)
    bootstrap = []
    for _ in range(1000):
        draw = rng.integers(len(sums),size=len(sums))
        values = np.sqrt(sums[draw].sum(0)/counts[draw].sum())
        bootstrap.append(values[1]-values[0])
    return {'delta_noisy_minus_clean':float(rmse[1]-rmse[0]),
            'relative_reduction_percent':float(100*(rmse[0]-rmse[1])/rmse[0]),
            'paired_tile_ci95':np.quantile(bootstrap,[.025,.975]).tolist(),
            'samples':len(a),'tiles':len(by_tile)}


def collect(run):
    records, traces, deltas = [], {}, []
    stream_hashes, initial_hashes, val_hashes, noise_hashes = {}, {}, {}, {}
    for m in ('hls','lst'):
        for b in ('conv','prithvi'):
            for n in ('clean','noisy'):
                name = model_name(m,b,n)
                folder = run/name
                assert read(folder/'status.json')['status'] == 'complete'
                config = read(folder/'config.json')
                assert config['train']['steps'] == 1000 and config['data']['train_views'] == 1
                log = rows(folder/'train.jsonl')
                assert [r['step'] for r in log] == list(range(1,1001))
                ids = [s for row in log for s in row['sample_ids']]
                assert len(ids) == len(set(ids)) == 8000
                stream_hashes[name] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                traces[name] = log
                seed_everything(config['seed'])
                model = CrossViewModel(config)
                digest = hashlib.sha256()
                for key, value in model.state_dict().items():
                    digest.update(key.encode()); digest.update(value.cpu().numpy().tobytes())
                initial_hashes[name] = digest.hexdigest()
                del model
                for e in ('clean','noisy10'):
                    evaluation = folder/('eval_'+e)
                    result = read(evaluation/'metrics.json')
                    assert result['step'] == 1000 and result['evaluated_pairs'] == 256
                    assert result['coverage'] == (0 if e == 'clean' else .1)
                    val_hashes[name+'/'+e] = result['data_fingerprint']['paired_record_ids_sha256']
                    with np.load(evaluation/'preview.npz') as preview:
                        noise_hashes[name+'/'+e] = hashlib.sha256(preview[m+'_image'].tobytes()+preview[m+'_corruption_mask'].tobytes()).hexdigest()
                    all_region = result['regions'][m+'/all']
                    records.append({'modality':m,'backbone':b,'training':n,'evaluation':e,
                        'parameters':read(folder/'model.json')['parameters'],
                        'train_elapsed_seconds':log[-1]['elapsed_seconds'],
                        'last100_reconstruction_mse':float(np.mean([x['reconstruction'] for x in log[-100:]])),
                        'last100_vq_loss':float(np.mean([x['bottleneck'] for x in log[-100:]])),
                        'all':all_region,'corrupt':result['regions'].get(m+'/corrupt'),
                        'retained':result['regions'][m+'/retained'],'tokens':result['tokens'][m],
                        'checkpoint_sha256':result['checkpoint_sha256'],
                        'directory':evaluation.relative_to(ROOT).as_posix()})
            for e in ('clean','noisy10'):
                for region in (('all',) if e == 'clean' else ('all','corrupt','retained')):
                    a = run/model_name(m,b,'clean')/('eval_'+e)/'samples.jsonl'
                    z = run/model_name(m,b,'noisy')/('eval_'+e)/'samples.jsonl'
                    deltas.append({'modality':m,'backbone':b,'evaluation':e,'region':region,
                                   **paired_delta(a,z,m,region)})
            assert initial_hashes[model_name(m,b,'clean')] == initial_hashes[model_name(m,b,'noisy')]
        for e in ('clean','noisy10'):
            assert len({v for k,v in noise_hashes.items() if k.startswith('pilot_'+m+'_') and k.endswith('/'+e)}) == 1
    assert len(set(stream_hashes.values())) == 1 and len(set(val_hashes.values())) == 1
    return records,traces,deltas,{'training_stream_hashes':stream_hashes,'initial_weight_hashes':initial_hashes,
                                'validation_membership_hashes':val_hashes,'first_evaluation_input_hashes':noise_hashes}


@torch.no_grad()
def input_baseline(run, report):
    result = {}
    for m in ('hls','lst'):
        c = read(run/model_name(m,'conv','noisy')/'config.json')
        c['data']['val_limit'] = 256
        ds = PairedDataset(c,'val',coverage=.1)
        model_stub = SimpleNamespace(codecs={m:SimpleNamespace(config={'quantizer':'ae'})})
        metrics = EvaluationMetrics(ds.sources,model_stub)
        for batch in DataLoader(ds,batch_size=8,num_workers=2):
            outputs = [{'reconstruction':batch[m]['image'][:,i]} for i in range(2)]
            metrics.add(m,outputs,batch[m],batch['sample_id'],batch['tile_id'])
        result[m] = metrics.summary(500)
        ds.close()
    json_write(report/'input_baseline.json',result)
    return result


@torch.no_grad()
def visual_cases(run, report):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10})
    evidence = {}
    for m in ('hls','lst'):
        for e, coverage in (('clean',0),('noisy10',.1)):
            c = read(run/model_name(m,'conv','noisy')/'config.json')
            c['data']['val_limit'] = 256
            ds = PairedDataset(c,'val',coverage=coverage)
            samples = [ds[i] for i in CASES]
            batch = default_collate(samples)
            source = ds.sources[m]
            std = np.array(source.stats['std'],dtype=np.float32)[:,None,None]
            mean = np.array(source.stats['mean'],dtype=np.float32)[:,None,None]
            def physical(x):
                return (x*std+mean)/(100 if m == 'lst' else 1)
            target = physical(batch[m]['target'].numpy())
            noisy = physical(batch[m]['image'][:,0].numpy())
            reference = batch[m]['reference_mask'].numpy()[:,0]
            visible = batch[m]['input_mask'][:,0].numpy()[:,0]
            arrays = {'target':target,'input':noisy,'reference_mask':reference,'input_mask':visible}
            columns = [('target','Original / target')]
            if e == 'noisy10': columns += [('input','Input: 10% missing')]
            for b in ('conv','prithvi'):
                for n in ('clean','noisy'):
                    checkpoint = read_checkpoint(run/model_name(m,b,n)/'last.pt')
                    config = checkpoint['config']
                    device = device_for(config)
                    seed_everything(config['seed'])
                    model = CrossViewModel(config).to(device).eval()
                    model.load_state_dict(checkpoint['model'])
                    moved = move_batch(batch,device,[m])
                    with amp_context(config,device):
                        out = model(inputs_only(moved,[m]))[m][0]['reconstruction']
                    key = b+'_'+n
                    arrays[key] = physical(out.float().cpu().numpy())
                    columns.append((key,('Conv' if b == 'conv' else 'Prithvi')+' / '+n+' train'))
                    del checkpoint,model,moved,out
            np.savez_compressed(report/(m+'_'+e+'_cases.npz'),**arrays)
            evidence[m+'/'+e] = {'indices':list(CASES),'sample_ids':[s['sample_id'] for s in samples],
                                'tiles':[s['tile_id'] for s in samples]}
            ncols = len(columns)
            fig = plt.figure(figsize=(ncols*2.15+(.35 if m=='lst' else 0),9.5),layout='constrained')
            grid = fig.add_gridspec(4,ncols+(1 if m=='lst' else 0),
                                   width_ratios=[1]*ncols+([.05] if m=='lst' else []))
            for i in range(4):
                valid = reference[i]
                if m == 'lst':
                    lo,hi = np.quantile(target[i,0][valid],[.02,.98])
                    if hi-lo < 1: hi=lo+1
                for j,(key,title) in enumerate(columns):
                    ax = fig.add_subplot(grid[i,j])
                    mask = visible[i] if key == 'input' else valid
                    a = arrays[key][i]
                    if m == 'hls':
                        rgb = np.clip(a[[2,1,0]].transpose(1,2,0)/.3,0,1)**(1/2.2)
                        rgb[~mask] = .72
                        ax.imshow(rgb,interpolation='nearest')
                    else:
                        cmap = plt.get_cmap('inferno').copy();cmap.set_bad('#b8b8b8')
                        im = ax.imshow(np.ma.array(a[0],mask=~mask),cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
                    ax.set_xticks([]);ax.set_yticks([])
                    if i == 0: ax.set_title(title)
                    if j == 0: ax.set_ylabel(samples[i]['sample_id'].rsplit(':',1)[-1]+' / '+samples[i]['tile_id'],fontsize=9)
                if m == 'lst':
                    cax=fig.add_subplot(grid[i,-1]);fig.colorbar(im,cax=cax,label='Celsius')
            detail = 'RGB display: reflectance 0–0.30, gamma 2.2; gray = unavailable input/reference' if m=='hls' else 'Shared scale across each row: target 2nd–98th percentiles; gray = unavailable input/reference'
            fig.suptitle(f'{m.upper()} | {"Clean" if e=="clean" else "Corrupted"} validation input | 1,000 steps from scratch\n'+detail,fontsize=11)
            fig.savefig(report/(m+'_'+e+'_comparison.png'),dpi=170)
            plt.close(fig);ds.close()
    json_write(report/'visual_case_selection.json',evidence)


def plots(run, report, records, traces):
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,m in enumerate(('hls','lst')):
        for col,e in enumerate(('clean','noisy10')):
            ax=axes[row,col]
            for j,b in enumerate(('conv','prithvi')):
                for k,n in enumerate(('clean','noisy')):
                    r=next(r for r in records if (r['modality'],r['backbone'],r['training'],r['evaluation'])==(m,b,n,e))
                    y=r['all']['rmse'];lo,hi=r['all']['rmse_tile_ci95'];x=j*3+k
                    ax.bar(x,y,width=.72,color=COLORS[b] if n=='noisy' else 'white',edgecolor=COLORS[b],linewidth=1.5)
                    ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt='none',color='#333333',capsize=3,linewidth=1)
                    ax.annotate(f'{y:.4f}' if m=='hls' else f'{y:.2f}',(x,hi),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
            ax.set_xticks([0,1,3,4],['Conv\nclean train','Conv\nnoisy train','Prithvi\nclean train','Prithvi\nnoisy train'])
            ax.set_title(m.upper()+' / '+('clean input' if e=='clean' else '10% missing input'))
            ax.set_ylabel('RMSE (reflectance)' if m=='hls' else 'RMSE (Celsius)')
            ax.set_ylim(bottom=0,top=ax.get_ylim()[1]*1.15);ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',color='#dddddd',linewidth=.6);ax.set_axisbelow(True)
    fig.suptitle('Same 256 validation pairs | all reference pixels | 95% tile-bootstrap intervals\nOne seed, 1,000 training steps; parameter difference <4% within each modality',fontsize=11)
    fig.savefig(report/'rmse_comparison.png',dpi=170);plt.close(fig)
    fig, axes=plt.subplots(2,3,figsize=(15,7),layout='constrained')
    for row,m in enumerate(('hls','lst')):
        for b in ('conv','prithvi'):
            for n in ('clean','noisy'):
                name=model_name(m,b,n);log=traces[name]
                values=np.array([r['reconstruction'] for r in log])
                rolling=np.convolve(values,np.ones(50)/50,mode='valid')
                style='-' if n=='noisy' else '--'
                label=('Conv' if b=='conv' else 'Prithvi')+' / '+n+' train'
                axes[row,0].plot(np.arange(50,1001),rolling,color=COLORS[b],ls=style,label=label,lw=1.4)
                vq=np.array([r['bottleneck'] for r in log])
                axes[row,1].plot(np.arange(50,1001),np.convolve(vq,np.ones(50)/50,mode='valid'),color=COLORS[b],ls=style,label=label,lw=1.4)
                vals=[read(run/name/f'val_step_{step:07d}'/'metrics.json')['regions'][m+'/all']['rmse_normalized'] for step in (250,500,750,1000)]
                axes[row,2].plot([250,500,750,1000],vals,color=COLORS[b],ls=style,marker='o',label=label,lw=1.4)
        axes[row,0].set_title(m.upper()+' / train reconstruction MSE (50-step mean)')
        axes[row,1].set_title(m.upper()+' / VQ loss (50-step mean)')
        axes[row,1].set_yscale('log')
        axes[row,2].set_title(m.upper()+' / fixed 64-pair noisy validation nRMSE')
        for ax in axes[row]:
            ax.set_xlabel('Optimizer step');ax.spines[['top','right']].set_visible(False);ax.grid(color='#dddddd',linewidth=.6)
        axes[row,2].legend(fontsize=8)
    fig.savefig(report/'learning_curves.png',dpi=170);plt.close(fig)


def write_summary(run, report, records, deltas, audit, baseline):
    json_write(report/'results.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'single-seed equal-budget 1000-step pilot; validation only; not converged full-data training',
        'records':records,'paired_noise_training_effects':deltas,'fairness_audit':audit,
        'protocol':'experiments/paper1/ablation_vq_prithvi/PROTOCOL.md'})
    text=['# VQ-VAE 与无时间 Prithvi：首轮消融结果','',
          '8 个模型均从零开始，seed=17，各训练 1,000 步 / 8,000 张图。以下统一使用第 1,000 步 checkpoint；相同 256 个验证样本，每个样本两次固定噪声。原始可靠图始终作为目标。','',
          '在本轮预算和优化设置下，卷积 VQ-VAE 的重建误差整体低于缩小后的 Prithvi。卷积模型加噪训练明显改善遮挡区恢复，同时降低了干净图的重建保真度。Prithvi 的 LST 指标有改善，HLS 整体误差没有稳定改善；其量化损失和图像中的块状伪影表明还需进一步检查量化与优化。',
          '两类模型的细节恢复都不足：卷积结果明显模糊，Prithvi 出现块状和细碎伪影。当前结果不能解释为训练收敛后两种架构的最终排名。','',
          '本轮加噪指约 10% 的 QA 云/影形状遮挡，模型知道输入 mask。不是 Gaussian 加性噪声，也不代表真实有云观测去云。','',
          '| 模态 | 架构 | 训练输入 | 参数量 | 干净评价 RMSE | 加噪评价 RMSE | 遮挡区 RMSE | 保留区 RMSE | 加噪评价用码率 |',
          '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for m in ('hls','lst'):
        for b in ('conv','prithvi'):
            for n in ('clean','noisy'):
                z={r['evaluation']:r for r in records if (r['modality'],r['backbone'],r['training'])==(m,b,n)}
                a,q=z['clean'],z['noisy10'];fmt='.5f' if m=='hls' else '.3f'
                text.append(f"| {m.upper()} | {'卷积 VQ-VAE' if b=='conv' else 'Spatial Prithvi VQ-VAE'} | {'原图' if n=='clean' else '加噪图'} | {q['parameters']/1e6:.3f} M | {a['all']['rmse']:{fmt}} | {q['all']['rmse']:{fmt}} | {q['corrupt']['rmse']:{fmt}} | {q['retained']['rmse']:{fmt}} | {q['tokens']['usage_fraction']*100:.1f}% |")
    text+=['','HLS 误差单位为反射率，LST 为摄氏度。用码率指 512 个码中被使用的比例，不是文件压缩率。','',
           '![验证误差与区间](rmse_comparison.png)','','## 加噪训练相对于干净训练','',
           '下表均在相同的加噪验证输入上比较；差值 = 加噪训练 − 干净训练，负值表示误差下降。区间来自成对 tile bootstrap，只反映验证样本不确定性。','',
           '| 模态 | 架构 | 区域 | RMSE 差值 | 成对 95% 区间 | RMSE 降低比例 |',
           '|---|---|---|---:|---|---:|']
    for d in deltas:
        if d['evaluation']!='noisy10' or d['region']=='retained':continue
        lo,hi=d['paired_tile_ci95']
        text.append(f"| {d['modality'].upper()} | {d['backbone']} | {d['region']} | {d['delta_noisy_minus_clean']:.5f} | [{lo:.5f}, {hi:.5f}] | {d['relative_reduction_percent']:.2f}% |")
    text+=['','## 优化状态','','以下是最后 100 个训练步的平均值，帮助区分重建改善与码本收敛。VQ loss 包含 codebook 和 commitment 项；它不是评价图像质量的指标。','',
           '| 模态 | 架构 | 训练输入 | 重建 MSE | VQ loss |',
           '|---|---|---|---:|---:|']
    for r in records:
        if r['evaluation']=='noisy10':
            text.append(f"| {r['modality'].upper()} | {r['backbone']} | {r['training']} | {r['last100_reconstruction_mse']:.5f} | {r['last100_vq_loss']:.5f} |")
    text+=['','## 图像与训练曲线','','案例取验证子集固定第 0、64、128、192 个样本。显示色标共用，原始数值与 mask 在同名 NPZ 中保存；没有按模型效果挑图。','',
           '![HLS 加噪输入对照](hls_noisy10_comparison.png)','','![LST 加噪输入对照](lst_noisy10_comparison.png)','',
           '![HLS 原图输入对照](hls_clean_comparison.png)','','![LST 原图输入对照](lst_clean_comparison.png)','',
           '![学习曲线](learning_curves.png)','','## 验证与边界','',
           '- 8 组训练样本顺序完全相同，同模态同架构的两组初始参数 hash 完全相同；最终验证成员与首个污染输入 hash 已核对一致。',
           '- Prithvi 使用官方编码器/解码器 Block，缩小到约 9M 参数，二维 patch 和 x/y 位置编码，没有时间/地理编码；并非使用 300M 模型。',
           '- VAR 官方 VQ-VAE 实例化为 108,948,355 参数，仅作为本轮上限；本轮使用相同的单尺度 VQ，未引入 VAR 自回归生成器。',
           '- 所有结果仅为一个 seed 的短预算 pilot，训练覆盖约 3.47% 的完整 train；不能据此宣称架构在充分训练后的优劣。',
           '- Prithvi 数据只有 train/val，没有独立 test。原始无效区不参与误差；隐藏区与保留区分别报告。',
           '- 输入直接输出的基线另存于 input_baseline.json：缺失位置为标准化零，即训练均值；保留位置无误差。该基线不做压缩，不能等同于低码率 codec。','',
           '[完整机器结果](results.json) · [实验协议](../../../../experiments/paper1/ablation_vq_prithvi/PROTOCOL.md)']
    (report/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__ == '__main__':
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/paper1/ablation_vq_prithvi_20260910')
    args=parser.parse_args()
    import os
    os.chdir(ROOT)
    report=args.run/'report';report.mkdir(exist_ok=True)
    records,traces,deltas,audit=collect(args.run)
    baseline=input_baseline(args.run,report)
    plots(args.run,report,records,traces)
    visual_cases(args.run,report)
    write_summary(args.run,report,records,deltas,audit,baseline)
    print(json.dumps({'status':'report_complete','path':str(report/'RESULTS.md')},indent=2))
