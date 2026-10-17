"""Audited physical-unit comparisons and fixed galleries; never invent visual review."""
import argparse
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from eo_data.core import sha256_file,json_write
from eo_denoise import engine
from .prepare import HERE,ROOT,OUTPUT,ARMS,read
from .data import DatasetV2,COUNTS
from . import runtime

NAMES={'linear_gray':'线性灰度','asinh_gray':'非线性灰度','asinh_rgb':'非线性 RGB'}
GALLERY_IDS=[231773,232129,232414,232565,232620,233073,233356,234070]


def load_and_audit(folder,arm,split,condition):
    p=OUTPUT/folder/arm/f'final_{split}_{condition}'
    m=read(p/'metrics.json')
    assert m['evaluated_pairs']==COUNTS[split] and m['realizations']==2
    assert m['samples_sha256']==sha256_file(p/'samples.jsonl')
    ds=DatasetV2(read(HERE/'configs/full'/f'{arm}.json'),split)
    np.testing.assert_array_equal(np.load(p/'record_ids.npy'),ds.record_ids)
    assert ds.fingerprint==m['data_fingerprint']
    ds.close()
    rows=[json.loads(s) for s in (p/'samples.jsonl').read_text().splitlines()]
    allrows=[r for r in rows if r['region']=='all']
    assert len(allrows)==2*COUNTS[split]
    assert len({(r['sample_id'],r['view']) for r in allrows})==len(allrows)
    assert abs(math.sqrt(np.mean([r['mse_physical'] for r in allrows]))-m['regions']['lst/all']['rmse'])<1e-9
    c=read(HERE/'configs/full'/f'{arm}.json')
    from eo_denoise.config import fingerprint
    assert m['config_sha256']==fingerprint(c)
    weight=Path(c['pretrained']['directory'])/'diffusion_pytorch_model.safetensors' if folder=='pretrained' else OUTPUT/folder/arm/'last.pt'
    assert m['checkpoint_sha256']==sha256_file(weight)
    return m,allrows


def paired_interval(a,b):
    # Paired sample means, cluster resampling over tiles; change B minus A in RMSE.
    def grouped(rows):
        d={}
        for r in rows:
            key=(r['sample_id'],r['tile_id'])
            d.setdefault(key,[]).append(r['mse_physical'])
        return {k:np.mean(v) for k,v in d.items()}
    a,b=grouped(a),grouped(b)
    assert a.keys()==b.keys()
    tiles={}
    for key in a:tiles.setdefault(key[1],[]).append([a[key],b[key]])
    sums=np.array([np.sum(v,axis=0) for v in tiles.values()])
    counts=np.array([len(v) for v in tiles.values()])
    rng=np.random.default_rng(20260921);d=[]
    for _ in range(2000):
        ix=rng.integers(len(counts),size=len(counts))
        rms=np.sqrt(sums[ix].sum(0)/counts[ix].sum())
        d.append(rms[1]-rms[0])
    return dict(delta_rmse_celsius=float(np.sqrt(np.mean(list(b.values())))-np.sqrt(np.mean(list(a.values())))),
        paired_tile_ci95=np.quantile(d,[.025,.975]).tolist(),tiles=len(tiles),replicates=2000)


def gallery(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=OUTPUT/'gallery';out.mkdir(exist_ok=True)
    c=read(HERE/'configs/full/linear_gray.json')
    device=engine.device_for(c);engine.seed_everything(17);torch.set_num_threads(4)
    arrays={}
    for condition,coverage in [('clean',0.),('noisy10',.1)]:
        ds=DatasetV2(c,'test',coverage)
        positions=[int(np.flatnonzero(ds.record_ids==i)[0]) for i in GALLERY_IDS]
        loader=DataLoader(Subset(ds,positions),batch_size=2,shuffle=False,num_workers=0)
        batches=list(loader)
        scale=ds.sources['lst'].stats['std'][0]/100
        mean=ds.sources['lst'].stats['mean'][0]/100
        arrays[condition]={
            'target':np.concatenate([b['lst']['target'].numpy()[:,0]*scale+mean for b in batches]),
            'input':np.concatenate([b['lst']['image'].numpy()[:,0,0]*scale+mean for b in batches]),
            'mask':np.concatenate([b['lst']['input_mask'].numpy()[:,0,0] for b in batches]),
            'reference':np.concatenate([b['lst']['reference_mask'].numpy()[:,0] for b in batches])}
        for arm in ARMS:
            cc=read(HERE/'configs/full'/f'{arm}.json')
            model=runtime.ExperimentModel(cc).to(device).eval()
            if folder=='full':
                state=torch.load(OUTPUT/'full'/arm/'last.pt',map_location='cpu',weights_only=True)
                model.load_state_dict(state['model']);del state
            predictions=[]
            with torch.inference_mode():
                for batch in batches:
                    moved=engine.move_batch(batch,device,model.codecs)
                    # Same batch size and float32 path as full evaluations, view 0.
                    pred=model(engine.inputs_only(moved,model.codecs))['lst'][0]['reconstruction']
                    predictions.extend((pred[:,0].cpu().numpy()*scale+mean))
            arrays[condition][arm]=np.array(predictions)
            rows=[json.loads(s) for s in (OUTPUT/folder/arm/f'final_test_{condition}/samples.jsonl').read_text().splitlines()]
            lookup={int(r['sample_id'].rsplit(':',1)[1]):r['mse_physical'] for r in rows if r['region']=='all' and r['view']==0}
            for i,rid in enumerate(GALLERY_IDS):
                valid=arrays[condition]['reference'][i].astype(bool)
                mse=np.mean((arrays[condition][arm][i][valid]-arrays[condition]['target'][i][valid])**2)
                assert abs(mse-lookup[rid])<.003,('Gallery disagrees with full evaluation',rid,arm,mse,lookup[rid])
            del model;gc.collect();torch.cuda.empty_cache()
        ds.close()
    for condition,a in arrays.items():
        for part in range(2):
            fig,axes=plt.subplots(4,8,figsize=(24,12),layout='constrained')
            for row,i in enumerate(range(part*4,(part+1)*4)):
                valid=a['reference'][i].astype(bool)
                target=a['target'][i]
                lo,hi=np.quantile(target[valid],[.01,.99]);hi=max(lo+.1,hi)
                errors=[a[arm][i]-target for arm in ARMS]
                lim=max(.1,max(float(np.quantile(np.abs(e[valid]),.99)) for e in errors))
                panels=[target,np.where(a['mask'][i],a['input'][i],np.nan)]+[a[arm][i] for arm in ARMS]+errors
                titles=['Reference C','Visible input C','Linear gray','Asinh gray','Asinh RGB','Linear error C','Asinh gray error C','RGB error C']
                for col,(z,title) in enumerate(zip(panels,titles)):
                    ax=axes[row,col]
                    z=np.where(valid,z,np.nan)
                    im=ax.imshow(z,cmap='coolwarm' if col>=5 else 'inferno',vmin=-lim if col>=5 else lo,vmax=lim if col>=5 else hi)
                    ax.set_title(title,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
                    if col==0:ax.set_ylabel(str(GALLERY_IDS[i]))
                    fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
            fig.suptitle(f'{folder} | {condition} | fixed view 0 | common row scales',fontsize=14)
            fig.savefig(out/f'{folder}_{condition}_{part+1}.png',dpi=130);plt.close(fig)
    np.savez_compressed(out/f'{folder}_predictions.npz',record_ids=np.array(GALLERY_IDS),
        **{f'{k}_{v}':arr for k,a in arrays.items() for v,arr in a.items()})


def costs():
    rows=[]
    intervals=[]
    for file in OUTPUT.glob('events_*.jsonl'):
        starts={}
        for line in file.read_text().splitlines():
            r=json.loads(line)
            key=(r['pid'],r['stage'])
            if r['event']=='start':starts[key]=datetime.fromisoformat(r['utc']).timestamp()
            if r['event']=='end':
                rows.append(dict(r,event_file=file.name))
                if key in starts and file.name.startswith('events_full_') and r['stage']=='train':
                    intervals.append((starts[key],datetime.fromisoformat(r['utc']).timestamp()))
    by_arm={}
    for arm in ARMS:
        rr=[r for r in rows if r['event_file']==f'events_full_{arm}.jsonl']
        by_arm[arm]=dict(train_stage_shared_gpu_job_hours=sum(r['elapsed_seconds'] for r in rr if r['stage']=='train')/3600,
            final_evaluation_seconds=sum(r['elapsed_seconds'] for r in rr if r['stage']=='evaluate'))
    gpu=[json.loads(s) for s in (OUTPUT/'gpu.jsonl').read_text().splitlines()]
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
        else:merged.append([start,end])
    return dict(per_arm=by_arm,peak_global_used_gib=max(r['used_gib'] for r in gpu),
        physical_single_gpu_training_stage_union_hours=sum(b-a for a,b in merged)/3600,
        note='Three jobs share one GPU. Per-job wall hours include validation/checkpoints/contention; summing them is not physical device GPU-hours or pure compute time.')


def main(partial=False):
    folders=['pretrained'] if partial else ['pretrained','full']
    metrics={};rows={};units=[]
    for folder in folders:
        for arm in ARMS:
            for split in ('val','test'):
                for condition in ('clean','noisy10'):
                    key=(folder,arm,split,condition)
                    metrics[key],rows[key]=load_and_audit(*key)
                    units.append(dict(folder=folder,arm=arm,split=split,condition=condition,count=COUNTS[split],views=2))
    for split in ('val','test'):
        for condition in ('clean','noisy10'):
            assert len({m['input_digest'] for k,m in metrics.items() if k[2:]==(split,condition)})==1
    comparisons={}
    for folder in folders:
        for split in ('val','test'):
            for condition in ('clean','noisy10'):
                comparisons[f'{folder}/{split}/{condition}/rgb_minus_asinh_gray']=paired_interval(
                    rows[folder,'asinh_gray',split,condition],rows[folder,'asinh_rgb',split,condition])
    for folder in folders:
        if not all((OUTPUT/'gallery'/f'{folder}_{cond}_{part}.png').exists() for cond in ('clean','noisy10') for part in (1,2)):
            gallery(folder)
    cost=costs()
    json_write(OUTPUT/'costs.json',cost)
    json_write(OUTPUT/('pretrained_checks.json' if partial else 'completion_checks.json'),
        dict(status='passed',units=units,identical_inputs_across_arms=True,comparison_intervals=comparisons,
             visual_review='pending; PNG generation is not human/model visual inspection'))
    lines=['# LST：自然图像预训练 SD VAE 与可逆颜色编码',
        '', '状态：'+('预训练权重已完成全量评估；5 轮微调尚未完成。' if partial else '三组均完成 5 轮微调及全量 val/test；图片等待逐张视觉复核。'),'',
        '## 实验问题与公平性','',
        '本实验检验把 LST 温度残差映射到 RGB 路径，能否改善自然图像预训练 VAE 的重建与缺失恢复。'
        '三组共用官方 stabilityai/sd-vae-ft-mse（约 8,365 万参数）及相同初始化，先评估未微调权重，再独立微调全部编码器和解码器参数。'
        '这是 SD 的 AutoencoderKL，不包含扩散 UNet，也没有 VQ 码本。使用最后一轮检查点，不按 test 分数选择。','',
        'HLP v2 固定划分：train 218,975、val 11,684、test 11,757。训练 seed=17，有效 batch=64，'
        '每组 5 个完整 epoch、1,094,875 次记录呈现、17,110 次更新；每轮末批 31 条，不丢弃、不补齐。'
        '各组记录顺序、10% 缺失生成、归一化和物理损失一致。完整评估每个条件每条记录 2 个视图；clean 两视图相同，不能当作两倍独立样本。','',
        '## 三种表示','',
        '三组都从可见输入计算均温 μ，并编码 r=T−μ；不读取缺失真值来计算 μ。解码需要传输同一个 float32 均温（32 bit/图）。'
        '用源有效支持 [-50,70]°C 推得保守残差界 ±120°C，不使用每幅真值的最大值、最小值或标准差。'
        '缺失位置填可见均温；保留预训练三通道结构，mask 不作为额外网络通道，因此网络无法显式区分均温像元和缺失填充值。', '',
        '| 组别 | 编码 | 目的 |','|---|---|---|',
        '| 线性灰度 | u=(r/120+1)/2，复制至 RGB | 普通灰度输入控制 |',
        '| 非线性灰度 | u=[asinh(r/s)/asinh(120/s)+1]/2，复制至 RGB | 检查非线性对比度变换 |',
        '| 非线性 RGB | 同一个 u，沿黑→红→黄→绿→青→蓝→品红→白路径 | 单独检查颜色路径的增益 |','',
        's=2.6950881812941048°C，只由当前训练成员估计，沿用 A06 固定统计。颜色路径浮点输出，无 8-bit 量化。'
        'RGB 解码为最近线段投影后逆变换；偏离路径的预测不保证无损，额外报告 off-path MSE 和端点饱和率。'
        '线性控制采用保守全支持，图内对比度较低；非线性灰度是避免把对比度变化误归因于颜色的关键控制。'
        '三组都加入均温残差适配器，未微调分数不是“原始温度直接输入 SD”的分数。','',
        '训练目标：归一化温度 MSE + 3×去均值空间 MSE + 0.1×梯度 MSE + 0.01×表示辅助损失 + 10⁻⁶×平均 KL。'
        'RGB 表示辅助 MSE 乘 3/49，使同线段局部标量扰动的权重与复制灰度匹配；拐点和远距离不等价。'
        '训练采样后验，评估取后验均值。训练主干 BF16、最后输出卷积 FP32，评估全 FP32。'
        '不使用 LPIPS/GAN，以保持温度物理损失清晰。','',
        '## 全量物理指标','',
        'RMSE、均温误差、去均值空间误差以 °C 报告，梯度误差以相邻像元温差的 °C 报告。'
        '先在每幅图有效像元内计算 MSE，再等权平均图像和视图，最后开方；不把 RGB 像素误差当作温度 RMSE。','']
    for split in ('val','test'):
        lines += [f'### {split}（{COUNTS[split]:,} 对）','',
            '| 权重 | 表示 | 条件 | RMSE°C | 均温 | 空间 | 梯度 | 缺失区 RMSE |','|---|---|---|---:|---:|---:|---:|---:|']
        for folder in folders:
            for arm in ARMS:
                for condition in ('clean','noisy10'):
                    m=metrics[folder,arm,split,condition];r=m['regions']['lst/all'];hidden=m['regions'].get('lst/corrupt')
                    lines.append(f"| {'预训练' if folder=='pretrained' else '微调5轮'} | {NAMES[arm]} | {condition} | {r['rmse']:.4f} | {m['mean_rmse']:.4f} | {m['spatial_rmse']:.4f} | {r['gradient_rmse']:.4f} | {format(hidden['rmse'],'.4f') if hidden else '—'} |")
        lines.append('')
    lines+=['## 颜色贡献的配对比较','','下表为 RGB 减去非线性灰度的 RMSE；负数表示 RGB 更低。按 tile 配对 bootstrap 2,000 次。', '',
        '| 权重/集合/条件 | ΔRMSE°C | 95% 配对区间 |','|---|---:|---|']
    for key,value in comparisons.items():
        lines.append(f"| {key} | {value['delta_rmse_celsius']:+.4f} | [{value['paired_tile_ci95'][0]:+.4f}, {value['paired_tile_ci95'][1]:+.4f}] |")
    lines+=['','## 连续潜变量与计算量','',
        '潜变量为 4×32×32=4,096 个连续值，另加均温侧信息。按 FP32 名义存储约 131,104 bit/图，'
        '未进行熵编码，不是实测压缩文件大小。与历史 VAR VQ-VAE 的 680 个离散索引不等码率，不能宣称架构的公平压缩优势。', '',
        '三组共享一张 RTX 5090 并行执行。下表是各作业占用这张共享卡的墙钟时长，包含每轮完整验证与存盘；'
        '不能相加作为实际单卡 GPU-hours，也不能称纯计算时间。','',
        '| 模型 | 训练阶段共享GPU作业小时 | 最终评估秒 |','|---|---:|---:|']
    for arm,cc in cost['per_arm'].items():
        lines.append(f"| {NAMES[arm]} | {cc['train_stage_shared_gpu_job_hours']:.3f} | {cc['final_evaluation_seconds']:.1f} |")
    lines += ['',f"采样监测的整卡显存峰值：{cost['peak_global_used_gib']:.2f} GiB（包含其他进程；进程内峰值见各 metrics/status）。",
        f"三组训练阶段时间区间的并集：{cost['physical_single_gpu_training_stage_union_hours']:.3f} 单卡 GPU-hours（含验证/存盘；不是独占计算积分）。",
        '', 'KL、后验标准差、路径投影误差和端点饱和率见各 metrics.json 的 vae_diagnostics；不存在码本困惑度。', '',
        '## 固定样本图与复核','',
        '固定 8 个 test 记录、view 0；每行共用参考温度色阶，三个误差图共用对称色阶。'
        '按分位色阶显示，图外极值可能被视觉饱和；所有指标仍使用全部有效像元、未经显示裁剪。'
        '图片数值已对照完整评估逐记录结果，视觉结论待逐张查看后填写。','']
    for folder in folders:
        for cond in ('clean','noisy10'):
            for part in (1,2):lines.append(f'![{folder} {cond} {part}](gallery/{folder}_{cond}_{part}.png)')
    lines+=['','## 来源与复现','',
        f'- 官方权重：[stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)，固定 revision `{read(OUTPUT/"pretrained.json")["revision"]}`。',
        '- 方法启发：[Vision Banana](https://arxiv.org/html/2604.20329v3)。这里用有符号温度残差 asinh，属于适配实验，不是原文深度公式的复现。',
        '- 配置、源文件与权重摘要：manifest.json；数据验证：data_preflight.json；编码/梯度检查：numerical_checks.json。',
        '- 冻结训练预算与随机续训校验：smoke/gate.json、full/gate.json。完整统计：各组 final_* 下 metrics.json、samples.jsonl、record_ids.npy。',
        '- 续跑：`.venv-paper1/Scripts/python.exe -m experiments.paper1.lst_sd_color.supervise`。先检查 STOP 原因及活跃进程，不启动重复作业。',
        '- 本轮只有一个训练 seed；区间反映 test/val tile 抽样差异，不反映重新训练的随机性。自然图像预训练成员与 HLP 的潜在重叠未逐张检索。','']
    (OUTPUT/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
