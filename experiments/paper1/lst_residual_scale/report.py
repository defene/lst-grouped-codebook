"""Audited full-set comparison with historical A05 control and fixed galleries."""
import gc
import json
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
from eo_data.core import sha256_file, json_write
from eo_denoise import engine
from .prepare import OUTPUT, BASE, BASE_RUN, NAME, read
from .data import DatasetV2
from . import runtime
from .train import read_checkpoint

RUN=OUTPUT/'full'/NAME


def audited(run,split,condition):
    p=run/f'final_{split}_{condition}'
    m=read(p/'metrics.json')
    assert m['checkpoint_sha256']==sha256_file(run/'last.pt')
    assert sha256_file(p/'samples.jsonl')==m['samples_sha256']
    c=read(run/'config.json');ds=DatasetV2(c,split,.1)
    assert np.array_equal(np.load(p/'record_ids.npy'),ds.record_ids)
    assert m['evaluated_pairs']==len(ds) and m['realizations']==2
    assert ds.fingerprint==m['data_fingerprint']
    members=set(ds.record_ids.tolist());ds.close()
    rows=[json.loads(s) for s in (p/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[r for r in rows if r['region']=='all']
    groups=defaultdict(list)
    for r in rows:groups[int(r['sample_id'].rsplit(':',1)[1])].append(r['view'])
    assert set(groups)==members and all(sorted(v)==[0,1] for v in groups.values())
    mse=np.mean([r['mse_physical'] for r in rows]);bias=np.mean([r['bias_physical']**2 for r in rows])
    assert np.isclose(m['regions']['lst/all']['rmse']**2,mse,rtol=1e-10)
    assert np.isclose(m['mean_rmse']**2,bias,rtol=1e-10)
    assert np.isclose(m['spatial_rmse']**2+bias,mse,rtol=1e-10)
    return m,rows


def timing():
    events=[json.loads(s) for s in (OUTPUT/'stages.jsonl').read_text(encoding='utf-8').splitlines()]
    starts={};seconds=defaultdict(float)
    for e in events:
        key=(e['phase'],e['stage'],e['pid'])
        if e['event']=='start':starts[key]=datetime.fromisoformat(e['utc']).timestamp()
        elif key in starts:seconds[(e['phase'],e['stage'])]+=datetime.fromisoformat(e['utc']).timestamp()-starts.pop(key)
    # This report stage is still running. All training and evaluation spans must be closed.
    assert not any(k[1] in ('train','evaluate','partial') for k in starts)
    gpu=[json.loads(s) for s in (OUTPUT/'gpu.jsonl').read_text(encoding='utf-8').splitlines()]
    result=dict(training_stage_single_gpu_job_hours=seconds['full','train']/3600,
        final_evaluation_single_gpu_job_hours=seconds['full','evaluate']/3600,
        smoke_stage_seconds=sum(v for (p,s),v in seconds.items() if p.startswith('smoke')),
        peak_global_used_gib=max(g['used_gib'] for g in gpu if g['phase']=='full'),
        scope='Single sequential experiment job. Train stage includes full epoch validation and checkpoint I/O; not pure CUDA kernel time. Background display GPU use is not subtracted.')
    json_write(OUTPUT/'gpu_hours.json',result)
    return result


@torch.inference_mode()
def gallery():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
    dest=OUTPUT/'gallery';dest.mkdir(exist_ok=True)
    with np.load(BASE/'gallery/predictions_8_records.npz') as z:old={k:z[k] for k in z.files}
    ids=old['record_ids'].tolist()
    assert ids==[231773,232129,232414,232565,232620,233073,233356,234070]
    c=read(RUN/'config.json');d=DatasetV2(c,'train')
    state=read_checkpoint(RUN/'last.pt',c,d);d.close()
    engine.seed_everything(c['seed']);torch.set_num_threads(4)
    device=engine.device_for(c);model=runtime.ExperimentModel(c).to(device).eval()
    model.load_state_dict(state['model']);del state
    artifacts=[];checks=[];saved={'record_ids':np.array(ids)}
    for condition,coverage in [('clean',0),('noisy10',.1)]:
        ds=DatasetV2(c,'test',coverage);positions={int(r):i for i,r in enumerate(ds.record_ids)}
        entries=[ds[positions[r]] for r in ids]
        pred=[]
        for i in range(0,8,4):
            x=torch.from_numpy(np.stack([e['lst']['image'][0] for e in entries[i:i+4]])).to(device)
            mask=torch.from_numpy(np.stack([e['lst']['input_mask'][0] for e in entries[i:i+4]])).to(device)
            with engine.amp_context(c,device):y=model.codecs['lst'](x,mask)['reconstruction']
            pred.append(y.float().cpu().numpy())
        pred=np.concatenate(pred);std=np.array(ds.sources['lst'].stats['std'],np.float32)[:,None,None]
        mean=np.array(ds.sources['lst'].stats['mean'],np.float32)[:,None,None]
        temp=(pred*std+mean)/100
        target=np.stack([e['lst']['target'] for e in entries]);valid=np.stack([e['lst']['reference_mask'][0] for e in entries]).astype(bool)
        np.testing.assert_allclose((target*std+mean)/100,old['target'],atol=1e-5)
        np.testing.assert_array_equal(valid,old['reference_mask'])
        inp=(np.stack([e['lst']['image'][0] for e in entries])*std+mean)/100
        np.testing.assert_allclose(inp,old[condition+'_input'],atol=1e-5)
        rows=[json.loads(s) for s in (RUN/f'final_test_{condition}/samples.jsonl').read_text(encoding='utf-8').splitlines()]
        expected={int(r['sample_id'].rsplit(':',1)[1]):r['mse_physical'] for r in rows if r['region']=='all' and r['view']==0}
        for i,rid in enumerate(ids):
            mse=float(np.mean((((pred[i]-target[i])*std/100)[:,valid[i]]).astype(np.float64)**2))
            assert np.isclose(mse,expected[rid],rtol=.002,atol=1e-7)
            checks.append(dict(record_id=rid,condition=condition,mse=mse,expected=expected[rid]))
        ds.close();saved[condition+'_scaled']=temp
        baseline=old['residual_spatial4_'+condition]
        for page in range(2):
            fig,axs=plt.subplots(4,6,figsize=(17,11),layout='constrained')
            for row,i in enumerate(range(page*4,page*4+4)):
                ref=old['target'][i,0];support=valid[i];lo,hi=np.quantile(ref[support],[.01,.99])
                if hi-lo<.01:lo-=.5;hi+=.5
                errs=[baseline[i,0]-ref,temp[i,0]-ref]
                limit=max(.1,float(np.quantile(np.abs(np.stack(errs)[:,support]),.99)))
                values=[ref,inp[i,0],baseline[i,0],temp[i,0],*errs]
                for col,value in enumerate(values):
                    mask=old[condition+'_mask'][i] if col==1 else support
                    cmap=plt.get_cmap('RdBu_r' if col>=4 else 'inferno').copy();cmap.set_bad('#888888')
                    im=axs[row,col].imshow(np.ma.masked_where(~mask,value),cmap=cmap,
                        vmin=-limit if col>=4 else lo,vmax=limit if col>=4 else hi,interpolation='nearest')
                    axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
                    if row==0:axs[row,col].set_title(['参考温度','输入','原残差＋空间4倍','缩放残差＋空间4倍','原残差误差','缩放残差误差'][col],fontsize=9)
                    if col==0:axs[row,col].set_ylabel(str(ids[i]))
                    if col==3:fig.colorbar(im,ax=axs[row,:4],shrink=.7,pad=.01,label='°C')
                    if col==5:fig.colorbar(im,ax=axs[row,4:],shrink=.7,pad=.01,label='预测−参考 °C')
            fig.suptitle(f'LST残差尺度消融 · 5轮 · {"干净输入" if condition=="clean" else "约10%缺失输入"}')
            fig.supxlabel('固定历史8例；每行温度共用参考1%–99%色标，误差共用对称99%色标；超范围仅显示截色。',fontsize=9)
            file=dest/f'{condition}_{page+1}.png';fig.savefig(file,dpi=150);plt.close(fig)
            artifacts.append(dict(path=str(file),sha256=sha256_file(file)))
    np.savez_compressed(dest/'predictions.npz',**saved)
    json_write(dest/'checks.json',dict(status='passed',records=ids,checks=checks,artifacts=artifacts,
        visual_review='pending human/assistant inspection; numerical gallery checks passed'))
    del model;gc.collect();torch.cuda.empty_cache()
    return artifacts


def main():
    assert read(OUTPUT/'full/gate.json')['status']=='passed'
    rows=[];comparisons=[]
    for split in ('val','test'):
        for condition in ('clean','noisy10'):
            old,old_rows=audited(BASE_RUN,split,condition)
            new,new_rows=audited(RUN,split,condition)
            assert old['input_digest']==new['input_digest']
            assert old['data_fingerprint']==new['data_fingerprint']
            for label,m in [('原残差＋空间4倍',old),('缩放残差＋空间4倍',new)]:
                rows.append(dict(m,label=label,condition=condition))
            grouped=[]
            for samples in (old_rows,new_rows):
                g=defaultdict(list)
                for r in samples:g[r['tile_id']].append(r['mse_physical'])
                grouped.append(g)
            tiles=sorted(grouped[0]);assert tiles==sorted(grouped[1])
            draws=np.random.default_rng(20260918).integers(0,len(tiles),(2000,len(tiles)))
            boot=[]
            for g in grouped:
                values=np.array([[sum(g[t]),len(g[t])] for t in tiles])[draws].sum(1)
                boot.append(np.sqrt(values[:,0]/values[:,1]))
            a=old['regions']['lst/all']['rmse'];b=new['regions']['lst/all']['rmse']
            comparisons.append(dict(split=split,condition=condition,delta_rmse=b-a,
                relative_reduction=1-b/a,paired_tile_ci95=np.quantile(boot[1]-boot[0],[.025,.975]).tolist()))
    time=timing();artifacts=gallery();scale=read(OUTPUT/'residual_scale.json')
    json_write(OUTPUT/'results.json',dict(status='complete',rows=rows,comparisons=comparisons))
    lines=['# LST 残差尺度消融：完整5轮对照','',
        '新模型从零训练，与已有 A05 残差＋空间4倍模型比较；仅新增固定残差缩放。相同seed17、训练成员、样本顺序、噪声、损失系数、有效batch64；每模型1,094,875次记录呈现、17,110次更新。', '',
        f"尺度仅由218,975条训练记录计算：s={scale['normalized_scale']:.8f}，残差输入放大{scale['input_gain']:.4f}倍；还原后计算原有损失，VQ损失系数不变。不是逐图标准差归一化。",'',
        '每轮全量val，固定最后第5轮检查点；val/test完整评估，各两个实现，匹配输入摘要与数据指纹。干净输入属于缺失训练模型的条件泛化测试；单种子，test已有历史查看记录。','',
        '| 集合 | 输入 | 模型 | 总体RMSE °C | 均值RMSE | 空间RMSE | 梯度RMSE | 缺失区RMSE | 困惑度 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        a=r['regions']['lst/all'];hole=r['regions'].get('lst/corrupt',{}).get('rmse')
        h='—' if hole is None else f'{hole:.4f}'
        lines.append(f"| {r['split']} | {r['condition']} | {r['label']} | {a['rmse']:.4f} | {r['mean_rmse']:.4f} | {r['spatial_rmse']:.4f} | {a['gradient_rmse']:.4f} | {h} | {r['tokens']['lst']['perplexity']:.2f} |")
    lines+=['','差值=新增缩放减去原残差；负值表示RMSE降低。以下区间为配对tile bootstrap 2,000次，不包含训练随机种子不确定性。','']
    for p in comparisons:
        lines.append(f"- {p['split']}/{p['condition']}：ΔRMSE={p['delta_rmse']:+.4f}°C，95%区间{p['paired_tile_ci95']}，相对下降{100*p['relative_reduction']:.2f}%。")
    lines+=['',f"本次新模型训练阶段单GPU作业耗时 {time['training_stage_single_gpu_job_hours']:.3f} GPU-hours；最终全量评估 {time['final_evaluation_single_gpu_job_hours']:.3f} GPU-hours。训练阶段包含每轮val和存盘，不等于纯CUDA计算时间。整卡采样峰值 {time['peak_global_used_gib']:.3f} GiB；技术检查单独记录。",'',
        '码率保持6800 bit索引＋32 bit逐图均值；尺度为共享模型常量。码字困惑度不能单独证明细节恢复。RMSE在有效像素内按记录等权统计；输入分布改变也会影响内部VQ优化动态。','',
        '固定沿用历史8例；预测MSE与完整评估核对通过。以下图已自动生成，仍需视觉复核，不能据此自动宣称细节改善。','']
    for a in artifacts:lines.append(f"![固定预测与误差]({a['path']})")
    (OUTPUT/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    json_write(OUTPUT/'completion_checks.json',dict(status='passed',verified_units=8,inputs_matched=True,
        exact_budget=True,samples_hashes_verified=True,visual_review='pending',report_sha256=sha256_file(OUTPUT/'REPORT_ZH.md')))


if __name__=='__main__':main()
