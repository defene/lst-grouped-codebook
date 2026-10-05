"""Verify finished A05 results and write the Chinese report with measured GPU time."""
import json
from datetime import datetime
from collections import defaultdict
import numpy as np
from eo_data.core import json_write, sha256_file
from .prepare import OUTPUT, HERE, ARMS, ROOT
from .data import DatasetV2

LABELS = ['原始预测', '原始预测＋空间4倍', '残差预测', '残差预测＋空间4倍']


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    assert read(OUTPUT/'full/gate.json')['status']=='passed'
    rows = []
    digests = {}
    for arm, label in zip(ARMS,LABELS):
        name=f'full_lst_v2_{arm}_seed17'
        path=OUTPUT/'full'/name
        c=read(HERE/'configs/full'/f'{name}.json')
        state=read(path/'status.json')
        assert state['step']==17110 and state['records_presented']==1094875
        cp=sha256_file(path/'last.pt')
        for split in ('val','test'):
            ds=DatasetV2(c,split,.1)
            members=set(map(int,ds.record_ids))
            ds.close()
            for condition in ('clean','noisy10'):
                dest=path/f'final_{split}_{condition}'
                m=read(dest/'metrics.json')
                assert m['checkpoint_sha256']==cp
                assert m['evaluated_pairs']==len(members)
                assert set(np.load(dest/'record_ids.npy').tolist())==members
                assert sha256_file(dest/'samples.jsonl')==m['samples_sha256']
                key=split,condition
                assert digests.setdefault(key,m['input_digest'])==m['input_digest']
                sample_rows=[json.loads(s) for s in (dest/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
                sample_rows=[r for r in sample_rows if r['region']=='all']
                groups=defaultdict(list)
                for r in sample_rows:
                    groups[int(r['sample_id'].rsplit(':',1)[1])].append(r['view'])
                assert set(groups)==members and all(sorted(v)==[0,1] for v in groups.values())
                mse=float(np.mean([r['mse_physical'] for r in sample_rows]))
                mean=float(np.mean([r['bias_physical']**2 for r in sample_rows]))
                assert np.isclose(m['regions']['lst/all']['rmse']**2,mse,rtol=1e-10)
                assert np.isclose(m['mean_rmse']**2,mean,rtol=1e-10)
                assert np.isclose(m['spatial_rmse']**2+mean,mse,rtol=1e-10)
                rows.append(dict(m,model=arm,label=label,condition=condition))
    events=[json.loads(s) for s in (OUTPUT/'stages.jsonl').read_text(encoding='utf-8').splitlines()]
    gpu=[json.loads(s) for s in (OUTPUT/'gpu.jsonl').read_text(encoding='utf-8').splitlines()]
    intervals=[]
    lower_intervals=[]
    active={}
    for e in events:
        if e['phase']!='full':
            continue
        key=e['name'],e['stage']
        if e['event']=='start':
            active[key]=datetime.fromisoformat(e['utc']).timestamp()
        elif key in active:
            start=active.pop(key)
            intervals.append((start,datetime.fromisoformat(e['utc']).timestamp(),*key))
            lower_intervals.append((start,datetime.fromisoformat(e.get('end_time_lower_bound_utc',e['utc'])).timestamp(),*key))
    def union_hours(stage, bounds=None):
        spans=sorted((a,b) for a,b,_,s in (intervals if bounds is None else bounds) if s==stage)
        merged=[]
        for a,b in spans:
            if merged and a<=merged[-1][1]:
                merged[-1][1]=max(merged[-1][1],b)
            else:
                merged.append([a,b])
        return sum(b-a for a,b in merged)/3600
    runtime={'training_stage_gpu_hours_union':union_hours('train'),
        'training_stage_gpu_hours_union_lower_bound':union_hours('train',lower_intervals),
        'training_stage_gpu_hours_union_upper_bound':union_hours('train'),
        'timing_note':'Reconstructed ends are uncertain: union and job wall point fields use upper bounds; report lower/upper ranges, not exact measured GPU time.',
        'final_evaluation_gpu_hours_union':union_hours('evaluate'),
        'shared_gpu_job_wall_hours':{f'full_lst_v2_{a}_seed17':sum(y-x for x,y,n,s in intervals if n==f'full_lst_v2_{a}_seed17' and s=='train')/3600 for a in ARMS},
        'shared_gpu_job_wall_hours_lower_bound':{f'full_lst_v2_{a}_seed17':sum(y-x for x,y,n,s in lower_intervals if n==f'full_lst_v2_{a}_seed17' and s=='train')/3600 for a in ARMS},
        'peak_global_used_gib':max(g['used_gib'] for g in gpu if g['phase']=='full'),
        'training_scope':'single GPU; training stage includes full epoch validation and checkpoint I/O; parallel job wall hours cannot be summed as GPU-hours',
        'unclosed_events':{str(k):v for k,v in active.items()}}
    recovered_ends=[e for e in events if e.get('end_time_is_upper_bound')]
    runtime['reconstructed_end_boundaries']=recovered_ends
    runtime['sum_end_boundary_uncertainty_seconds']=sum(
        (datetime.fromisoformat(e['utc'])-datetime.fromisoformat(e['end_time_lower_bound_utc'])).total_seconds()
        for e in recovered_ends)
    assert not active
    json_write(OUTPUT/'gpu_hours.json',runtime)
    json_write(OUTPUT/'results.json',{'status':'complete','rows':rows,'verified_units':16,'epochs':5,'updates':17110,'records_per_model':1094875})
    lines=['# HLP v2：四组 LST 模型各训练5个完整 epoch','',
        '四组均从随机初始化训练，使用 HLP v2 的218,975条训练记录及其归一化。每组恰好1,094,875次记录呈现、17,110次更新；每轮末次更新31条，未丢弃或重复填充尾批。种子17，有效batch64，微批4，累积16，训练输入均为约10%已知掩码缺失。', '',
        '固定第5轮最终检查点用于正式对照。每轮全量val监测，但不提前停止或按test选模。val为11,684条独立于本轮训练的记录；test为11,757条，存在此前历史实验的监测记录，不能视为从未查看的测试集。HLS和Prithvi不参与本轮。', '',
        '空间4倍表示原有MSE额外增加3倍逐图去均值空间MSE，VQ损失不变。残差模型只从可见输入计算逐图基准均值，量化图内残差，重建后加回基准；这不是额外噪声残差的预测。辅助均值32bit计入码率。', '',
        '| 集合 | 模型 | 输入 | RMSE °C | 均值 RMSE | 空间 RMSE | 梯度 RMSE | 缺失区 RMSE | 保留区 RMSE | 码字困惑度 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        regions=r['regions']; hole=regions.get('lst/corrupt',{}).get('rmse')
        lines.append(f"| {r['split']} | {r['label']} | {'干净' if r['condition']=='clean' else '缺失'} | {regions['lst/all']['rmse']:.4f} | {r['mean_rmse']:.4f} | {r['spatial_rmse']:.4f} | {regions['lst/all']['gradient_rmse']:.4f} | {f'{hole:.4f}' if hole is not None else '—'} | {regions['lst/retained']['rmse']:.4f} | {r['tokens']['lst']['perplexity']:.2f} |")
    lines+=['','RMSE按有效像素计算每条记录MSE，先平均两个实现，再对记录等权平均后开方。总体MSE＝均值MSE＋空间MSE；空间项包含大尺度变化，码字困惑度也不能单独证明细节恢复。16项均复核完整成员、两个实现、同条件输入摘要、检查点及逐条文件哈希与MSE分解。','',
        '## GPU 时间','',f"训练阶段单GPU作业占用时间区间：{runtime['training_stage_gpu_hours_union_lower_bound']:.3f}–{runtime['training_stage_gpu_hours_union_upper_bound']:.3f} GPU-hours；最终评估：{runtime['final_evaluation_gpu_hours_union']:.3f} GPU-hours。训练阶段含每轮全量val和存盘。由于中断时刻不确定，训练时间报告上下界，不能视为精确GPU计算时长。整卡采样峰值占用：{runtime['peak_global_used_gib']:.3f} GiB。",'',
        '| 模型 | 与另一作业共享GPU时的训练作业墙钟小时 |','|---|---:|']
    for arm,label in zip(ARMS,LABELS):
        lines.append(f"| {label} | {runtime['shared_gpu_job_wall_hours_lower_bound'][f'full_lst_v2_{arm}_seed17']:.3f}–{runtime['shared_gpu_job_wall_hours'][f'full_lst_v2_{arm}_seed17']:.3f} |")
    lines+=['','这些单模型墙钟时长不是独占GPU测速，也不能相加冒充实际GPU-hours。原始起止事件见stages.jsonl，显存采样见gpu.jsonl。失败恢复所丢弃的运算也计入活动区间，但不额外计入已提交的训练样本预算。','']
    if recovered_ends:
        lines += [f"本次存在调度器文件写入故障及后续系统中断后的检查点恢复。2026-09-17日志停止附近有D盘I/O错误，随后系统更新重启；最初中断原因未确认。后一次实际结束时间仅能界定在最后GPU活动记录06:08:53 UTC与首次计划重启08:59:06 UTC之间，故按区间报告耗时，并排除已确定的停机空档。{len(recovered_ends)}个重建结束边界的逐事件不确定性合计{runtime['sum_end_boundary_uncertainty_seconds']:.2f}秒；同卡并行的重叠区间只计一次。原事件与恢复核验保留在stages.jsonl和recovery目录。冻结训练代码、配置和样本预算未改变。",'']
    (OUTPUT/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    json_write(OUTPUT/'completion_review.json',{'status':'passed','units':16,'training_stream':'matched_all_four','budget':'exact_5_epochs',
        'report_sha256':sha256_file(OUTPUT/'REPORT_ZH.md')})


if __name__=='__main__':
    main()
