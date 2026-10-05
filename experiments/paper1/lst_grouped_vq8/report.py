"""Verify complete evaluations before presenting A06 comparisons."""
from collections import defaultdict
from datetime import datetime
import json
import numpy as np
from eo_data.core import sha256_file
from .prepare import OUTPUT, HERE, ROOT, ARMS, LABELS
from experiments.paper1.lst_grouped_vq.io import read, json_write
from experiments.paper1.lst_grouped_vq.data import DatasetV2


def verify_evaluations(phase):
    rows, inputs = [], {}
    for arm,label in zip(ARMS,LABELS):
        name=f'{phase}_lst_grouped_{arm}_seed17'
        path=OUTPUT/phase/name
        c=read(HERE/'configs'/phase/(name+'.json'))
        cp=sha256_file(path/'last.pt')
        for split in ('val','test'):
            ds=DatasetV2(c,split,.1)
            members=set(map(int,ds.record_ids))
            ds.close()
            for condition in ('clean','noisy10'):
                dest=path/f'final_{split}_{condition}'
                m=read(dest/'metrics.json')
                assert m['checkpoint_sha256']==cp and m['evaluated_pairs']==len(members)
                assert set(np.load(dest/'record_ids.npy').tolist())==members
                assert sha256_file(dest/'samples.jsonl')==m['samples_sha256']
                assert sha256_file(dest/'token_counts.npz')==m['token_counts_sha256']
                assert inputs.setdefault((split,condition),m['input_digest'])==m['input_digest']
                samples=[json.loads(s) for s in (dest/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
                all_rows=[r for r in samples if r['region']=='all']
                groups=defaultdict(list)
                for r in all_rows:
                    groups[int(r['sample_id'].rsplit(':',1)[1])].append(r['view'])
                assert set(groups)==members and all(sorted(v)==[0,1] for v in groups.values())
                mse=float(np.mean([r['mse_physical'] for r in all_rows]))
                mean=float(np.mean([r['bias_physical']**2 for r in all_rows]))
                assert np.isclose(mse,m['regions']['lst/all']['rmse']**2,rtol=1e-10)
                assert np.isclose(mean,m['mean_rmse']**2,rtol=1e-10)
                assert np.isclose(m['spatial_rmse']**2+mean,mse,rtol=1e-10)
                counts=np.load(dest/'token_counts.npz')['counts']
                assert counts.shape==(c['model']['groups'],10,c['model']['codes_per_group'])
                for si,pn in enumerate(c['model']['var_scales']):
                    assert (counts[:,si].sum(1)==len(members)*2*pn*pn).all()
                rows.append(dict(m,model=arm,label=label,condition=condition))
    if phase == 'full':
        old_root = ROOT/'runs/paper1/lst_grouped_vq_20260924/full/full_lst_grouped_g4k256_seed17'
        for row in rows:
            old = read(old_root/f"final_{row['split']}_{row['condition']}/metrics.json")
            assert row['input_digest'] == old['input_digest'], 'A06 g4 evaluation inputs differ'
            assert row['data_fingerprint'] == old['data_fingerprint']
    json_write(OUTPUT/phase/'evaluation_gate.json',dict(status='passed',units=len(rows),identical_inputs=True,
        checkpoint_and_sample_hashes=True,token_histograms_verified=True,complete_members_and_two_views=True))
    return rows


def main():
    assert read(OUTPUT/'full/gate.json')['status']=='passed'
    rows=verify_evaluations('full')
    events=[json.loads(s) for s in (OUTPUT/'stages.jsonl').read_text(encoding='utf-8').splitlines()]
    active,intervals={},[]
    for e in events:
        key=e['phase'],e['stage'],e['name']
        if e['event']=='start':
            assert key not in active
            active[key]=datetime.fromisoformat(e['utc']).timestamp()
        else:
            assert key in active
            start=active.pop(key)
            upper=datetime.fromisoformat(e['utc']).timestamp()
            lower=datetime.fromisoformat(e.get('end_time_lower_bound_utc',e['utc'])).timestamp()
            intervals.append(dict(phase=key[0],stage=key[1],name=key[2],start=start,lower_end=lower,upper_end=upper))
    assert not active
    def union(phase,stage,bound):
        spans=sorted((i['start'],i[bound+'_end']) for i in intervals if i['phase']==phase and i['stage'] in stage)
        merged=[]
        for a,b in spans:
            if merged and a<=merged[-1][1]:
                merged[-1][1]=max(merged[-1][1],b)
            else:
                merged.append([a,b])
        return sum(b-a for a,b in merged)/3600
    times={phase+'_'+stage:{b:union(phase,stages,b) for b in ('lower','upper')}
           for phase,stage,stages in [('full','training',['train']),('full','evaluation',['evaluate']),('smoke','all',['partial','train','evaluate'])]}
    times['per_model_training_wall_hours']={a:{b:sum(i[b+'_end']-i['start'] for i in intervals if i['phase']=='full' and i['stage']=='train' and i['name']==f'full_lst_grouped_{a}_seed17')/3600 for b in ('lower','upper')} for a in ARMS}
    times['scope']='Single GPU process activity union; training includes full epoch validation and checkpoint I/O; concurrent wall hours cannot be summed. Some jobs can run alone.'
    times['intervals']=intervals
    json_write(OUTPUT/'gpu_hours.json',times)
    json_write(OUTPUT/'results.json',dict(status='complete',rows=rows,verified_units=len(rows),epochs=5,updates=17110,records_per_model=1094875))
    lines=['# A07：LST 八组码本续实验','',
        '两组均从零训练5轮，HLP v2 train218975、val11684、test11757；残差预测＋空间4倍，有效batch64，seed17，每组17110次更新、1094875条记录呈现。固定第5轮last.pt，未用test选模；test有历史监测记录。', '',
        'g8k16为21792 bit/图（含32bit可见均温），与A06 g4k256同码率；g8k256为43552 bit/图。额外表示容量与分组结构不能混为同一因素。参数接近但码本参数不同；不同子码本的编号分开统计。','',
        '| 集合 | 模型 | 输入 | 总体RMSE °C | 均值RMSE | 空间RMSE | 梯度RMSE | 缺失区RMSE | 保留区RMSE |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        z=r['regions']; hole=z.get('lst/corrupt',{}).get('rmse')
        lines.append(f"| {r['split']} | {r['label']} | {r['condition']} | {z['lst/all']['rmse']:.4f} | {r['mean_rmse']:.4f} | {r['spatial_rmse']:.4f} | {z['lst/all']['gradient_rmse']:.4f} | {f'{hole:.4f}' if hole is not None else '—'} | {z['lst/retained']['rmse']:.4f} |")
    lines+=['','## 子码本与最细尺度的图内变化','',
            '| 集合 | 模型 | 输入 | 组 | 使用码字 | 困惑度 | top1占比 | 最细层图内平均码字数 | 最细层图内熵 |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        for g in r['tokens']['lst']['per_group']:
            fine=g['scales'][-1]
            lines.append(f"| {r['split']} | {r['label']} | {r['condition']} | {g['group']} | {g['used_codes']}/{g['vocabulary']} | {g['perplexity']:.3f} | {g['top1_fraction']:.2%} | {fine['within_image_mean_unique_codes']:.3f} | {fine['within_image_mean_entropy_bits']:.3f} |")
    lines+=['','原始逐组逐尺度频数保存在每个评估目录token_counts.npz；统计覆盖所有传输位置，不能仅凭困惑度或图内熵上升宣称细节恢复。', '',
            f"训练阶段GPU-hours并集：{times['full_training']['lower']:.3f}-{times['full_training']['upper']:.3f}；最终评估：{times['full_evaluation']['lower']:.3f}-{times['full_evaluation']['upper']:.3f}；技术检查：{times['smoke_all']['lower']:.3f}-{times['smoke_all']['upper']:.3f}。包含训练期val和存盘，非纯GPU内核耗时；单模型可能同时经历并行和独占阶段，详见gpu_hours.json。",'']
    for a,label in zip(ARMS,LABELS):
        t=times['per_model_training_wall_hours'][a]
        lines.append(f"- {label}训练作业墙钟：{t['lower']:.3f}-{t['upper']:.3f}小时。")
    (OUTPUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    baseline = read(ROOT/'runs/paper1/lst_grouped_vq_20260924/results.json')
    references = [r for r in baseline['rows'] if r['model']=='g4k256']
    json_write(OUTPUT/'comparison_results.json', dict(status='passed', reference_experiment='A06', identical_inputs=True, same_five_epoch_training_stream=True, rows=references+rows))
    json_write(OUTPUT/'completion_review.json',dict(status='passed',units=len(rows),report_sha256=sha256_file(OUTPUT/'REPORT_ZH.md'),
        remaining_delivery='Fixed-record paired gallery and visual review; Chinese interpretation; registry completion and notify user.'))


if __name__=='__main__':
    main()
