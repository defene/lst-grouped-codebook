"""Summarize all five completed A06/A07 arms; no training or metric selection."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import os
import numpy as np
from .gallery import ROOT, OLD, NEW, OUTPUT, ARMS


def read(p):
    return json.loads(p.read_text('utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def link(p):
    return Path(os.path.relpath(p,OUTPUT)).as_posix()


def main():
    rows=[]; provenance=[]; inputs={}; fingerprints={}; member_sets={}
    gal=read(OUTPUT/'gallery/review.json')
    assert gal['status']=='passed' and len(gal['checks'])==80
    for artifact in gal['artifacts']:
        assert sha(OUTPUT/'gallery'/artifact['path'])==artifact['sha256']
    assert sha(OUTPUT/'gallery/predictions_8_records.npz')==gal['predictions_sha256']
    for arm in ARMS:
        root=OLD if arm in ARMS[:3] else NEW
        run=root/'full'/f'full_lst_grouped_{arm}_seed17'
        c=read(run/'config.json'); st=read(run/'status.json')
        assert st['step']==17110 and st['records_presented']==1094875 and st['validated_epoch']==5
        archived={ (x['split'],x['condition']):x for x in read(root/'results.json')['rows'] if x['model']==arm }
        for split,n in [('val',11684),('test',11757)]:
            for cond in ('clean','noisy10'):
                dest=run/f'final_{split}_{cond}'
                r=read(dest/'metrics.json')
                assert r['evaluated_pairs']==n and r['realizations']==2
                ids=set(np.load(dest/'record_ids.npy').tolist())
                assert len(ids)==n and member_sets.setdefault(split,ids)==ids
                assert inputs.setdefault((split,cond),r['input_digest'])==r['input_digest']
                assert fingerprints.setdefault(split,r['data_fingerprint'])==r['data_fingerprint']
                assert sha(dest/'samples.jsonl')==r['samples_sha256']
                assert sha(dest/'token_counts.npz')==r['token_counts_sha256']
                assert r['checkpoint_sha256']==gal['checkpoint_hashes'][arm]
                assert np.isclose(r['mean_rmse']**2+r['spatial_rmse']**2,r['regions']['lst/all']['rmse']**2,rtol=1e-10)
                assert all(archived[split,cond][k]==v for k,v in r.items())
                r.update(model=arm,condition=cond,experiment='A06' if root==OLD else 'A07')
                rows.append(r)
                provenance.append(dict(model=arm,split=split,condition=cond,path=link(dest/'metrics.json'),sha256=sha(dest/'metrics.json')))
    assert len(rows)==20
    keyed={(r['model'],r['split'],r['condition']):r for r in rows}
    settings=[(1,1024),(2,32),(4,256),(8,16),(8,256)]
    bits={a:680*g*int(np.log2(k))+32 for a,(g,k) in zip(ARMS,settings)}
    def val(a,s,c,region='lst/all'):
        return keyed[a,s,c]['regions'][region]['rmse']
    def pct(a,b,s,c,region='lst/all'):
        return 100*(1-val(a,s,c,region)/val(b,s,c,region))
    lines=['# 组合码本实验总览：A06＋A07全部5配置与新8个case','',
        '汇总日期：2026-10-02。范围为本线程已完成的组合码本对照线：A06 single1024、g2k32、g4k256，A07 g8k16、g8k256，共5个唯一模型、20个完整评估单元。single1024作为单码本基线一并列出；A07复用的g4结果不重复计数。', '',
        '## 结论','',
        '在已测配置中，g2k32是6832 bit/图档位的较优方案；g8k16是21792 bit/图档位的较优方案。g8k256取得最低全量总体RMSE，但索引位数再翻倍，仅带来约0.5%的test误差下降。若重视当前误差与位数的权衡，g8k16更值得保留为后续基线。所有结论仅限seed17，未证明多种子稳定优越性。','',
        '## 统一条件与表示预算','',
        '五模型均仅输入LST与mask，32维潜在表示、680个多尺度位置，选码后拼接再使用全通道Phi；可见均温残差预测＋空间误差4倍。HLP v2固定train218975/val11684/test11757，新train归一化和同A05 split-local QA库；没有HLS或Prithvi模型参与。每模型从零5完整epoch，17110更新、1094875条记录呈现，有效batch64、尾31实际加权。固定epoch5 last.pt，每轮完整val，最终每条件两个视图；未以test选模。','',
        '| 实验 | 配置 | 组数×K | 组内维数 | 码本参数 | bit/位置 | bit/图含均温 | 相对基线位数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for a,(g,k) in zip(ARMS,settings):
        lines.append(f"| {'A06' if a in ARMS[:3] else 'A07'} | {a} | {g}×{k} | {32//g} | {32*k} | {g*int(np.log2(k))} | {bits[a]} | {bits[a]/6832:.3f}× |")
    lines+=['','bit/图为固定索引加32bit可见均温；不含mask、模型、文件头，非实际熵编码文件大小。独立分组码字的笛卡尔积不是实际使用的联合码字数。g2与single同码率，g8k16与g4同码率；其余比较不能称同码率。','',
            '## 全量总体RMSE（°C，越低越好）','',
            '| 配置 | val clean | val noisy10 | test clean | test noisy10 |',
            '|---|---:|---:|---:|---:|']
    for a in ARMS:
        lines.append('| '+a+' | '+' | '.join(f'{val(a,s,c):.6f}' for s in ('val','test') for c in ('clean','noisy10'))+' |')
    lines+=['','## 同码率收益与扩容量收益','',
            '| 比较（新←旧） | 位数关系 | test clean RMSE降低 | test noisy10 RMSE降低 | noisy10缺失区RMSE降低 |',
            '|---|---|---:|---:|---:|']
    for a,b,desc in [('g2k32','single1024','同码率'),('g4k256','g2k32','总位数3.19倍'),('g8k16','g4k256','同码率'),('g8k256','g8k16','索引位数2倍')]:
        lines.append(f"| {a} ← {b} | {desc} | {pct(a,b,'test','clean'):.2f}% | {pct(a,b,'test','noisy10'):.2f}% | {pct(a,b,'test','noisy10','lst/corrupt'):.2f}% |")
    lines+=['','同码率分组改善总体误差，但缺失区改善明显更小。模型骨干保持一致，组数、每组维数和字典大小同时改变，不能将收益解释为纯组数的独立因果效应。','',
            '## 全量误差分解','',
            '单位°C。总体、均值、空间、梯度、缺失和保留误差均覆盖完整集合；clean无人工缺失区，记—。先记录内平均视图MSE，再记录等权平均开方。总体MSE=均值MSE+空间MSE；各区域独立归一，不能用一个全局缺失比例直接拼回总体RMSE。','',
            '| 集合 | 条件 | 配置 | 总体 | 均值 | 空间 | 梯度 | 缺失区 | 保留区 |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for s in ('val','test'):
        for c in ('clean','noisy10'):
            for a in ARMS:
                r=keyed[a,s,c];z=r['regions'];hole=z.get('lst/corrupt',{}).get('rmse');h='—' if hole is None else f'{hole:.6f}'
                lines.append(f"| {s} | {c} | {a} | {z['lst/all']['rmse']:.6f} | {r['mean_rmse']:.6f} | {r['spatial_rmse']:.6f} | {z['lst/all']['gradient_rmse']:.6f} | {h} | {z['lst/retained']['rmse']:.6f} |")
    lines+=['','## 码字使用与稳定性','',
            '下面为test noisy10逐组数据，组编号从0开始。top1/top10和困惑度跨尺度汇总；图内指标为最细16×16尺度。频数覆盖所有传输位置，包含参考无效区。翻转仅有按组跨尺度汇总，不能推断逐尺度翻转率。不同组同编号不是同一码字，单组困惑度不是联合容量。','',
            '| 配置 | 组 | 使用/K | 困惑度 | top1 | top10 | 图内独有码字 | 图内熵bit | 相邻变化 | 参考区翻转 | 共同可见区翻转 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        r=keyed[a,'test','noisy10']
        for g in r['tokens']['lst']['per_group']:
            fine=g['scales'][-1];gi=g['group']
            flip={k:next(x['flip_rate'] for x in v if x['group']==gi) for k,v in r['stability'].items()}
            lines.append(f"| {a} | {gi} | {g['used_codes']}/{g['vocabulary']} | {g['perplexity']:.3f} | {g['top1_fraction']:.2%} | {g['top10_fraction']:.4%} | {fine['within_image_mean_unique_codes']:.3f} | {fine['within_image_mean_entropy_bits']:.3f} | {fine['within_image_adjacent_change_fraction']:.2%} | {flip['reference']:.2%} | {flip['common_visible']:.2%} |")
    lines+=['','用过许多码字不等于使用均匀；低翻转可能伴随强top1集中。token变化和高熵都不能单独证明细节恢复正确。完整val/test逐组逐尺度统计与原始频数见下列原始报告附录：',
            f"- [A06全部3配置统计]({link(OLD/'CODEBOOK_DETAILS_ZH.md')})",
            f"- [A07两配置及g4参考统计]({link(NEW/'CODEBOOK_DETAILS_ZH.md')})",'',
            '## 新8个case：五配置同图比较','',
            '按(SHA256("grouped-vq-case-20261002:"＋记录ID), 记录ID)排序取前8条，排除旧8条，未依据图像或模型误差筛选；只改变展示抽样，没有修改数据划分。记录：'+ '、'.join(map(str,gal['image_record_ids']))+'。','',
            '展示view0。每一行所有模型和输入共用参考图1%–99%色标，超出范围截色；灰色为参考无效/输入缺失。图下RMSE是该记录view0的全有效区误差，不是缺失区误差或全量均值。输入与checkpoint哈希核对通过，80个模型×条件×记录的MSE与原全量评估对齐。','']
    for c,label in [('clean','干净输入'),('noisy10','约10%缺失输入')]:
        for page in (1,2):
            lines += [f'### {label}，第{1 if page==1 else 5}–{4 if page==1 else 8}条','',
                      f'![{label}，第{page}页](gallery/predictions_{c}_{page}.png)','']
    lines+=['四张PNG均已逐张实际打开检查，标签、统一色标和单图RMSE可读。第1页的240455、235502、233332显示分组模型在一些局部冷热结构上改善；236343干净输入的河道轮廓更清楚，但遮挡河道后所有模型都未能正确连接。第2页的235386和238221温差很小，绝对RMSE低，但细碎纹理与条带仍被抹平；低RMSE不等于这些结构已恢复。234327和237686仍存在纹理平滑和缺失区域形状偏差。','',
            '这8条样例中g8k16经常优于g8k256，与全量g8k256略优不矛盾：小样本的局部排名不能替代全量结果。样例规则固定且保留，不能再因排名不符合预期而重选。','',
            '## 训练与评估用时','',
            '两次实验分别使用同一块GPU上的作业活动区间并集，包含训练期val与存盘，非CUDA内核计时；故障结束时刻未知时保留上下界，确定停机排除。各模型墙钟不可相加当GPU-hours，也不可据并行墙钟直接判断模型速度。','',
            '| 实验 | 训练GPU活动并集(h) | 最终评估(h) | 技术检查(h) |',
            '|---|---:|---:|---:|']
    times={}
    for exp,root in [('A06',OLD),('A07',NEW)]:
        t=read(root/'gpu_hours.json');times[exp]=t
        lines.append(f"| {exp} | {t['full_training']['lower']:.3f}–{t['full_training']['upper']:.3f} | {t['full_evaluation']['lower']:.3f} | {t['smoke_all']['lower']:.3f} |")
    lines+=['','| 配置 | 训练作业活动墙钟下界–上界(h) |','|---|---:|']
    for a in ARMS:
        t=times['A06' if a in ARMS[:3] else 'A07']['per_model_training_wall_hours'][a]
        lines.append(f"| {a} | {t['lower']:.3f}–{t['upper']:.3f} |")
    lines+=['',f"本次仅加载已有epoch5权重做8记录推理与绘图，额外墙钟{gal['wall_seconds']:.3f}秒（包括CPU绘图和IO）。未新增训练或改变原始评估分数。历史绘图用时仍保留在各原报告中，不混入训练。",'',
            '## 证据与限制','',
            f"- [A06原始报告]({link(OLD/'REPORT_ZH.md')})；[A07原始报告]({link(NEW/'REPORT_ZH.md')})。",
            '- [20单元指标及来源哈希](comparison_results.json)、[新case选择规则](selection.json)、[80项预测对齐核验](gallery/review.json)、[本次汇总审查](review.json)。',
            '- 五配置具有相同评估成员、双视图输入摘要和数据指纹；全量样本与token文件哈希和总体MSE分解通过复核。原训练门禁保留，均为精确5轮共同样本流。',
            '- 单种子17，历史test查看记录保留；不将test用于新方案选择。tile ID隔离不等于已核验地理footprint零重叠。fine LST为卫星反演参考产品。',
            '- 本汇总覆盖组合码本线A06/A07及单码本基线；没有把旧数据/旧预算的A02–A05或Prithvi AE混入同条件排名。','']
    (OUTPUT/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    write(OUTPUT/'comparison_results.json',dict(status='verified',unique_models=5,evaluation_units=20,rows=rows,provenance=provenance))
    write(OUTPUT/'review.json',dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),unique_models=5,evaluation_units=20,new_case_checks=80,
        identical_input_digests=True,complete_members=True,source_sample_and_token_hashes=True,mse_decomposition=True,
        actual_visual_review=[x['path'] for x in gal['artifacts']],visual_notes='All four images inspected; masked river continuity fails for all models; low-temperature-range cases retain poor texture despite low RMSE.',
        no_new_training=True,artifact_hashes={x:sha(OUTPUT/x) for x in ['REPORT_ZH.md','comparison_results.json','selection.json','gallery/review.json']}))
    print(json.dumps(dict(status='passed',models=5,units=20,case_checks=80,report=str(OUTPUT/'REPORT_ZH.md')),ensure_ascii=False))


if __name__=='__main__':
    main()
