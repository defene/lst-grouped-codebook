"""Read-only GPU allocation-hours from A03's single-GPU stage telemetry."""
from datetime import datetime
from pathlib import Path
import argparse
import json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def timestamp(row):
    return datetime.fromisoformat(row['utc']).timestamp()


def measure_run(run, telemetry):
    run = Path(run)
    config = read(run / 'config.json')
    status = read(run / 'status.json')
    if status['status'] != 'complete':
        raise ValueError('Final GPU-hours require completed training')
    logs = [json.loads(line) for line in (run / 'train.jsonl').read_text().splitlines()]
    if logs[-1]['step'] != config['train']['steps']:
        raise ValueError('Training budget is incomplete')
    warnings = []
    phases = {}
    for stage in ('train', 'eval_clean', 'eval_noisy10'):
        indexes = [i for i, row in enumerate(telemetry)
                   if row.get('stage') == stage and row.get('active') == [run.name]]
        if not indexes:
            warnings.append(stage + ': no stage telemetry')
            phases[stage] = dict(gpu_hours=None)
            continue
        first, last = indexes[0], indexes[-1]
        rows = [telemetry[i] for i in indexes]
        contiguous = indexes == list(range(first, last + 1))
        timely = all(0 < timestamp(b)-timestamp(a) <= 60 for a, b in zip(rows, rows[1:]))
        if stage == 'train':
            end_utc = status['finished_utc']
            uninterrupted = logs[0]['step'] == 1 and all(
                b['elapsed_seconds'] > a['elapsed_seconds'] for a, b in zip(logs, logs[1:]))
            contiguous = contiguous and uninterrupted
        else:
            end_utc = telemetry[last+1]['utc'] if last+1 < len(telemetry) else None
            # Do not charge a long unsampled pause to an evaluation job.
            if end_utc and timestamp(telemetry[last+1])-timestamp(telemetry[last]) > 60:
                timely = False
        valid = contiguous and timely and end_utc is not None
        duration = datetime.fromisoformat(end_utc).timestamp()-timestamp(rows[0]) if valid else None
        if duration is not None and duration < 0:
            raise ValueError('Negative stage duration')
        if not valid:
            warnings.append(stage + ': missing boundary, resume, or telemetry gap; timing needs review')
        phases[stage] = dict(start_utc=rows[0]['utc'], end_utc=end_utc,
                            gpu_hours=duration/3600 if duration is not None else None,
                            single_uninterrupted_segment=contiguous, telemetry_timely=timely)
    periodic = {p.parent.name: read(p)['elapsed_seconds']/3600
                for p in sorted(run.glob('val_step_*/metrics.json'))}
    periodic_hours = sum(periodic.values())
    train_hours = phases['train']['gpu_hours']
    final_values = [phases[k]['gpu_hours'] for k in ('eval_clean', 'eval_noisy10')]
    final_hours = sum(final_values) if all(v is not None for v in final_values) else None
    return dict(name=run.name, modality=config['data']['modalities'][0],
        training='clean' if config['noise']['coverages'] == [0] else 'noisy10',
        gpu=read(run / 'environment.json')['gpu'], gpu_count=1,
        training_records=len({s for row in logs for s in row['sample_ids']}),
        sample_presentations=sum(row['batch_samples'] for row in logs), steps=logs[-1]['step'],
        training_stage_gpu_hours=train_hours,
        periodic_validation_gpu_hours=periodic_hours,
        training_less_recorded_validation_gpu_hours=(train_hours-periodic_hours if train_hours is not None else None),
        final_validation_gpu_hours=final_hours,
        training_and_final_validation_gpu_hours=(train_hours+final_hours if train_hours is not None and final_hours is not None else None),
        stages=phases, periodic_validation=periodic, warnings=warnings)


def summarize(output):
    output = Path(output)
    manifest = read(output / 'manifest.json')
    if manifest.get('parallel_limits') != {'conv': 1}:
        raise ValueError('This accounting requires A03 single-GPU, single-model scheduling')
    telemetry = [json.loads(line) for line in (output / 'gpu.jsonl').read_text().splitlines()]
    rows = []
    for run in sorted(output.glob('full_*_conv_*')):
        if run.is_dir() and (run / 'status.json').exists() and read(run / 'status.json')['status'] == 'complete':
            rows.append(measure_run(run, telemetry))
    values = [r['training_stage_gpu_hours'] for r in rows]
    return dict(gpu_count=1, rows=rows, completed_training_runs=len(rows),
        mean_training_gpu_hours=sum(values)/len(values) if values and all(v is not None for v in values) else None,
        definition='one allocated GPU times stage wall-clock hours, without GPU-utilization weighting',
        training_includes='model startup, data loading, optimization, periodic validation, checkpoint I/O',
        excluded='other models, queue waiting, pilot runs, final validation (reported separately), post-hoc codebook audit and reporting',
        timing_precision='approximate stage boundaries from 10-second telemetry; training end from finished_utc; two decimal hours',
        nonvalidation_caveat='subtracts recorded evaluation-loop durations only; the remainder still includes I/O, setup and checkpointing, not pure CUDA kernel time')


def markdown(result):
    lines = ['', '## 单个卷积 VQ-VAE 的 GPU-hours', '',
        '硬件为单张 NVIDIA RTX 5090。本表 GPU-hours = 1 张 GPU × 对应阶段的墙钟小时数，不乘 GPU 利用率，也不按显存比例折算。', '',
        '| 模态 | 训练输入 | 训练阶段 GPU-h | 其中定期验证 GPU-h | 扣除已记录验证后 GPU-h | 最终两种验证 GPU-h | 训练及最终验证合计 GPU-h |',
        '|---|---|---:|---:|---:|---:|---:|']
    keys = ('training_stage_gpu_hours', 'periodic_validation_gpu_hours',
            'training_less_recorded_validation_gpu_hours', 'final_validation_gpu_hours',
            'training_and_final_validation_gpu_hours')
    for row in result['rows']:
        values = ['待核查' if row[k] is None else f'{row[k]:.2f}' for k in keys]
        lines.append(f"| {row['modality'].upper()} | {'干净' if row['training']=='clean' else '10% 缺失'} | " + ' | '.join(values) + ' |')
    if result['mean_training_gpu_hours'] is not None:
        lines += ['', f"已完成的 {len(result['rows'])} 组单模型平均训练成本：**{result['mean_training_gpu_hours']:.2f} GPU-hours**。"]
    lines += ['', '每组使用 230,659 条独立训练记录，累计 692,017 次样本呈现（完整三遍加 40 条），10,813 次优化器更新；HLS 和 LST 独立训练。', '',
        '训练阶段包含启动、数据读取、优化、定期验证和检查点写入。定期验证为训练阶段的子项，不能再加一次；扣除后仍含 I/O 等开销，并非纯 GPU 内核计算时长。最终验证为干净/10% 缺失两种条件各 11,757 条记录、每条两次实现。', '',
        '不计入其他模型、排队等待、小测试、事后码本核查或报告生成。阶段边界来自约每 10 秒的采样，训练终点使用 finished_utc，按近似值保留两位小数；如有恢复或长采样缺口则标为待核查，不把停机时间计入。详见 gpu_hours.json。']
    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    result = summarize(parser.parse_args().output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
