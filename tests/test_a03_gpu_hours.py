"""Check GPU-hour boundaries, validation accounting and interrupted-run safeguards."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from experiments.paper1.codebook1024.gpu_hours import measure_run


def fixture_run(tmp_path):
    run = tmp_path / 'full_hls_conv_clean_seed17'
    run.mkdir()
    start = datetime(2026, 9, 13, tzinfo=timezone.utc)
    def utc(seconds):
        return (start + timedelta(seconds=seconds)).isoformat()
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')
    write(run / 'config.json', dict(train={'steps': 2}, data={'modalities': ['hls']}, noise={'coverages': [0]}))
    write(run / 'status.json', dict(status='complete', finished_utc=utc(100)))
    write(run / 'environment.json', {'gpu': 'RTX 5090'})
    logs = [dict(step=i, elapsed_seconds=i*40, batch_samples=1, sample_ids=[str(i)]) for i in (1, 2)]
    (run / 'train.jsonl').write_text('\n'.join(json.dumps(x) for x in logs))
    validation = run / 'val_step_0000002'
    validation.mkdir()
    write(validation / 'metrics.json', {'elapsed_seconds': 10})
    rows = [dict(utc=utc(t), active=[run.name], stage=stage, utilization_percent=0)
            for stage, times in [('train', range(0, 101, 10)), ('eval_clean', [110, 120]), ('eval_noisy10', [130, 140])]
            for t in times]
    rows.append(dict(utc=utc(150), active=['next_run'], stage='train', utilization_percent=100))
    return run, rows, utc


def test_wall_hours_and_validation_subitem(tmp_path):
    run, rows, _ = fixture_run(tmp_path)
    result = measure_run(run, rows)
    assert result['training_stage_gpu_hours'] == pytest.approx(100/3600)
    assert result['periodic_validation_gpu_hours'] == pytest.approx(10/3600)
    assert result['training_less_recorded_validation_gpu_hours'] == pytest.approx(90/3600)
    assert result['final_validation_gpu_hours'] == pytest.approx(40/3600)
    assert result['training_and_final_validation_gpu_hours'] == pytest.approx(140/3600)
    assert not result['warnings']  # Utilization 0 must not turn allocation time into zero.


def test_telemetry_gap_requires_review(tmp_path):
    run, rows, utc = fixture_run(tmp_path)
    rows = [x for x in rows if x['stage'] != 'train' or x['utc'] in (utc(0), utc(100), utc(150))]
    result = measure_run(run, rows)
    assert result['training_stage_gpu_hours'] is None
    assert result['training_and_final_validation_gpu_hours'] is None
    assert result['warnings']


def test_resumed_elapsed_clock_is_not_wall_time(tmp_path):
    run, rows, _ = fixture_run(tmp_path)
    logs = [json.loads(x) for x in (run / 'train.jsonl').read_text().splitlines()]
    logs[1]['elapsed_seconds'] = 5
    (run / 'train.jsonl').write_text('\n'.join(json.dumps(x) for x in logs))
    result = measure_run(run, rows)
    assert result['training_stage_gpu_hours'] is None
    assert result['warnings']
