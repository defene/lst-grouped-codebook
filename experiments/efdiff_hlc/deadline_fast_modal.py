"""Finite two-GPU-per-model pipeline for the user's one-day deadline."""
import json
from pathlib import Path
import modal
from modal_app import image, mounts, runs_volume

APP = 'efdiff-hlc-epoch1-deadline-io16-20260916'
RUN = 'hlc-synthetic-full-epoch1-ddp2-io16-20260916-v3'
ORIGINAL = 'hlc-synthetic-full-epoch1-20260916-v1'
app = modal.App(APP)


def root():
    p = Path('/runs')/RUN
    p.mkdir(parents=True, exist_ok=True)
    return p


def phase(path):
    return json.loads(path.read_text()).get('phase') if path.exists() else None


def execute(variant, smoke=False, resume=False, probe=False):
    import subprocess
    import sys
    runs_volume.reload()
    out = root()/('smoke' if smoke else 'probe' if probe else 'train')/variant
    if phase(out/'status.json') in ('smoke_passed', 'training_complete'):
        return json.loads((out/'status.json').read_text())
    if smoke or probe:
        stats = Path('/runs/hlc-synthetic-pilot-16384-512-v1/normalization.json')
    else:
        assert phase(root()/'smoke'/variant/'status.json') == 'smoke_passed'
        stats = root()/'normalization.json'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        '/code/efdiff_hlc/deadline_fast_train.py', '--variant', variant, '--output', str(out), '--stats', str(stats)]
    if smoke:
        command.append('--smoke')
        if not resume:
            command += ['--stop-after', '2']
    if probe:
        command += ['--probe', '--stop-after', '1080']
    subprocess.run(command, check=True)
    # Child processes commit checkpoints; parent reads the same mounted files.
    result = json.loads((out/'segment_result.json').read_text())
    runs_volume.commit()
    return result


@app.function(image=image, volumes=mounts, gpu='H200:2', cpu=40, memory=131072,
              timeout=2400, max_containers=2, single_use_containers=True)
def smoke(variant:str, resume:bool=False):
    return execute(variant, smoke=True, resume=resume)


@app.function(image=image, volumes=mounts, gpu='H200:2', cpu=40, memory=131072,
              timeout=2400, max_containers=1, single_use_containers=True)
def probe():
    return execute('epsilon', probe=True)


def segment(variant):
    from epoch1_train import save_json
    try:
        result = execute(variant)
        if result['phase'] == 'checkpointed':
            following = (train_x0 if variant == 'x0' else train_epsilon).spawn()
            save_json(root()/f'continuation_{variant}.json', dict(result, next_call_id=following.object_id))
            runs_volume.commit()
        else:
            assert result['phase'] == 'training_complete' and result['updates'] == 40763
            assert result['record_presentations'] == 2608772
            report.spawn()
        return result
    except Exception as exc:
        save_json(root()/f'failure_{variant}.json', {'phase':'failed', 'error':repr(exc)})
        runs_volume.commit()
        raise


@app.function(image=image, volumes=mounts, gpu='H200:2', cpu=40, memory=131072,
              timeout=21600, max_containers=1, single_use_containers=True)
def train_x0():
    return segment('x0')


@app.function(image=image, volumes=mounts, gpu='H200:2', cpu=40, memory=131072,
              timeout=21600, max_containers=1, single_use_containers=True)
def train_epsilon():
    return segment('epsilon')


@app.function(image=image, volumes=mounts, cpu=2, memory=4096, timeout=900,
              max_containers=1, single_use_containers=True)
def report():
    from epoch1_report import build
    from epoch1_train import save_json
    runs_volume.reload()
    if not all(phase(root()/'train'/v/'status.json') == 'training_complete' for v in ('x0','epsilon')):
        return {'phase':'waiting_for_other_variant'}
    result = build(root()/'train')
    save_json(root()/'pipeline.json', {'phase':'complete','comparison':result})
    runs_volume.commit()
    return result


@app.function(image=image, volumes=mounts, cpu=1, memory=4096, timeout=14400,
              max_containers=1, single_use_containers=True)
def launch():
    import time
    from epoch1_stats import merge
    from epoch1_train import save_json
    runs_volume.reload()
    out = root()
    if phase(out/'pipeline.json') in ('training','complete'):
        return json.loads((out/'pipeline.json').read_text())
    assert json.loads((Path('/runs')/ORIGINAL/'preflight.json').read_text())['status'] == 'passed'
    try:
        save_json(out/'pipeline.json', {'phase':'distributed_smoke_and_statistics',
            'deadline_utc':'2026-09-18T04:23:04Z','gpu_per_model':'H200:2'})
        runs_volume.commit()
        speed_call = probe.spawn()
        calls = [smoke.spawn(v,False) for v in ('x0','epsilon')]
        assert all(c.get()['phase'] in ('checkpointed','smoke_passed') for c in calls)
        calls = [smoke.spawn(v,True) for v in ('x0','epsilon')]
        assert all(c.get()['phase'] == 'smoke_passed' for c in calls)
        assert speed_call.get()['phase'] == 'checkpointed'
        runs_volume.reload()
        rows = [json.loads(line) for line in (out/'probe/epsilon/metrics.jsonl').read_text().splitlines()]
        start_row, end_row = rows[1], rows[-1]
        seconds_per_update = (end_row['elapsed_seconds']-start_row['elapsed_seconds'])/(end_row['step']-start_row['step'])
        from datetime import datetime, timezone
        remaining_hours = (datetime.fromisoformat('2026-09-18T04:23:04+00:00')-datetime.now(timezone.utc)).total_seconds()/3600
        projection = seconds_per_update*40763/3600
        save_json(out/'throughput_gate.json', {'seconds_per_update':seconds_per_update,
            'projected_training_hours':projection,'remaining_deadline_hours':remaining_hours,
            'reserve_hours':3,'passed':projection+3 < remaining_hours})
        runs_volume.commit()
        assert projection+3 < remaining_hours, 'Measured throughput leaves insufficient deadline margin'
        started = time.monotonic()
        while True:
            runs_volume.reload()
            paths = [Path('/runs')/ORIGINAL/'stats_shards'/f'{i:02d}.json' for i in range(32)]
            if all(p.exists() for p in paths):
                break
            if time.monotonic()-started > 7200:
                raise TimeoutError('Full training statistics not ready; no model started')
            time.sleep(30)
        stats = merge([json.loads(p.read_text()) for p in paths])
        save_json(out/'normalization.json', stats)
        runs_volume.commit()
        calls = {'x0':train_x0.spawn().object_id, 'epsilon':train_epsilon.spawn().object_id}
        save_json(out/'pipeline.json', {'phase':'training','calls':calls,
            'deadline_utc':'2026-09-18T04:23:04Z', 'gpu_per_model':'H200:2'})
        runs_volume.commit()
        return {'phase':'training','calls':calls}
    except Exception as exc:
        save_json(out/'pipeline.json', {'phase':'preparation_failed','error':repr(exc)})
        runs_volume.commit()
        raise
