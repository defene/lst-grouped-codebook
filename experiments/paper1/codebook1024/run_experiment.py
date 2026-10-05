"""A03 bounded single-GPU queue: four pilots, gate, four full runs, audited report."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from eo_data.core import sha256_file
from eo_data.locking import build_lock
from eo_denoise.engine import source_hashes
from experiments.paper1.var_vs_prithvi_ae.run_suite import json_write, read, now, command_for
from experiments.paper1.codebook1024.make_configs import check_configs


class Stopped(RuntimeError):
    pass


def gpu_snapshot():
    raw = subprocess.check_output(['nvidia-smi',
        '--query-gpu=memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    used, total, utilization = [float(x.strip()) for x in raw.strip().split(',')]
    return dict(utc=now(), used_gib=used/1024, total_gib=total/1024,
                utilization_percent=utilization)


def stop_process(process, destination, stage):
    if process.poll() is not None:
        return
    if stage == 'train' and destination.exists():
        (destination / 'STOP').write_text('checkpoint and stop\n', encoding='utf-8')
        deadline = time.monotonic() + 45
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(.2)
    if process.poll() is None:
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
        process.wait(timeout=15)


def execute(command, root, phase_output, identity, state, destination, stage, logfile):
    if (root / 'STOP').exists() or (phase_output / 'STOP').exists():
        raise Stopped('Explicit STOP marker; remove only before an authorized resume')
    if source_hashes() != identity['source_hashes']:
        raise ValueError('Framework changed while A03 was queued')
    if {p.name: sha256_file(p) for p in check_configs(state['phase'])} != identity['config_hashes']:
        raise ValueError('Configuration changed while A03 was queued')
    before = gpu_snapshot()
    if before['total_gib'] - before['used_gib'] < 1:
        raise RuntimeError('Device free VRAM below 1 GiB before launch')
    process = None
    with logfile.open('a', encoding='utf-8') as stream:
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            last_gpu = 0
            while True:
                code = process.poll()
                snapshot = dict(state, status='running', supervisor_pid=os.getpid(),
                    updated_utc=now(), active=[dict(name=destination.name, stage=stage, pid=process.pid)])
                json_write(root / 'status.json', snapshot)
                if stage in ('train', 'eval_clean', 'eval_noisy10'):
                    json_write(phase_output / 'status.json', snapshot)
                if time.monotonic() - last_gpu >= 10:
                    gpu = dict(gpu_snapshot(), active=[destination.name], stage=stage, phase=state['phase'])
                    for folder in (root, phase_output):
                        with (folder / 'gpu.jsonl').open('a', encoding='utf-8') as f:
                            f.write(json.dumps(gpu) + '\n')
                    if gpu['total_gib'] - gpu['used_gib'] < 1:
                        raise RuntimeError('Device free VRAM below 1 GiB; saving checkpoint and stopping')
                    last_gpu = time.monotonic()
                if code is not None:
                    if code:
                        raise RuntimeError(f'{destination.name}/{stage} exited {code}; see {logfile}')
                    return
                if (root / 'STOP').exists() or (phase_output / 'STOP').exists():
                    raise Stopped('Explicit STOP marker')
                time.sleep(2)
        except BaseException:
            if process is not None:
                stop_process(process, destination, stage)
            raise


def run_phase(root, phase, frozen_sources):
    output = root / phase
    output.mkdir(exist_ok=True)
    paths = check_configs(phase)
    identity = dict(config_hashes={p.name: sha256_file(p) for p in paths}, source_hashes=frozen_sources)
    if phase == 'full':
        gate = read(root / 'pilot/gate.json')
        if gate['status'] != 'passed' or gate['source_hashes'] != frozen_sources:
            raise ValueError('Full training requires a successful current-source A03 pilot gate')
        if gate['configs'] != {p.name: sha256_file(p) for p in check_configs('pilot')}:
            raise ValueError('Pilot configurations changed after the gate')
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if any(manifest[k] != identity[k] for k in identity):
            raise ValueError('Cannot resume with changed model source or configuration')
    else:
        json_write(manifest_path, dict(identity, created_utc=now(), phase=phase,
            experiment='A03', pretrained_weights=False, codebook_size=1024,
            baseline_codebook_size=4096, only_configuration_change='model.codebook_size',
            steps_per_run=128 if phase == 'pilot' else 10813,
            effective_batch=16 if phase == 'pilot' else 64,
            train_pairs=64 if phase == 'pilot' else 230659,
            final_val_pairs=64 if phase == 'pilot' else 11757,
            parallel_limits={'conv': 1}, memory_limit_gib=20,
            checkpoint_selection='fixed final last.pt; no baseline or pilot weights loaded'))
    completed = []
    for index, path in enumerate(paths):
        job = dict(name=path.stem, config=path)
        destination = output / path.stem
        state = dict(phase=phase, completed=completed.copy(), pending=[p.stem for p in paths[index+1:]])
        for stage in ('train', 'eval_clean', 'eval_noisy10'):
            command = command_for(job, stage, output, phase)
            if command is not None:
                execute(command, root, output, identity, state, destination, stage,
                        output / (path.stem + '.stdout.log'))
            if stage == 'train':
                status = read(destination / 'status.json')
                if status['status'] != 'complete' or status['step'] != read(path)['train']['steps']:
                    raise RuntimeError(f'{path.stem}: training budget incomplete')
            else:
                metric = read(destination / stage / 'metrics.json')
                if metric['step'] != read(path)['train']['steps'] or metric['evaluated_pairs'] != (64 if phase == 'pilot' else 11757):
                    raise ValueError(f'{path.stem}/{stage}: incomplete evaluation')
        completed.append(path.stem)
    state = dict(status='training_and_evaluation_complete', phase=phase, completed=completed,
                 active=[], pending=[], updated_utc=now(), supervisor_pid=os.getpid())
    json_write(output / 'status.json', state)
    json_write(root / 'status.json', state)
    return identity, state


def main(output):
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    frozen_sources = source_hashes()
    baseline = ROOT / 'runs/paper1/var_vs_prithvi_ae_full/manifest.json'
    if read(baseline)['source_hashes'] != frozen_sources:
        raise ValueError('A03 must use the unchanged A02 model source')
    with build_lock(root):
        try:
            for phase in ('pilot', 'full'):
                identity, state = run_phase(root, phase, frozen_sources)
                execute([sys.executable, str(HERE / 'validate_and_report.py'), '--phase', phase,
                         '--output', str(root / phase)], root, root / phase, identity, state,
                        root / phase, 'audit', root / phase / 'audit.stdout.log')
                # The stage monitor writes running status; restore phase completion.
                json_write(root / phase / 'status.json', state)
            execute([sys.executable, str(ROOT / 'experiments/paper1/var_vs_prithvi_ae/inspect_codebook.py'),
                     '--output', str(root / 'full')], root, root / 'full', identity, state,
                    root / 'full', 'codebook_audit', root / 'full/codebook.stdout.log')
            execute([sys.executable, str(HERE / 'validate_and_report.py'), '--phase', 'full',
                     '--output', str(root / 'full'), '--report-only'], root, root / 'full', identity, state,
                    root / 'full', 'report', root / 'full/report.stdout.log')
            json_write(root / 'full/status.json', dict(state, report=str(root / 'full/report/RESULTS_ZH.md')))
            json_write(root / 'status.json', dict(state, status='complete', experiment='A03',
                updated_utc=now(), report=str(root / 'full/report/RESULTS_ZH.md')))
        except BaseException as error:
            stopped = isinstance(error, Stopped)
            previous = read(root / 'status.json') if (root / 'status.json').exists() else {}
            result = dict(previous, status='stopped_at_checkpoints' if stopped else 'failed',
                          error=repr(error), updated_utc=now(), active=[])
            json_write(root / 'status.json', result)
            if not stopped:
                json_write(root / 'failure.json', result)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/paper1/codebook1024')
    main(parser.parse_args().output)
