"""Two bounded GPU lanes, smoke gate, 2000-step pilots, automatic reporting."""
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eo_data.locking import build_lock
from eo_data.core import sha256_file
from experiments.paper1.var_vs_prithvi_ae.run_suite import read, now, json_write
from experiments.paper1.codebook1024.run_experiment import gpu_snapshot, stop_process
from experiments.paper1.lst_residual.runtime import source_hashes
HERE = Path(__file__).resolve().parent


def phase(output, name, expected_sources):
    root = output/name
    root.mkdir(exist_ok=True)
    paths = sorted((HERE/'configs'/name).glob('*.json'))
    assert len(paths) == 4
    identity = dict(source_hashes=expected_sources, configs={p.name: sha256_file(p) for p in paths},
                    parallel_slots=2, per_process_allocator_limit_gib=12, minimum_global_free_gib=2)
    if (root/'manifest.json').exists():
        assert read(root/'manifest.json') == identity
    else:
        json_write(root/'manifest.json', identity)
    pending, active, completed = list(paths), [], []
    stages = ['train', 'eval_clean', 'eval_noisy10']

    def launch(path, stage):
        destination = root/path.stem
        if stage == 'train':
            command = [sys.executable, str(HERE/'run.py'), 'train', '--config', str(path), '--output', str(destination)]
        else:
            command = [sys.executable, str(HERE/'run.py'), 'evaluate', '--checkpoint', str(destination/'last.pt'),
                       '--output', str(destination/stage), '--coverage', '0' if stage == 'eval_clean' else '.1']
        log = (root/(path.stem+'.stdout.log')).open('a', encoding='utf-8')
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        entry = dict(path=path, stage=stage, process=process, log=log, started=now())
        with (root/'stages.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(event='start', name=path.stem, stage=stage, pid=process.pid, utc=entry['started']))+'\n')
        return entry

    def already_done(path, stage):
        destination = root/path.stem
        if stage == 'train':
            p = destination/'status.json'
            return p.exists() and read(p).get('status') == 'complete' and read(p)['step'] == read(path)['train']['steps']
        p = destination/stage/'metrics.json'
        return p.exists() and read(p).get('step') == read(path)['train']['steps']

    def next_stage(path):
        return next((s for s in stages if not already_done(path, s)), None)

    last_gpu = 0
    try:
        while pending or active:
            if source_hashes() != expected_sources or any(sha256_file(p) != identity['configs'][p.name] for p in paths):
                raise RuntimeError('Frozen experiment source/config changed')
            if (output/'STOP').exists():
                raise RuntimeError('Explicit experiment STOP marker')
            while pending and len(active) < 2:
                p = pending.pop(0)
                stage = next_stage(p)
                if stage:
                    active.append(launch(p, stage))
                else:
                    completed.append(p.stem)
            for entry in list(active):
                code = entry['process'].poll()
                if code is None:
                    continue
                entry['log'].close()
                with (root/'stages.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(dict(event='end', name=entry['path'].stem, stage=entry['stage'],
                                            exit_code=code, utc=now()))+'\n')
                active.remove(entry)
                if code:
                    raise RuntimeError(f"{entry['path'].stem}/{entry['stage']} failed with exit {code}")
                pending.insert(0, entry['path'])
            state = dict(status='running', phase=name, supervisor_pid=os.getpid(), updated_utc=now(), completed=completed,
                         active=[dict(name=e['path'].stem, stage=e['stage'], pid=e['process'].pid) for e in active],
                         pending=[p.stem for p in pending])
            json_write(output/'status.json', state)
            if time.monotonic()-last_gpu >= 10:
                gpu = dict(gpu_snapshot(), phase=name, active=state['active'])
                with (output/'gpu.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(gpu)+'\n')
                if gpu['total_gib']-gpu['used_gib'] < 2:
                    raise RuntimeError('Global GPU free memory below 2 GiB; checkpoint and stop')
                last_gpu = time.monotonic()
            time.sleep(2)
    finally:
        for e in active:
            stop_process(e['process'], root/e['path'].stem, e['stage'])
            e['log'].close()
    for p in paths:
        c = read(p)
        rows = [json.loads(line) for line in (root/p.stem/'train.jsonl').read_text(encoding='utf-8').splitlines()]
        assert rows[-1]['step'] == c['train']['steps'] and all(math.isfinite(r['total']) for r in rows)
        for condition in ('eval_clean', 'eval_noisy10'):
            metric = read(root/p.stem/condition/'metrics.json')
            assert metric['evaluated_pairs'] == c['data']['val_limit']
            assert metric['checkpoint_sha256'] == sha256_file(root/p.stem/'last.pt')
    # Every arm must consume exactly the same record stream and optimizer steps.
    reference = [json.loads(s)['sample_ids'] for s in (root/paths[0].stem/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    for p in paths[1:]:
        assert [json.loads(s)['sample_ids'] for s in (root/p.stem/'train.jsonl').read_text(encoding='utf-8').splitlines()] == reference
    json_write(root/'gate.json', dict(status='passed', phase=name, source_hashes=expected_sources,
        configs=identity['configs'], completed=completed, same_training_stream_verified=True, updated_utc=now()))


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    with build_lock(output):
        try:
            frozen = source_hashes()
            for name in ('smoke', 'pilot'):
                phase(output, name, frozen)
            subprocess.run([sys.executable, str(HERE/'report.py'), '--output', str(output)], cwd=ROOT, check=True)
            json_write(output/'status.json', dict(status='complete', phase='report', updated_utc=now(),
                report=str(output/'report/REPORT_ZH.md'), active=[], completed_models=4))
        except BaseException as error:
            previous = read(output/'status.json') if (output/'status.json').exists() else {}
            json_write(output/'status.json', dict(previous, status='failed', error=repr(error), active=[], updated_utc=now()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'runs/paper1/lst_residual_20260913')
    main(parser.parse_args().output.resolve())
