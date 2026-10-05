"""Two GPU jobs, exact-epoch smoke/resume gate, then four frozen full runs."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import math
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eo_data.core import sha256_file, json_write as _json_write
from eo_data.locking import build_lock
from experiments.paper1.codebook1024.run_experiment import gpu_snapshot, stop_process
from experiments.paper1.lst_v2_5epoch.prepare import OUTPUT, HERE
from experiments.paper1.lst_v2_5epoch.train import source_hashes


def json_write(path, value):
    """Retry transient Windows file-sharing failures in supervisor metadata only."""
    for attempt in range(12):
        try:
            return _json_write(path, value)
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.05 * 2**attempt, .5))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def append(path, value):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value)+'\n')


def run_queue(phase, stage, identity):
    paths = sorted((HERE/'configs'/phase).glob('*.json'))
    root = OUTPUT/phase
    root.mkdir(exist_ok=True)
    pending, active = paths[:], []
    last_gpu = 0

    def done(path):
        dest = root/path.stem
        c = read(path)
        if stage == 'partial':
            return (dest/'last.pt').exists() and (dest/'status.json').exists() and read(dest/'status.json').get('step', 0) >= 2
        if stage == 'train':
            return (dest/'status.json').exists() and read(dest/'status.json').get('status') == 'complete' and read(dest/'status.json')['step'] == c['train']['steps']
        if stage == 'evaluate':
            return all((dest/f'final_{s}_{n}/metrics.json').exists() for s in ('val','test') for n in ('clean','noisy10'))

    try:
        while pending or active:
            assert source_hashes() == identity['sources'], 'Frozen model/training source changed'
            assert {str(p.relative_to(HERE)):sha256_file(p) for p in (HERE/'configs').rglob('*.json')} == identity['configs']
            if (OUTPUT/'STOP').exists():
                raise RuntimeError('Explicit suite STOP')
            while pending and len(active)<2:
                path = pending.pop(0)
                if done(path):
                    continue
                dest = root/path.stem
                if (dest/'STOP').exists():
                    raise RuntimeError(f'Run STOP remains: {dest}')
                command = [sys.executable, str(HERE/'run.py'), 'train' if stage=='partial' else stage,
                           '--config', str(path), '--output', str(dest)]
                if stage=='partial':
                    command += ['--stop-after','2']
                log = (root/(path.stem+'.log')).open('a', encoding='utf-8')
                p = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                e = {'process':p,'path':path,'log':log,'destination':dest}
                active.append(e)
                append(OUTPUT/'stages.jsonl', {'event':'start','phase':phase,'stage':stage,'name':path.stem,'pid':p.pid,'utc':now()})
            for e in active[:]:
                code = e['process'].poll()
                if code is not None:
                    e['log'].close()
                    active.remove(e)
                    append(OUTPUT/'stages.jsonl', {'event':'end','phase':phase,'stage':stage,'name':e['path'].stem,'exit_code':code,'utc':now()})
                    if code or not done(e['path']):
                        raise RuntimeError(f"{e['path'].stem}/{stage} incomplete (exit {code})")
            state = {'status':'running','phase':phase,'stage':stage,'utc':now(),'pid':os.getpid(),
                'active':[{'name':e['path'].stem,'pid':e['process'].pid} for e in active],
                'pending':[p.stem for p in pending]}
            json_write(OUTPUT/'status.json',state)
            if time.monotonic()-last_gpu>10:
                gpu = dict(gpu_snapshot(), phase=phase, stage=stage, active=state['active'])
                append(OUTPUT/'gpu.jsonl', gpu)
                if gpu['total_gib']-gpu['used_gib']<2:
                    raise RuntimeError('Less than 2 GiB GPU free memory; checkpoint and stop')
                last_gpu = time.monotonic()
            time.sleep(2)
    finally:
        for e in active:
            stop_process(e['process'],e['destination'],'train' if stage in ('train','partial') else 'evaluate')
            e['log'].close()
            append(OUTPUT/'stages.jsonl', {'event':'end','phase':phase,'stage':stage,
                'name':e['path'].stem,'exit_code':e['process'].poll(),'utc':now(),
                'reason':'supervisor_cleanup_stop'})


def gate(phase, identity):
    configs = sorted((HERE/'configs'/phase).glob('*.json'))
    reference = None
    fingerprints = []
    for path in configs:
        c = read(path)
        dest = OUTPUT/phase/path.stem
        lines = [json.loads(s) for s in (dest/'train.jsonl').read_text(encoding='utf-8').splitlines()]
        n = c['data']['train_limit'] or 218975
        epochs = c['experiment']['epochs']
        assert len(lines) == c['train']['steps']
        assert [r['step'] for r in lines] == list(range(1,c['train']['steps']+1))
        assert all(math.isfinite(r['total']) for r in lines)
        stream = [r['sample_ids'] for r in lines]
        if reference is None:
            reference = stream
        else:
            assert stream == reference, 'Different training stream across models'
        import numpy as np
        members = set(np.load(dest/'train_record_ids.npy').tolist())
        assert len(members) == n
        for epoch in range(epochs):
            entries = [r for r in lines if r['epoch_index']==epoch]
            ids = [int(s.rsplit(':',1)[1]) for r in entries for s in r['sample_ids']]
            assert len(ids) == n and set(ids) == members
            assert entries[-1]['batch_samples'] == n%64
        assert lines[-1]['records_presented'] == n*epochs
        assert read(dest/'status.json')['validated_epoch'] == epochs
        fingerprints.append(read(dest/'data_fingerprint.json'))
    assert all(p==fingerprints[0] for p in fingerprints)
    json_write(OUTPUT/phase/'gate.json',{'status':'passed','phase':phase,'utc':now(),
        'source_hashes':identity['sources'],'same_record_stream':True,'exact_epochs':True,
        'tail_records':n%64,'records_per_model':n*epochs,'updates_per_model':c['train']['steps']})


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with build_lock(OUTPUT):
        try:
            assert read(OUTPUT/'data_preflight.json')['status']=='passed'
            identity={'sources':source_hashes(),'configs':{str(p.relative_to(HERE)):sha256_file(p) for p in (HERE/'configs').rglob('*.json')},
                      'parallel_jobs':2,'per_process_allocator_gib':12,'minimum_free_gib':2}
            if (OUTPUT/'manifest.json').exists():
                assert read(OUTPUT/'manifest.json')==identity
            else:
                json_write(OUTPUT/'manifest.json',identity)
            run_queue('smoke','partial',identity)
            run_queue('smoke','train',identity)
            run_queue('smoke','evaluate',identity)
            gate('smoke',identity)
            # All four final tests wait until the complete five-epoch training stage finishes.
            run_queue('full','train',identity)
            gate('full',identity)
            run_queue('full','evaluate',identity)
            subprocess.run([sys.executable,'-m','experiments.paper1.lst_v2_5epoch.report'],cwd=ROOT,check=True)
            json_write(OUTPUT/'status.json',{'status':'complete','phase':'report','utc':now(),'completed_models':4})
        except BaseException as e:
            json_write(OUTPUT/'status.json',{'status':'failed','error':repr(e),'utc':now(),'pid':os.getpid()})
            raise


if __name__=='__main__':
    main()
