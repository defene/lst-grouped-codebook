"""Prepare K512 configs, then wait for the original queue before GPU preflight."""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from eo_data.locking import build_lock
from experiments.paper1.lst_ema_full_20261014.config import PROJECT, file_digest, validate, write_json
from experiments.paper1.lst_ema_full_20261014.runtime import protocol_digest, source_hashes, utc


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def own_sources():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [PROJECT/'tests/test_lst_ema_k512_20261015.py']
    return {p.relative_to(PROJECT).as_posix(): file_digest(p) for p in paths}


def prepare(root, parent, tests):
    if root.exists():
        raise FileExistsError('Use a fresh output directory')
    suite = ET.parse(tests).getroot()
    if len(list(suite.iter('testcase'))) < 28 or any(list(suite.iter(t)) for t in ('failure','error','skipped')):
        raise ValueError('The complete CPU checks must pass before scheduling')
    old = read(parent/'queue.json')
    if old['source_hashes'] != source_hashes():
        raise ValueError('Frozen runtime changed')
    root.mkdir(parents=True)
    jobs = []
    for job in old['jobs']:
        c = read(job['config'])
        grouped = c['model']['kind'] == 'grouped_g8k256'
        name = ('grouped_g8k512' if grouped else 'vq_k512') + '_G_' + c['ema_method']
        c.update(version='lst-ema-k512-full-20261015', name=name)
        c['model'].update(codebook_size=512, bits_per_token=9)
        c['tracking']['run_id'] = uuid.uuid4().hex
        c['execution'] = dict(enabled=False, preflight_report=str(root/'preflight'/f'{name}.json'))
        assert validate(c)['ok'], validate(c)
        path = root/'configs'/f'{name}.json'
        write_json(path, c)
        tokens = 8*sum(s*s for s in c['model']['scales']) if grouped else c['model']['grid']**2
        jobs.append(dict(name=name, config=str(path), parent_config=job['config'],
                         parent_config_sha256=file_digest(job['config']),
                         protocol_sha256=protocol_digest(c), codebook_size_per_group=512,
                         groups=8 if grouped else 1, nominal_index_bits=tokens*9))
    write_json(root/'queue.json', dict(status='prepared_not_launched', jobs=jobs,
        scope='G; ordinary VQ K512 and Grouped 8 x K512; baseline, EMA, EMA maintenance; 5 full epochs',
        total_record_presentations=6*1094875,
        note='Legacy internal grouped_g8k256 routes the generic Grouped implementation; actual K is 512.'))
    manifest = dict(parent=str(parent), parent_queue_sha256=file_digest(parent/'queue.json'),
        runtime_sources=source_hashes(), extension_sources=own_sources(),
        protocols={j['name']:j['protocol_sha256'] for j in jobs}, created_utc=utc(),
        cpu_tests=str(tests), cpu_tests_sha256=file_digest(tests))
    write_json(root/'manifest.json', manifest)
    with zipfile.ZipFile(root/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for rel in {**manifest['runtime_sources'], **manifest['extension_sources']}:
            z.write(PROJECT/rel, rel)
    (root/'README.txt').write_text(
        'K512 formal local extension, G representation only.\n'
        'Six jobs: VQ K512 / Grouped 8 x K512, each baseline / EMA / EMA+K-means+dead reset.\n'
        '5 full HLP v2 epochs, 218975 train records/epoch, effective batch64, micro8.\n'
        'Per model: 1094875 record presentations, 17110 updates; final batch31 each epoch.\n'
        'Each epoch: full val11684 records, clean and pixel10, two fixed views each.\n'
        'Select minimum pixel10 valid-pixel-weighted Celsius RMSE; retain every epoch. No test selection.\n'
        'Same data, sample-order rules, LR, losses, EMA settings and latent geometry as parent.\n'
        'Fresh initialization; no pilot or parent weights reused. No latent RMS normalization added.\n'
        'Grouped internal kind retains legacy grouped_g8k256 routing; actual book size512 in all8 groups.\n'
        'Nominal index bits/image: VQ2304 (was2560), Grouped48960 (was43520); plus mean32bits.\n'
        'This is a codebook-size ablation, not an equal-bit-rate comparison.\n'
        'Scheduler waits for all6 parent jobs to finish successfully, audits their validation receipts,\n'
        'then runs real CUDA full/tail preflight for all6 K512 models before enabling training.\n'
        'Parent failure/stopping keeps this queue waiting. STOP in this root stops scheduler/training.\n'
        'Keep this computer running. Restart launch.ps1 to resume after interruption.\n'
        f'Parent: {parent}\n', encoding='utf-8')


def verify(root, tests):
    m = read(root/'manifest.json')
    if m['runtime_sources'] != source_hashes() or m['extension_sources'] != own_sources():
        raise ValueError('Frozen source changed')
    if file_digest(Path(m['parent'])/'queue.json') != m['parent_queue_sha256']:
        raise ValueError('Parent queue changed')
    queue = read(root/'queue.json')
    if {j['name']:protocol_digest(read(j['config'])) for j in queue['jobs']} != m['protocols']:
        raise ValueError('K512 protocol changed')
    if file_digest(tests) != m['cpu_tests_sha256']:
        raise ValueError('CPU test report changed')
    return m


def parent_ready(parent):
    status_path = parent/'queue_status.json'
    if not status_path.exists():
        return False
    status = read(status_path)
    jobs = read(parent/'queue.json')['jobs']
    return status.get('status') == 'completed' and status.get('completed') == [j['name'] for j in jobs]


def audit_parent(parent):
    from experiments.paper1.lst_ema_full_20261014.epoch_validation import reconcile_selection
    # Wait for the parent queue to release its writer lock before auditing completion.
    with build_lock(parent):
        if not parent_ready(parent):
            raise ValueError('Parent not complete')
        for job in read(parent/'queue.json')['jobs']:
            run = parent/'runs'/job['name']
            state = read(run/'training_complete.json')
            if state['records'] != 1094875 or state['updates'] != 17110:
                raise ValueError('Parent budget incomplete')
            reconcile_selection(read(job['config']), run, 5)


def wait_and_run(root, tests):
    def status(phase, **extra):
        write_json(root/'scheduler_status.json', dict(status=phase, pid=os.getpid(), utc=utc(), **extra))

    def stopped():
        if (root/'STOP').exists():
            status('stopped')
            return True
        return False

    def child(module, args, phase):
        status(phase)
        with (root/f'{phase}.log').open('a', encoding='utf-8') as log:
            p = subprocess.Popen([sys.executable, '-u', '-m', module, *args], cwd=PROJECT,
                                 stdout=log, stderr=subprocess.STDOUT)
            status(phase, child_pid=p.pid)
            code = p.wait()
        if code:
            raise RuntimeError(f'{phase} exited with code {code}; see {phase}.log')

    with build_lock(root/'scheduler'):
        try:
            m = verify(root, tests)
            parent = Path(m['parent'])
            while not parent_ready(parent):
                if stopped():
                    return
                status('waiting_for_parent', parent=str(parent),
                       parent_status=read(parent/'queue_status.json') if (parent/'queue_status.json').exists() else None)
                time.sleep(30)
            if stopped():
                return
            verify(root, tests)
            audit_parent(parent)
            if read(root/'queue.json')['status'] != 'ready':
                child('experiments.paper1.lst_ema_full_20261014.prepare',
                      ['preflight', '--output', str(root), '--tests', str(tests)], 'preflight')
            if stopped():
                return
            verify(root, tests)
            child('experiments.paper1.lst_ema_full_20261014.queue', ['--output', str(root)], 'training')
            if read(root/'queue_status.json')['status'] != 'completed':
                raise RuntimeError('Queue did not complete')
            status('completed')
        except Exception as exc:
            status('failed', error=str(exc))
            raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'wait'])
    ap.add_argument('--output', required=True)
    ap.add_argument('--parent')
    ap.add_argument('--tests')
    args = ap.parse_args()
    root = Path(args.output)
    if args.action == 'prepare':
        prepare(root, Path(args.parent), Path(args.tests))
    else:
        wait_and_run(root, Path(args.tests))
