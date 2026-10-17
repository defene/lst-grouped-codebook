"""Detached local supervisor for the already-authorized public dsaaf upload.

Retains resumable receipts and retries failed child commands up to three times.
Does not restart Windows, change account settings, or modify power configuration.
"""
import ctypes
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/huggingface_upload_20261005'


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}


def save(value):
    path = OUT / 'supervisor_status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hlc-only', action='store_true')
    args = parser.parse_args()
    import msvcrt
    lock = (OUT / 'supervisor.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    from huggingface_hub import HfApi
    if HfApi().whoami()['name'] != 'dsaaf':
        raise ValueError('Expected account dsaaf')
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    jobs = {
        'hlp': {'args': ['upload_huggingface.py', '--owner', 'dsaaf', '--visibility', 'public', '--dataset', 'HLP'],
                'done': lambda: read(OUT / 'hlp_upload_status.json').get('phase') == 'complete'},
        'pack_hlc': {'args': ['prepare_huggingface.py', '--dataset', 'HLC'],
                     'done': lambda: read(OUT / 'hlc_release_receipt.json').get('status') == 'packaged'},
        'upload_hlc': {'args': ['upload_hlc_incremental.py'],
                      'done': lambda: read(OUT / 'hlc_upload_status.json').get('phase') == 'complete'},
    }
    if args.hlc_only:
        jobs.pop('hlp')
    for job in jobs.values():
        job.update(process=None, attempts=0, next_retry=0, status='pending')
    environment = os.environ.copy()
    environment['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    environment['HF_HUB_DISABLE_XET'] = '1'  # HTTPS/LFS fallback after native Xet timed out and hung.
    try:
        while True:
            for name, job in jobs.items():
                process = job['process']
                if process is not None:
                    code = process.poll()
                    if code is None:
                        timed_out = False
                        if name == 'upload_hlc':
                            last_progress = max(job['started_unix'], read(OUT / 'hlc_incremental_status.json').get('updated_unix', 0))
                            timed_out = time.time() - last_progress > 1200
                            dependency_failed = jobs['pack_hlc']['status'] == 'failed'
                            if timed_out or dependency_failed:
                                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
                                process.wait(timeout=30)
                                job['failure_reason'] = 'No completed batch for 20 minutes' if timed_out else 'Local packer failed'
                                code = process.returncode
                        if code is None:
                            continue
                    job['stream'].close()
                    job['process'] = None
                    job['last_exit_code'] = code
                    job['status'] = 'complete' if job['done']() else 'retry_wait'
                    job['next_retry'] = time.time() + 60
                    if job['status'] != 'complete' and job['attempts'] >= 3:
                        job['status'] = 'failed'
                if job['done']():
                    job['status'] = 'complete'
                if job['status'] in ('complete', 'failed') or time.time() < job['next_retry']:
                    continue
                if name == 'upload_hlc' and jobs['pack_hlc']['status'] == 'failed':
                    job['status'] = 'failed'
                    continue
                job['attempts'] += 1
                log = OUT / f'{name}_resume_{int(time.time())}_{job["attempts"]}.log'
                stream = log.open('wb')
                args = [sys.executable, '-u', str(Path(__file__).with_name(job['args'][0])), *job['args'][1:]]
                job['process'] = subprocess.Popen(args, cwd=ROOT, env=environment, stdout=stream,
                                                   stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                job.update(stream=stream, log=str(log), status='running', started_unix=time.time())
            phase = 'complete' if all(j['status'] == 'complete' for j in jobs.values()) else 'running'
            if all(j['status'] in ('complete', 'failed') for j in jobs.values()) and phase != 'complete':
                phase = 'failed'
            save({'phase': phase, 'pid': os.getpid(), 'updated_unix': time.time(),
                  'jobs': {name: {'status': j['status'], 'attempts': j['attempts'],
                                   'pid': j['process'].pid if j['process'] else None,
                                   'log': j.get('log'), 'last_exit_code': j.get('last_exit_code')}
                           for name, j in jobs.items()}})
            if phase != 'running':
                return
            time.sleep(15)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__ == '__main__':
    main()
