"""Sequential single-GPU A06 run; gated smoke, full training, evaluation, report."""
import json
import os
import subprocess
import sys
import time
from eo_data.locking import build_lock
from eo_data.core import json_write as raw_write, sha256_file
from experiments.paper1.codebook1024.run_experiment import gpu_snapshot, stop_process
from .prepare import HERE, ROOT, OUTPUT, NAME, read
from .train import source_hashes, utc


def write(path,obj):
    for i in range(12):
        try:return raw_write(path,obj)
        except PermissionError:
            if i==11:raise
            time.sleep(.25)


def append(path,obj):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(obj)+'\n')


def execute(phase,stage,command,dest=None):
    manifest=read(OUTPUT/'manifest.json')
    assert source_hashes()==manifest['sources']
    assert {str(p.relative_to(HERE)):sha256_file(p) for p in (HERE/'configs').rglob('*.json')}==manifest['configs']
    if (OUTPUT/'STOP').exists():raise RuntimeError('Explicit STOP')
    dest=dest or OUTPUT/phase/NAME.replace('full_',phase+'_',1)
    process=None
    with (OUTPUT/f'{phase}_{stage}.log').open('a',encoding='utf-8') as log:
        try:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            append(OUTPUT/'stages.jsonl',dict(event='start',phase=phase,stage=stage,pid=process.pid,utc=utc()))
            last=0
            while process.poll() is None:
                write(OUTPUT/'status.json',dict(status='running',phase=phase,stage=stage,utc=utc(),
                    pid=os.getpid(),child_pid=process.pid,destination=str(dest)))
                if time.monotonic()-last>10:
                    gpu=gpu_snapshot()
                    append(OUTPUT/'gpu.jsonl',dict(gpu,phase=phase,stage=stage,pid=process.pid))
                    if gpu['total_gib']-gpu['used_gib']<2:raise RuntimeError('Less than 2 GiB free GPU memory')
                    last=time.monotonic()
                if (OUTPUT/'STOP').exists():raise RuntimeError('Explicit STOP')
                time.sleep(2)
            if process.returncode:raise RuntimeError(f'{phase}/{stage}: exit {process.returncode}; inspect log')
        finally:
            if process is not None:
                if process.poll() is None:stop_process(process,dest,'train' if stage in ('partial','train') else 'evaluate')
                append(OUTPUT/'stages.jsonl',dict(event='end',phase=phase,stage=stage,pid=process.pid,
                    exit_code=process.poll(),utc=utc()))


def run(phase,stage,folder=None):
    name=NAME.replace('full_',phase+'_',1)
    dest=OUTPUT/(folder or phase)/name
    if stage=='train' and (dest/'status.json').exists() and read(dest/'status.json').get('status')=='complete':return
    if stage=='partial' and (dest/'last.pt').exists():return
    if stage=='evaluate' and all((dest/f'final_{s}_{c}/metrics.json').exists() for s in ('val','test') for c in ('clean','noisy10')):return
    args=[sys.executable,'-m','experiments.paper1.lst_residual_scale.run',
        'train' if stage=='partial' else stage,'--config',str(HERE/'configs'/phase/f'{name}.json'),'--output',str(dest)]
    if stage=='partial':args+=['--stop-after','2']
    execute(folder or phase,stage,args,dest)
    if stage=='train':assert read(dest/'status.json')['status']=='complete'


def module(name,*args):
    return [sys.executable,'-m','experiments.paper1.lst_residual_scale.'+name,*args]


def main():
    with build_lock(OUTPUT):
        try:
            assert read(OUTPUT/'data_preflight.json')['status']=='passed'
            identity=dict(sources=source_hashes(),configs={str(p.relative_to(HERE)):sha256_file(p) for p in (HERE/'configs').rglob('*.json')},
                parallel_jobs=1,per_process_allocator_gib=12,minimum_free_gib=2)
            if (OUTPUT/'manifest.json').exists():assert read(OUTPUT/'manifest.json')==identity
            else:write(OUTPUT/'manifest.json',identity)
            if not (OUTPUT/'smoke/gate.json').exists():
                execute('smoke','numerical',module('check','numerical'))
                run('smoke','partial');run('smoke','train')
                run('smoke','train','smoke_uninterrupted')
                run('smoke','evaluate')
                execute('smoke','gate',module('check','smoke'))
            run('full','train')
            execute('full','gate',module('check','full'))
            run('full','evaluate')
            execute('full','report',module('report'))
            write(OUTPUT/'status.json',dict(status='complete',utc=utc(),report=str(OUTPUT/'REPORT_ZH.md')))
        except BaseException as e:
            write(OUTPUT/'status.json',dict(status='failed',utc=utc(),error=repr(e),pid=os.getpid()))
            raise


if __name__=='__main__':main()
