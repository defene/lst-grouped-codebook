"""Three bounded concurrent jobs, frozen identities, restartable stages and automatic full evaluations."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from eo_data.core import json_write,sha256_file
from eo_data.locking import build_lock
from .prepare import HERE,ROOT,OUTPUT,ARMS,read
from .train import source_hashes,utc


def append(path,row):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')


def identity():
    return dict(sources=source_hashes(),configs={str(p.relative_to(HERE)):sha256_file(p)
        for p in sorted((HERE/'configs').rglob('*.json'))},pretrained=read(OUTPUT/'pretrained.json'),
        packages=read(OUTPUT/'packages.json'),parallel_jobs=3,per_process_allocator_gib=8.5,
        minimum_free_gib=2)


def execute(phase,arm,stage,command,dest):
    assert identity()==read(OUTPUT/'manifest.json'),'Frozen source/config/weights identity changed'
    if (OUTPUT/'STOP').exists() or (dest/'STOP').exists():raise RuntimeError('Explicit STOP; inspect before resuming')
    start=time.perf_counter()
    with (OUTPUT/f'{phase}_{arm}_{stage}.log').open('a',encoding='utf-8') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        event=OUTPUT/f'events_{phase}_{arm}.jsonl'
        append(event,dict(event='start',stage=stage,pid=p.pid,utc=utc()))
        json_write(OUTPUT/f'job_{arm}.json',dict(status='running',phase=phase,stage=stage,pid=p.pid,utc=utc()))
        try:
            while p.poll() is None:
                # STOP is handled by training/evaluation at their safe boundaries.
                time.sleep(2)
            assert p.returncode==0,f'{phase}/{arm}/{stage}: exit {p.returncode}; inspect log'
        finally:
            append(event,dict(event='end',stage=stage,pid=p.pid,exit_code=p.poll(),utc=utc(),
                elapsed_seconds=time.perf_counter()-start))
        json_write(OUTPUT/f'job_{arm}.json',dict(status='complete',phase=phase,stage=stage,pid=p.pid,utc=utc()))


def run(phase,arm,stage,folder=None):
    dest=OUTPUT/(folder or phase)/arm
    if stage=='train' and (dest/'status.json').exists() and read(dest/'status.json').get('status')=='complete':return
    if stage=='partial' and (dest/'last.pt').exists():return
    if stage in ('evaluate','pretrained') and all((dest/f'final_{s}_{c}/metrics.json').exists()
        for s in ('val','test') for c in ('clean','noisy10')):return
    args=[sys.executable,'-m','experiments.paper1.lst_sd_color.run','train' if stage=='partial' else stage,
        '--config',str(HERE/'configs'/phase/(arm+'.json')),'--output',str(dest)]
    if stage=='partial':args+=['--stop-after','2']
    execute(folder or phase,arm,stage,args,dest)
    if stage=='train':assert read(dest/'status.json')['status']=='complete','Training stopped before exact budget'


def module(name,*args):return [sys.executable,'-m','experiments.paper1.lst_sd_color.'+name,*args]


def parallel(function):
    errors=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(function,a):a for a in ARMS}
        while futures:
            done=[f for f in futures if f.done()]
            for f in done:
                arm=futures.pop(f)
                try:f.result()
                except BaseException as e:
                    errors.append((arm,repr(e)))
                    json_write(OUTPUT/f'job_{arm}.json',dict(status='failed',error=repr(e),utc=utc()))
            snap=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            used,total,util=map(float,snap.stdout.strip().split(','))
            append(OUTPUT/'gpu.jsonl',dict(utc=utc(),used_gib=used/1024,total_gib=total/1024,
                utilization_percent=util,active_arms=list(futures.values())))
            if (total-used)/1024<2 and not (OUTPUT/'STOP').exists():
                (OUTPUT/'STOP').write_text('Automatic memory safety stop: less than 2 GiB free. Inspect GPU before resuming.',encoding='utf-8')
            if futures:time.sleep(10)
    if errors:raise RuntimeError(str(errors))


def smoke(arm):
    run('smoke',arm,'partial')
    run('smoke',arm,'train')
    run('smoke',arm,'train','smoke_uninterrupted')
    run('smoke',arm,'evaluate')


def full(arm):
    run('full',arm,'train')
    run('full',arm,'evaluate')


def main():
    with build_lock(OUTPUT):
        try:
            assert read(OUTPUT/'data_preflight.json')['status']=='passed'
            if (OUTPUT/'manifest.json').exists():assert read(OUTPUT/'manifest.json')==identity()
            else:json_write(OUTPUT/'manifest.json',identity())
            json_write(OUTPUT/'status.json',dict(status='running',stage='smoke',pid=os.getpid(),utc=utc()))
            if not (OUTPUT/'smoke/gate.json').exists():
                assert read(OUTPUT/'numerical_checks.json')['status']=='passed'
                parallel(smoke)
                execute('smoke','all','gate',module('check','smoke'),OUTPUT/'smoke')
            json_write(OUTPUT/'status.json',dict(status='running',stage='pretrained_full_evaluation',pid=os.getpid(),utc=utc()))
            parallel(lambda arm:run('full',arm,'pretrained','pretrained'))
            execute('pretrained','all','report',module('report','--partial'),OUTPUT)
            json_write(OUTPUT/'status.json',dict(status='running',stage='finetune_and_evaluate',pid=os.getpid(),utc=utc()))
            parallel(full)
            execute('full','all','gate',module('check','full'),OUTPUT/'full')
            execute('full','all','report',module('report'),OUTPUT)
            json_write(OUTPUT/'status.json',dict(status='complete_pending_visual_review',utc=utc(),pid=os.getpid()))
        except BaseException as e:
            json_write(OUTPUT/'status.json',dict(status='failed',error=repr(e),utc=utc(),pid=os.getpid()))
            raise


if __name__=='__main__':main()
