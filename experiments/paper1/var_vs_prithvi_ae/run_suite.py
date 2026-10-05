"""Bounded GPU queues for eight scratch ablations, with memory telemetry and safe stop."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from eo_data.core import json_write as atomic_json_write,sha256_file
from eo_data.locking import build_lock
from eo_denoise.engine import source_hashes

def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(path.read_text(encoding='utf-8'))

def json_write(path,value):
    """Retry Windows sharing conflicts without hiding persistent write failures."""
    for attempt in range(8):
        try:
            atomic_json_write(path,value)
            return
        except PermissionError:
            if attempt==7:raise
            time.sleep(min(.1*2**attempt,1.0))

def command_for(job,stage,output,phase):
    destination=output/job['name']
    if stage=='train':
        status=destination/'status.json'
        if status.exists() and read(status)['status']=='complete':return None
        command=[sys.executable,'-m','eo_denoise','train','--config',str(job['config']),'--output',str(destination)]
        if (destination/'last.pt').exists():command+=['--resume',str(destination/'last.pt')]
        return command
    target=destination/stage
    if (target/'metrics.json').exists():return None
    return [sys.executable,'-m','eo_denoise','evaluate','--checkpoint',str(destination/'last.pt'),
        '--output',str(target),'--coverage','0' if stage=='eval_clean' else '.1',
        '--limit','64' if phase=='pilot' else '11757','--noise-kind','bank_missing']

def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    paths=list(sorted((HERE/'configs'/args.phase).glob('*.json')))
    if len(paths)!=8:raise ValueError('Exactly eight configurations required')
    caps={b:max(read(p)['train']['memory_limit_gib'] for p in paths if ('prithvi' in p.stem)==(b=='prithvi')) for b in ('conv','prithvi')}
    if caps['conv']*args.conv_slots+caps['prithvi']*args.prithvi_slots>26:
        raise ValueError('Parallel process memory caps exceed the tested 26 GiB allocator budget')
    identity=dict(config_hashes={p.name:sha256_file(p) for p in paths},source_hashes=source_hashes())
    if args.phase=='full':
        gate=read(args.pilot_gate)
        if gate.get('status')!='passed' or gate.get('source_hashes')!=identity['source_hashes']:
            raise ValueError('Full runs require a successful small-test gate for the current source')
    manifest=output/'manifest.json'
    if manifest.exists():
        old=read(manifest)
        if any(old[k]!=identity[k] for k in identity):raise ValueError('Suite source/config changed')
    else:
        json_write(manifest,dict(**identity,created_utc=now(),phase=args.phase,
            pretrained_weights=False,steps_per_run=128 if args.phase=='pilot' else 10813,
            effective_batch=16 if args.phase=='pilot' else 64,
            train_pairs=64 if args.phase=='pilot' else 230659,
            final_val_pairs=64 if args.phase=='pilot' else 11757,
            comparison='VAR ResNet VQ-VAE versus native continuous spatial Prithvi ViT AE',
            prithvi_vq=False,prithvi_kl=False,rate_matched=False,
            parallel_limits={'conv':args.conv_slots,'prithvi':args.prithvi_slots},
            checkpoint_selection='fixed final last.pt; final clean and noisy validation use all val records'))
    jobs=[dict(name=p.stem,config=p,backbone='prithvi' if 'prithvi' in p.stem else 'conv',stage=0) for p in paths]
    stages=('train','eval_clean','eval_noisy10')
    active=[];completed=[];stopped=False
    def state(status):
        return dict(status=status,phase=args.phase,supervisor_pid=os.getpid(),updated_utc=now(),
            completed=completed,pending=[j['name'] for j in jobs],
            active=[dict(name=j['name'],stage=stages[j['stage']],pid=j['process'].pid) for j in active])
    def launch(job):
        if source_hashes()!=identity['source_hashes']:raise ValueError('Framework changed while queue was active')
        while job['stage']<len(stages):
            command=command_for(job,stages[job['stage']],output,args.phase)
            if command is not None:
                stream=(output/(job['name']+'.stdout.log')).open('a',encoding='utf-8')
                job['stream']=stream
                job['process']=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                return True
            job['stage']+=1
        completed.append(job['name']);return False
    def stop_active():
        for job in active:
            if stages[job['stage']]=='train':
                destination=output/job['name']
                if destination.exists():(destination/'STOP').write_text('checkpoint and stop\n')
        deadline=time.time()+45
        while any(j['process'].poll() is None for j in active) and time.time()<deadline:time.sleep(.2)
        for job in active:
            if job['process'].poll() is None:
                subprocess.run(['taskkill','/PID',str(job['process'].pid),'/T','/F'],capture_output=True)
            job['stream'].close()
    with build_lock(output):
        try:
            last_gpu=0
            while jobs or active:
                if (output/'STOP').exists():
                    stopped=True;break
                for job in active[:]:
                    code=job['process'].poll()
                    if code is None:continue
                    job['stream'].close();active.remove(job)
                    if code:raise RuntimeError(f"{job['name']} / {stages[job['stage']]} exited {code}; see stdout log")
                    if stages[job['stage']]=='train' and read(output/job['name']/'status.json')['status']!='complete':
                        raise RuntimeError(f"{job['name']} stopped before configured training completion")
                    job['stage']+=1
                    if launch(job):active.append(job)
                for backbone,limit in [('conv',args.conv_slots),('prithvi',args.prithvi_slots)]:
                    while sum(j['backbone']==backbone for j in active)<limit:
                        candidate=next((j for j in jobs if j['backbone']==backbone),None)
                        if candidate is None:break
                        jobs.remove(candidate)
                        if launch(candidate):active.append(candidate)
                if time.time()-last_gpu>=10:
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                    used,total,util=[float(v.strip()) for v in raw.strip().split(',')]
                    gpu=dict(utc=now(),used_gib=used/1024,total_gib=total/1024,utilization_percent=util,
                             active=[j['name'] for j in active])
                    with (output/'gpu.jsonl').open('a') as f:f.write(json.dumps(gpu)+'\n')
                    if total-used<1024:raise RuntimeError('Device free VRAM below 1 GiB; stopping queues at checkpoints')
                    last_gpu=time.time()
                json_write(output/'status.json',state('running'))
                time.sleep(1)
            if stopped:
                stop_active();json_write(output/'status.json',state('stopped_at_checkpoints'))
            else:
                json_write(output/'status.json',state('training_and_evaluation_complete'))
                if args.phase=='full':
                    with (output/'summary.stdout.log').open('w',encoding='utf-8') as log:
                        subprocess.run([sys.executable,str(HERE/'summarize.py'),'--output',str(output),
                            '--phase','full'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                    json_write(output/'status.json',dict(**state('training_and_evaluation_complete'),
                        report=str(output/'report'/'RESULTS.md')))
        except BaseException as error:
            stop_active()
            failure=dict(**state('failed'),error=repr(error))
            json_write(output/'failure.json',failure);json_write(output/'status.json',failure)
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','full'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot-gate',type=Path,default=ROOT/'runs/paper1/var_vs_prithvi_ae_pilot/gate.json')
    p.add_argument('--conv-slots',type=int,default=1);p.add_argument('--prithvi-slots',type=int,default=1)
    args=p.parse_args()
    if not 1<=args.conv_slots<=2 or not 1<=args.prithvi_slots<=2:p.error('At most two slots per architecture')
    run(args)
