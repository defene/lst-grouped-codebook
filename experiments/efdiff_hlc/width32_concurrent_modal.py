"""Two independent trainers sharing exactly one H200; preserve frozen trainer code."""
import json
import time
from pathlib import Path
import modal
from modal_app import image,mounts,runs_volume

APP='efdiff-hlc-width32-concurrent-12h-20260923'
RUN='hlc-width32-single-12h-20260923-v1'
app=modal.App(APP)


def root():
    return Path('/runs')/RUN


def pipeline(value):
    from epoch1_train import save_json
    value.update(execution='concurrent independent processes on one H200',gpu_count=1)
    save_json(root()/'pipeline.json',value)
    runs_volume.commit()
    return value


@app.function(image=image,volumes=mounts,cpu=4,memory=16384,timeout=300,max_containers=1)
def prepare():
    import torch
    from epoch1_train import save_json
    runs_volume.reload()
    plan=json.loads((root()/'plan.json').read_text())
    assert time.time()<plan['deadline_unix']-1800
    migration=root()/'concurrent_migration.json'
    if migration.exists():return json.loads(migration.read_text())
    previous=json.loads((root()/'pipeline.json').read_text())
    state=torch.load(root()/'train/x0/latest.pt',map_location='cpu',weights_only=False)
    assert state['variant']=='x0' and not state['smoke']
    result={'previous_pipeline':previous,'resume_step':state['step'],'resume_records':state['offset'],
        'original_elapsed_seconds':state['elapsed_seconds'],'changed_at_unix':time.time(),
        'new_app':APP,'new_execution':'two simultaneous independent trainers on a single H200',
        'unchanged':'trainer, data, membership, normalization, width32, batch64, optimizer, EMA, stages, deadline',
        'frozen_plan_note':'original plan execution label is superseded by this migration record',
        'checkpoint_source_hashes':state['hashes']}
    save_json(migration,result)
    del state
    pipeline({'phase':'concurrent_prepared','migration':result})
    return result


@app.function(image=image,volumes=mounts,cpu=2,memory=8192,timeout=600,max_containers=1)
def report(records:int):
    from width32_deadline_report import build
    runs_volume.reload()
    result=build(root(),records)
    # The report builder was frozen for the serial phase; annotate actual scheduling.
    path=root()/'comparisons'/str(records)/'REPORT.md'
    path.write_text(path.read_text().replace('One H200 shared sequentially.',
        'One H200 shared by two concurrent training processes after a checkpointed serial start.'),encoding='utf-8')
    result['execution']='concurrent on one H200 after checkpointed serial start'
    from epoch1_train import save_json
    save_json(root()/'comparisons'/str(records)/'comparison.json',result)
    save_json(root()/'latest_comparison.json',result)
    runs_volume.commit()
    return result


@app.function(image=image,volumes=mounts,gpu='H200',cpu=48,memory=131072,timeout=7200,
              max_containers=1,single_use_containers=True)
def job(stage:int=0):
    import subprocess
    import sys
    from data import HERE
    from epoch1_train import save_json
    runs_volume.reload()
    assert (root()/'concurrent_migration.json').exists()
    plan=json.loads((HERE/'width32_12h_plan.json').read_text())
    if time.time()>=plan['deadline_unix']-1800:
        return pipeline({'phase':'deadline_stopped'})
    records=plan['stage_records'][stage]
    children={}
    try:
        # Scheduling-only recovery: identical frozen trainers, offset the two
        # prefetch queues so they do not continuously request the same cold files.
        # Keep the update/member sequence unchanged for each model.
        import torch
        resume_steps={}
        for variant in ('x0','epsilon'):
            cp=root()/'train'/variant/'latest.pt'
            if cp.exists():
                state=torch.load(cp,map_location='cpu',weights_only=False)
                resume_steps[variant]=state['step']
                del state
            else:resume_steps[variant]=0
        scheduling={'unix':time.time(),'stage':stage,'resume_steps':resume_steps,
            'startup_lead_updates':256,'maximum_startup_delay_seconds':600,
            'scope':'execution scheduling only; unchanged frozen trainer/data/config; same GPU'}
        save_json(root()/'concurrent_scheduling.json',scheduling)
        for variant in ('x0','epsilon'):
            assert json.loads((root()/'smoke'/variant/'stage_131.json').read_text())['phase']=='smoke_passed'
            marker=root()/'train'/variant/f'stage_{records}.json'
            if marker.exists():continue
            if variant=='epsilon' and 'x0' in children and resume_steps['x0']-resume_steps['epsilon']<256:
                target_step=resume_steps['epsilon']+256
                start_wait=time.time()
                pipeline({'phase':'concurrent_startup_stagger','stage':stage,
                    'lead_variant':'x0','waiting_variant':'epsilon','target_lead_step':target_step,
                    'deadline':plan['deadline_utc'],'scheduling':scheduling})
                while time.time()-start_wait<600 and time.time()<plan['deadline_unix']-1800:
                    code=children['x0'].poll()
                    if code is not None:
                        if code!=0:raise RuntimeError(f'x0 startup exited {code}')
                        break
                    p=root()/'train/x0/status.json'
                    current=json.loads(p.read_text()) if p.exists() else {}
                    if current.get('step',current.get('updates',0))>=target_step:break
                    time.sleep(10)
                scheduling['actual_startup_delay_seconds']=time.time()-start_wait
                save_json(root()/'concurrent_scheduling.json',scheduling)
            cmd=[sys.executable,'/code/efdiff_hlc/width32_deadline_train.py',
                 '--variant',variant,'--records',str(records),'--output',str(root()/'train'/variant)]
            children[variant]=subprocess.Popen(cmd)
        pipeline({'phase':'training','stage':stage,'target_records':records,
            'variants':list(children),'child_pids':{v:p.pid for v,p in children.items()},
            'deadline':plan['deadline_utc']})
        while any(p.poll() is None for p in children.values()):
            failed={v:p.returncode for v,p in children.items() if p.poll() not in (None,0)}
            if failed:raise RuntimeError(f'Trainer failure: {failed}')
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).strip()
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                '--format=csv,noheader,nounits'],text=True).strip()
            snapshot={'unix':time.time(),'gpu':gpu,'processes':processes,
                'child_pids':{v:p.pid for v,p in children.items()},
                'alive':{v:p.poll() is None for v,p in children.items()}}
            save_json(root()/'concurrent_gpu.json',snapshot)
            with (root()/'concurrent_gpu.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
            runs_volume.commit()
            time.sleep(30)
        for v,p in children.items():
            if p.returncode!=0:raise RuntimeError(f'{v} exited {p.returncode}')
        results={v:json.loads((root()/'train'/v/'segment_result.json').read_text())
                 for v in children}
        if any(r['phase']=='deadline_stop' for r in results.values()):
            return pipeline({'phase':'deadline_stop','results':results})
        complete=all((root()/'train'/v/f'stage_{records}.json').exists() for v in ('x0','epsilon'))
        if not complete:
            call=job.spawn(stage)
            return pipeline({'phase':'training_queued','stage':stage,'call_id':call.object_id,'results':results})
        comparison=report.remote(records)
        runs_volume.reload()
        if stage+1<len(plan['stage_records']):
            needed=(plan['stage_records'][stage+1]-records)/64*plan['conservative_pair_seconds_per_update']+1800
            if time.time()+needed<plan['deadline_unix']:
                call=job.spawn(stage+1)
                return pipeline({'phase':'training_queued','stage':stage+1,'call_id':call.object_id,
                    'latest_comparison':comparison})
        return pipeline({'phase':'complete','latest_comparison':comparison,'test_used':False})
    except Exception as exc:
        for p in children.values():
            if p.poll() is None:p.terminate()
        for p in children.values():
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        pipeline({'phase':'failed','stage':stage,'error':repr(exc)})
        raise


@app.function(image=image,volumes=mounts,cpu=1,memory=1024,timeout=120,max_containers=1)
def status():
    runs_volume.reload()
    out={}
    for name in ('pipeline.json','plan.json','latest_comparison.json','concurrent_migration.json','concurrent_gpu.json','concurrent_scheduling.json'):
        p=root()/name
        if p.exists():out[name]=json.loads(p.read_text())
    for v in ('x0','epsilon'):
        p=root()/'train'/v/'status.json'
        if p.exists():out[v]=json.loads(p.read_text())
    return out
