"""Two independent trainers sharing exactly one H200; preserve frozen trainer code."""
import json
import time
from pathlib import Path
import modal
from modal_app import image,mounts,runs_volume

APP='efdiff-hlc-width32-epoch1-resume-20260925'
RUN='hlc-width32-epoch1-resume-20260925-v1'
app=modal.App(APP)


def root():
    return Path('/runs')/RUN


def pipeline(value):
    from epoch1_train import save_json
    value.update(execution='concurrent independent processes on one H200',gpu_count=1)
    save_json(root()/'pipeline.json',value)
    runs_volume.commit()
    return value


@app.function(image=image,volumes=mounts,cpu=4,memory=32768,timeout=1800,max_containers=1)
def prepare():
    from width32_epoch1_migrate import prepare_run
    runs_volume.reload()
    result=prepare_run(root())
    runs_volume.commit()
    return result


@app.function(image=image,volumes=mounts,cpu=2,memory=8192,timeout=600,max_containers=1)
def report(records:int):
    from width32_deadline_report import build
    runs_volume.reload()
    result=build(root(),records)
    # The report builder was frozen for the serial phase; annotate actual scheduling.
    path=root()/'comparisons'/str(records)/'REPORT.md'
    path.write_text(path.read_text().replace('This deadline-limited comparison is separate from the default five-epoch formal budget.', 'This user-authorized one-epoch continuation is separate from the default five-epoch formal budget.').replace('One H200 shared sequentially.',
        'One H200 shared by two concurrent training processes; resumed from the 819200-record stage.'),encoding='utf-8')
    result['execution']='concurrent on one H200; continued narrow checkpoints to exactly one epoch'
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
    assert (root()/'resume_migration.json').exists()
    plan=json.loads((HERE/'width32_epoch1_plan.json').read_text())
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
            assert json.loads((Path('/runs')/plan['source_run']/'smoke'/variant/'stage_131.json').read_text())['phase']=='smoke_passed'
            marker=root()/'train'/variant/f'stage_{records}.json'
            if marker.exists():continue
            if variant=='epsilon' and 'x0' in children and resume_steps['x0']-resume_steps['epsilon']<256:
                target_step=resume_steps['epsilon']+256
                start_wait=time.time()
                pipeline({'phase':'concurrent_startup_stagger','stage':stage,
                    'lead_variant':'x0','waiting_variant':'epsilon','target_lead_step':target_step,
                    'stop_at_records':records,'scheduling':scheduling})
                while time.time()-start_wait<600:
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
            cmd=[sys.executable,'/code/efdiff_hlc/width32_epoch1_train.py',
                 '--variant',variant,'--records',str(records),'--output',str(root()/'train'/variant)]
            children[variant]=subprocess.Popen(cmd)
        pipeline({'phase':'training','stage':stage,'target_records':records,
            'variants':list(children),'child_pids':{v:p.pid for v,p in children.items()},
            'stop_at_records':records})
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
    for name in ('pipeline.json','plan.json','latest_comparison.json','resume_migration.json','concurrent_gpu.json','concurrent_scheduling.json'):
        p=root()/name
        if p.exists():out[name]=json.loads(p.read_text())
    for v in ('x0','epsilon'):
        p=root()/'train'/v/'status.json'
        if p.exists():out[v]=json.loads(p.read_text())
    return out
