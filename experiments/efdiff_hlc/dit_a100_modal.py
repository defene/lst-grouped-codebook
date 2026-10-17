"""Two independent A100-40GB jobs with bounded automatic continuations; no schedules."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import modal
from modal_app import image, mounts, runs_volume

APP = 'efdiff-hlc-dit-flow-epsilon-a100-20261001'
RUN = 'hlc-dit-center-flow-epsilon-a100-epoch1-20261001-v1'
app = modal.App(APP)
locks = modal.Dict.from_name(APP+'-launch-locks', environment_name='main', create_if_missing=True)


def root():
    p=Path('/runs')/RUN
    p.mkdir(parents=True,exist_ok=True)
    return p


@app.function(image=image,volumes=mounts,gpu='A100-40GB',cpu=24,memory=98304,
              timeout=7200,max_containers=2,single_use_containers=True,retries=0)
def job(variant:str,smoke:bool=False):
    from data import HERE
    from epoch1_train import save_json
    assert variant in ('flow','epsilon')
    runs_volume.reload()
    out=root()/('smoke' if smoke else 'train')/variant
    out.mkdir(parents=True,exist_ok=True)
    child=None
    try:
        if not smoke:
            plan=json.loads((HERE/'dit_a100_plan.json').read_text())
            assert plan['budget_confirmed'] and plan['epochs'] in (1,5)
            assert (root()/'preflight.json').exists()
        if (out/'complete.json').exists():
            return json.loads((out/'complete.json').read_text())
        cmd=[sys.executable,'/code/efdiff_hlc/dit_a100_train.py','--variant',variant,'--output',str(out)]
        if smoke:
            cmd+=['--smoke']
            if not (out/'latest.pt').exists():
                cmd+=['--stop-after','2']
        save_json(out/'pipeline.json',dict(phase='starting',variant=variant,smoke=smoke,unix=time.time()))
        with (out/'console.log').open('a',buffering=1) as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
            while child.poll() is None:
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True).strip()
                snapshot=dict(unix=time.time(),gpu=gpu,child_pid=child.pid,variant=variant,smoke=smoke)
                save_json(out/'gpu.json',snapshot)
                with (out/'gpu.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
                runs_volume.commit()
                time.sleep(20)
            if child.returncode != 0:
                raise RuntimeError(f'Trainer exited {child.returncode}; see {out}/console.log')
        result=json.loads((out/'segment_result.json').read_text())
        if result['phase']=='checkpointed':
            call=job.spawn(variant,smoke)
            status=dict(phase='queued_continuation',call_id=call.object_id,result=result,unix=time.time())
        else:
            status=dict(phase=result['phase'],result=result,unix=time.time())
        save_json(out/'pipeline.json',status)
        runs_volume.commit()
        return status
    except Exception as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        save_json(out/'pipeline.json',dict(phase='failed',error=repr(exc),unix=time.time()))
        runs_volume.commit()
        raise


@app.function(image=image,volumes=mounts,cpu=2,memory=4096,timeout=300,max_containers=1)
def launch():
    from data import HERE
    from epoch1_train import save_json
    runs_volume.reload()
    plan=json.loads((HERE/'dit_a100_plan.json').read_text())
    old_plan=json.loads((HERE/'dit_plan.json').read_text())
    assert plan['budget_confirmed'] and plan['epochs']==1
    allowed={'run_id','app','gpu_per_variant','execution','authorization'}
    assert {k:v for k,v in plan.items() if k not in allowed}=={k:v for k,v in old_plan.items() if k not in allowed}
    old=Path('/runs')/old_plan['run_id']
    evidence={}
    for v in ('flow','epsilon'):
        done=json.loads((old/'smoke'/v/'complete.json').read_text())
        assert done['phase']=='smoke_passed' and done['training_records']==132 and done['tail_records']==4
        evidence[v]=json.loads((old/'smoke'/v/'provenance.json').read_text())
        assert not (old/'train'/v/'latest.pt').exists(), 'Unexpected checkpoint: migrate explicitly instead of restarting'
    for key in ('initial_dit_sha256','frozen_encoder_sha256','stats_hash','train_order_sha256'):
        assert evidence['flow'][key]==evidence['epsilon'][key],key
    for name in ('dit_train.py','dit_model.py','dit_objectives.py','dit_plan.json','center_data.py','center_crop_protocol.json','vscore_diffusion.py','data.py','pilot_data.py'):
        p=HERE/name
        for v in evidence:
            assert evidence[v]['hashes'][str(p.relative_to(HERE.parent))]==hashlib.sha256(p.read_bytes()).hexdigest()
    expected=(HERE/'dit_train.py').read_text().replace("PLAN = HERE / 'dit_plan.json'","PLAN = HERE / 'dit_a100_plan.json'")
    assert (HERE/'dit_a100_train.py').read_text()==expected
    if not locks.put(RUN+':train',dict(unix=time.time()),skip_if_exists=True):
        return dict(phase='already_submitted',scope='train')
    save_json(root()/'preflight.json',dict(phase='passed',functional_smoke_source_run=old_plan['run_id'],
        hardware_validation='verify first A100 training updates',same_initialization=True,same_encoder=True,
        same_normalization=True,same_members=True,trainer_change='plan file path only',plan=plan,test_used=False,unix=time.time()))
    save_json(root()/'hardware_migration.json',dict(source_app=old_plan['app'],source_run=old_plan['run_id'],
        old_app_stopped=True,old_app_id='ap-8UUy7rDC88EnHoDepZ7uSc',target_app=APP,target_run=RUN,
        gpu='A100-40GB',resume_records=0,reason='User requested A100; H200 stopped before first checkpoint',
        old_last_logged={v:json.loads((old/'train'/v/'status.json').read_text()) for v in evidence},unix=time.time()))
    calls={}
    for v in ('flow','epsilon'):
        calls[v]=job.spawn(v,False).object_id
        save_json(root()/'train_launch.json',dict(phase='submitted',calls=calls,plan=plan,unix=time.time()))
        runs_volume.commit()
    return dict(phase='submitted',calls=calls,run_id=RUN,smoke=False)


@app.function(image=image,volumes=mounts,cpu=1,memory=2048,timeout=180,max_containers=1)
def status():
    runs_volume.reload()
    result={}
    for name in ('preflight.json','hardware_migration.json','train_launch.json'):
        p=root()/name
        if p.exists():result[name]=json.loads(p.read_text())
    for scope in ('smoke','train'):
        result[scope]={}
        for v in ('flow','epsilon'):
            result[scope][v]={}
            for name in ('status.json','pipeline.json','gpu.json','complete.json'):
                p=root()/scope/v/name
                if p.exists():result[scope][v][name]=json.loads(p.read_text())
            p=root()/scope/v/'console.log'
            if p.exists():
                with p.open('rb') as f:
                    f.seek(max(0,p.stat().st_size-5000))
                    result[scope][v]['log_tail']=f.read().decode(errors='replace')
    return result
