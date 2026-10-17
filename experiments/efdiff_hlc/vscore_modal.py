"""Two independent H200 jobs with bounded automatic continuations; no schedules."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import modal
from modal_app import image, mounts, runs_volume

APP = 'efdiff-hlc-wide-vscore-20260927'
RUN = 'hlc-wide-vscore-20260927-v1'
app = modal.App(APP)
locks = modal.Dict.from_name(APP+'-launch-locks', environment_name='main', create_if_missing=True)


def root():
    p=Path('/runs')/RUN
    p.mkdir(parents=True,exist_ok=True)
    return p


@app.function(image=image,volumes=mounts,gpu='H200',cpu=24,memory=98304,
              timeout=7200,max_containers=2,single_use_containers=True)
def job(variant:str,smoke:bool=False):
    from data import HERE
    from epoch1_train import save_json
    assert variant in ('v','score')
    runs_volume.reload()
    out=root()/('smoke' if smoke else 'train')/variant
    out.mkdir(parents=True,exist_ok=True)
    child=None
    try:
        if not smoke:
            plan=json.loads((HERE/'vscore_plan.json').read_text())
            assert plan['budget_confirmed'] and plan['epochs'] in (1,5)
            assert (root()/'preflight.json').exists()
        if (out/'complete.json').exists():
            return json.loads((out/'complete.json').read_text())
        cmd=[sys.executable,'/code/efdiff_hlc/vscore_train.py','--variant',variant,'--output',str(out)]
        if smoke:
            cmd+=['--smoke']
            if not (out/'latest.pt').exists():
                cmd+=['--stop-after','2']
        save_json(out/'pipeline.json',dict(phase='starting',variant=variant,smoke=smoke,unix=time.time()))
        with (out/'console.log').open('a',buffering=1) as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
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
def launch(smoke:bool=True):
    from data import HERE
    from epoch1_train import save_json
    runs_volume.reload()
    plan=json.loads((HERE/'vscore_plan.json').read_text())
    if not smoke:
        assert plan['budget_confirmed'] and plan['epochs'] in (1,5)
        evidence={}
        for v in ('v','score'):
            out=root()/'smoke'/v
            done=json.loads((out/'complete.json').read_text())
            assert done['phase']=='smoke_passed' and done['training_records']==132 and done['tail_records']==4
            evidence[v]=json.loads((out/'provenance.json').read_text())
        for key in ('initial_unet_sha256','frozen_encoder_sha256','stats_hash','train_order_sha256'):
            assert evidence['v'][key]==evidence['score'][key],key
        for p in (HERE/'vscore_train.py',HERE/'vscore_diffusion.py',HERE/'data.py'):
            key=str(p.relative_to(HERE.parent))
            for v in evidence:
                assert evidence[v]['hashes'][key]==hashlib.sha256(p.read_bytes()).hexdigest(),key
        save_json(root()/'preflight.json',dict(phase='passed',smoke_records_per_model=132,
                  same_initialization=True,same_encoder=True,same_normalization=True,same_members=True,
                  plan=plan,test_used=False,unix=time.time()))
    key='smoke' if smoke else 'train'
    if not locks.put(RUN+':'+key,dict(unix=time.time()),skip_if_exists=True):
        return dict(phase='already_submitted',scope=key)
    calls={}
    # Submission is recorded immediately so a partially interrupted launch is auditable.
    for v in ('v','score'):
        calls[v]=job.spawn(v,smoke).object_id
        save_json(root()/f'{key}_launch.json',dict(phase='submitted',calls=calls,plan=plan,unix=time.time()))
        runs_volume.commit()
    return dict(phase='submitted',calls=calls,run_id=RUN,smoke=smoke)


@app.function(image=image,volumes=mounts,cpu=1,memory=2048,timeout=180,max_containers=1)
def status():
    runs_volume.reload()
    result={}
    for name in ('preflight.json','smoke_launch.json','train_launch.json'):
        p=root()/name
        if p.exists():result[name]=json.loads(p.read_text())
    for scope in ('smoke','train'):
        result[scope]={}
        for v in ('v','score'):
            result[scope][v]={}
            for name in ('status.json','pipeline.json','gpu.json'):
                p=root()/scope/v/name
                if p.exists():result[scope][v][name]=json.loads(p.read_text())
            p=root()/scope/v/'console.log'
            if p.exists():
                with p.open('rb') as f:
                    f.seek(max(0,p.stat().st_size-5000))
                    result[scope][v]['log_tail']=f.read().decode(errors='replace')
    return result
