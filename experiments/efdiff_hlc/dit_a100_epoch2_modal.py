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

APP = 'efdiff-hlc-dit-a100-b128-fullval-20261001'
RUN = 'hlc-dit-a100-b128-fullval-epoch2-20261001-v1'
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
            plan=json.loads((HERE/'dit_a100_epoch2_plan.json').read_text())
            assert plan['budget_confirmed'] and plan['epochs']==2
            assert (root()/'preflight.json').exists()
        if (out/'complete.json').exists():
            return json.loads((out/'complete.json').read_text())
        cmd=[sys.executable,'/code/efdiff_hlc/dit_a100_epoch2_train.py','--variant',variant,'--output',str(out)]
        if not smoke and not (out/'batch128_verified.json').exists():
            migration=json.loads((root()/'resume_migration.json').read_text())
            cmd+=['--stop-after',str(migration['models'][variant]['step']+5)]
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
            if result.get('update_records')==128 and not (out/'batch128_verified.json').exists():
                save_json(out/'batch128_verified.json',result);runs_volume.commit()
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


@app.function(image=image,volumes=mounts,cpu=4,memory=16384,timeout=900,max_containers=1)
def launch():
    import shutil
    import torch
    import numpy as np
    from data import HERE
    from center_data import HLCPairs, EpochSampler, check_stats
    from epoch1_train import save_json
    runs_volume.reload()
    plan=json.loads((HERE/'dit_a100_epoch2_plan.json').read_text())
    old_plan=json.loads((HERE/'dit_a100_plan.json').read_text())
    assert plan['epochs']==2 and plan['batch_size']==128 and plan['validation_records']==141711
    allowed={'run_id','app','epochs','updates_per_model','record_presentations_per_model','authorization','scope','resume_source_run','learning_rate_extension','batch_size','tail_records','validation_records','validation_fraction_per_epoch','warmup_records','ema_start_records','ema_decay','ema_notes','updates_definition'}
    assert {k:v for k,v in plan.items() if k not in allowed}=={k:v for k,v in old_plan.items() if k not in allowed}
    assert plan['record_presentations_per_model']==5217544
    if not locks.put(RUN+':train',dict(unix=time.time()),skip_if_exists=True):
        return dict(phase='already_submitted',scope='train')
    old=Path('/runs')/plan['resume_source_run']
    stats=json.loads(Path(plan['normalization_path']).read_text())
    ds=HLCPairs('/data',stats=stats,augment=False,seed=plan['seed']);check_stats(stats,ds)
    orders={}
    for epoch in range(2):
        order=EpochSampler(ds,epoch).order.astype('<i4')
        assert len(order)==len(ds) and len(np.unique(order))==len(ds)
        orders[str(epoch)]=hashlib.sha256(order.tobytes()).hexdigest()
    records={};provenances={}
    for variant in ('flow','epsilon'):
        src=old/'train'/variant;out=root()/'train'/variant;out.mkdir(parents=True,exist_ok=True)
        provenance=json.loads((src/'provenance.json').read_text())
        assert provenance['train_order_sha256']['0']==orders['0']
        for name,digest in provenance['hashes'].items():
            assert hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest()==digest,name
        checkpoint=src/'latest.pt'
        with checkpoint.open('rb') as f:source_sha=hashlib.file_digest(f,'sha256').hexdigest()
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['hashes']==provenance['hashes'] and state['stats_hash']==provenance['stats_hash']
        assert state['fingerprint']==ds.fingerprint and state['variant']==variant and not state['smoke']
        assert state['presented']==state['epoch']*len(ds)+state['offset']
        assert state['step']==state['epoch']*((len(ds)+63)//64)+(state['offset']+63)//64
        assert state['epoch']<2
        old_hashes=dict(state['hashes']);new_hashes=dict(old_hashes)
        for old_name,new_name in [('dit_a100_plan.json','dit_a100_epoch2_plan.json'),('dit_a100_train.py','dit_a100_epoch2_train.py')]:
            del new_hashes['efdiff_hlc/'+old_name]
            new_hashes['efdiff_hlc/'+new_name]=hashlib.sha256((HERE/new_name).read_bytes()).hexdigest()
        del new_hashes['efdiff_hlc/pilot_membership/val.npy']
        new_hashes['efdiff_hlc/dit_full_validation.py']=hashlib.sha256((HERE/'dit_full_validation.py').read_bytes()).hexdigest()
        state['hashes']=new_hashes
        state['best']=None  # Old 512-subset scores cannot select full-validation best.
        state['schedule_total_steps']=state['step']+(len(ds)-state['offset']+127)//128+(2-state['epoch']-1)*((len(ds)+127)//128)
        # All tensors, optimizer moments, EMA, RNG and progress remain unchanged.
        tmp=out/'latest.tmp';torch.save(state,tmp);tmp.replace(out/'latest.pt')
        records[variant]=dict(source_checkpoint=str(checkpoint),source_sha256=source_sha,
            epoch=state['epoch'],offset=state['offset'],step=state['step'],presented=state['presented'],
            old_hashes=old_hashes,new_hashes=new_hashes,schedule_total_steps=state['schedule_total_steps'],best_reset_reason='full validation replaces subset',preserved=['model','optimizer','ema','cpu_rng','cuda_rng','elapsed_seconds'])
        del state
        for pattern in ('metrics.jsonl',):
            for source in src.glob(pattern):
                lines=[line for line in source.read_text().splitlines() if json.loads(line)['step']<=records[variant]['step']]
                (out/source.name).write_text('\n'.join(lines)+'\n')
        provenance.update(plan=plan,validation_records=141711,hashes=new_hashes,train_order_sha256=orders,resume_source_run=plan['resume_source_run'],resume_source_checkpoint_sha256=source_sha)
        save_json(out/'provenance.json',provenance);provenances[variant]=provenance
        for name in new_hashes:
            dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE.parent/name,dest)
    for key in ('initial_dit_sha256','frozen_encoder_sha256','stats_hash','train_order_sha256','fingerprint'):
        assert provenances['flow'][key]==provenances['epsilon'][key],key
    save_json(root()/'resume_migration.json',dict(source_run=plan['resume_source_run'],models=records,
        cumulative_target_records=5217544,cumulative_target_updates={v:records[v]["schedule_total_steps"] for v in records},batch_size=128,epochs=2,validation_records=141711,unix=time.time()))
    save_json(root()/'preflight.json',dict(phase='passed',plan=plan,same_members=True,same_initialization=True,test_used=False,unix=time.time()))
    runs_volume.commit()
    calls={}
    for variant in ('flow','epsilon'):
        calls[variant]=job.spawn(variant,False).object_id
        save_json(root()/'train_launch.json',dict(phase='submitted',calls=calls,plan=plan,unix=time.time()));runs_volume.commit()
    return dict(phase='submitted',calls=calls,run_id=RUN,resume=records)


@app.function(image=image,volumes=mounts,cpu=1,memory=2048,timeout=180,max_containers=1)
def status():
    runs_volume.reload()
    result={}
    for name in ('preflight.json','resume_migration.json','train_launch.json'):
        p=root()/name
        if p.exists():result[name]=json.loads(p.read_text())
    for scope in ('smoke','train'):
        result[scope]={}
        for v in ('flow','epsilon'):
            result[scope][v]={}
            for name in ('status.json','pipeline.json','gpu.json','complete.json','batch128_verified.json'):
                p=root()/scope/v/name
                if p.exists():result[scope][v][name]=json.loads(p.read_text())
            p=root()/scope/v/'console.log'
            if p.exists():
                with p.open('rb') as f:
                    f.seek(max(0,p.stat().st_size-5000))
                    result[scope][v]['log_tail']=f.read().decode(errors='replace')
    return result
