"""One complete full-HLC epoch per variant, finite checkpointed GPU continuations."""
import json
import hashlib
from pathlib import Path
import modal
from modal_app import image,mounts,runs_volume

APP_NAME='efdiff-hlc-full-epoch1-20260916'
RUN='hlc-synthetic-full-epoch1-20260916-v1'
app=modal.App(APP_NAME)


def root():
    p=Path('/runs')/RUN
    p.mkdir(parents=True,exist_ok=True)
    return p


def phase_of(path):
    return json.loads(path.read_text()).get('phase') if path.exists() else None


@app.function(image=image,volumes=mounts,cpu=4,memory=8192,timeout=1800)
def preflight():
    import numpy as np
    from data import HLCPairs,HERE
    from pilot_data import HLCPairs as ValidationPairs
    from epoch1_train import save_json
    verified=json.loads(Path('/data/_transfer/verification.json').read_text())
    assert verified['status']=='complete' and verified['sizes_and_nanosecond_timestamps_verified']
    train=HLCPairs('/data')
    val=ValidationPairs('/data',split='val')
    assert len(train)==2608772 and len(val)==512
    trids=set(map(int,train.lst.indices[:,0]))
    vids=set(map(int,val.lst.indices[val.members,0]))
    assert not trids&vids
    tiles={s:{r[0] for r in train.lst._connect().execute('SELECT DISTINCT tile FROM records WHERE split=?',(s,))} for s in ('train','val')}
    assert not tiles['train']&tiles['val']
    for ds in (train,val):
        for i in (0,len(ds)//2,len(ds)-1):
            x=ds[i]
            assert x['mask'].any() and x['HLS'].shape==(6,224,224)
            assert x['synthetic_coarse_c'].shape==(1,7,7) and np.isfinite(x['synthetic_coarse_c']).all()
    result={'status':'passed','train_months':len(train),'validation_months':len(val),
            'train_fingerprint':train.fingerprint,'val_fingerprint':val.fingerprint,
            'record_and_tile_split_isolation':True,'test_used':False,
            'budget':json.loads((HERE/'epoch1_budget.json').read_text())}
    save_json(root()/'preflight.json',result)
    runs_volume.commit()
    return result


@app.function(image=image,volumes=mounts,cpu=10,memory=16384,timeout=10800,max_containers=8)
def stats_shard(shard:int):
    from epoch1_stats import fit
    from epoch1_train import save_json
    from data import HERE
    out=root()/'stats_shards'
    out.mkdir(exist_ok=True)
    path=out/f'{shard:02d}.json'
    if path.exists():
        result=json.loads(path.read_text())
        assert result['data_code_sha256']==hashlib.sha256((HERE/'data.py').read_bytes()).hexdigest()
        assert result['stats_code_sha256']==hashlib.sha256((HERE/'epoch1_stats.py').read_bytes()).hexdigest()
        return result
    result=fit('/data',shard,32)
    save_json(path,result)
    runs_volume.commit()
    return result


@app.function(image=image,volumes=mounts,gpu='H200',cpu=12,memory=65536,
              timeout=1800,max_containers=2,single_use_containers=True)
def smoke(variant:str,resume:bool=False):
    from epoch1_train import run
    runs_volume.reload()
    destination=root()/'smoke'/variant
    if phase_of(destination/'status.json')=='smoke_passed':
        return json.loads((destination/'status.json').read_text())
    # Existing train-only pilot statistics are used ONLY for technical smoke checks.
    stats=json.loads(Path('/runs/hlc-synthetic-pilot-16384-512-v1/normalization.json').read_text())
    return run('/data',destination,stats,variant=variant,smoke=True,
               stop_after_updates=None if resume else 2,commit=runs_volume.commit)


def train_segment(variant):
    from epoch1_train import run,save_json
    runs_volume.reload()
    out=root()
    destination=out/variant
    if phase_of(destination/'status.json')=='training_complete':
        report.spawn()
        return json.loads((destination/'status.json').read_text())
    assert json.loads((out/'preflight.json').read_text())['status']=='passed'
    assert phase_of(out/'smoke'/variant/'status.json')=='smoke_passed'
    stats=json.loads((out/'normalization.json').read_text())
    try:
        result=run('/data',destination,stats,variant=variant,commit=runs_volume.commit)
        assert result['phase'] in ('checkpointed','training_complete')
        if result['phase']=='checkpointed':
            follow=(train_x0 if variant=='x0' else train_epsilon).spawn()
            save_json(out/f'continuation_{variant}.json',dict(result,next_call_id=follow.object_id))
            runs_volume.commit()
        else:
            assert result['epochs']==1 and result['updates']==40763 and result['record_presentations']==2608772
            report.spawn()
        return result
    except Exception as exc:
        save_json(out/f'failure_{variant}.json',{'phase':'failed','error':repr(exc)})
        runs_volume.commit()
        raise


# A separate serialized queue per variant prevents overlapping copies of that model.
@app.function(image=image,volumes=mounts,gpu='H200',cpu=12,memory=65536,
              timeout=21600,max_containers=1,single_use_containers=True)
def train_x0():
    return train_segment('x0')


@app.function(image=image,volumes=mounts,gpu='H200',cpu=12,memory=65536,
              timeout=21600,max_containers=1,single_use_containers=True)
def train_epsilon():
    return train_segment('epsilon')


@app.function(image=image,volumes=mounts,cpu=2,memory=4096,timeout=900,max_containers=1,single_use_containers=True)
def report():
    from epoch1_report import build
    from epoch1_train import save_json
    runs_volume.reload()
    if not all(phase_of(root()/v/'status.json')=='training_complete' for v in ('x0','epsilon')):
        return {'phase':'waiting_for_other_variant'}
    result=build(root())
    save_json(root()/'pipeline.json',{'phase':'complete','comparison':result})
    runs_volume.commit()
    return result


@app.function(image=image,volumes=mounts,cpu=1,memory=4096,timeout=43200,max_containers=1,single_use_containers=True)
def launch():
    from epoch1_train import save_json
    from epoch1_stats import merge
    runs_volume.reload()
    if phase_of(root()/'pipeline.json') in ('training','complete'):
        return json.loads((root()/'pipeline.json').read_text())
    try:
        save_json(root()/'pipeline.json',{'phase':'preflight','scope':'full train, one epoch'})
        runs_volume.commit()
        preflight.remote()
        stats_calls=[stats_shard.spawn(i) for i in range(32)]
        calls=[smoke.spawn(v,False) for v in ('x0','epsilon')]
        for c in calls:
            assert c.get()['phase'] in ('checkpointed','smoke_passed')
        calls=[smoke.spawn(v,True) for v in ('x0','epsilon')]
        for c in calls:
            assert c.get()['phase']=='smoke_passed'
        runs_volume.reload()
        save_json(root()/'pipeline.json',{'phase':'full_training_statistics','smoke_resume':'both passed','stats_shards':32})
        runs_volume.commit()
        stats=merge([c.get() for c in stats_calls])
        runs_volume.reload()
        save_json(root()/'normalization.json',stats)
        runs_volume.commit()
        calls={'x0':train_x0.spawn().object_id,'epsilon':train_epsilon.spawn().object_id}
        save_json(root()/'pipeline.json',{'phase':'training','scope':'full train, one epoch','calls':calls})
        runs_volume.commit()
        return {'phase':'training','calls':calls}
    except Exception as exc:
        runs_volume.reload()
        save_json(root()/'pipeline.json',{'phase':'preparation_failed','error':repr(exc)})
        runs_volume.commit()
        raise
