"""Finite, explicitly authorized subset pilot; automatically reports both variants."""
import json
import shutil
from pathlib import Path
import modal
from modal_app import image, mounts, runs_volume

APP_NAME = 'efdiff-hlc-pilot-20260915'
RUN = 'hlc-synthetic-pilot-16384-512-v1'
app = modal.App(APP_NAME)
locks = modal.Dict.from_name(APP_NAME+'-locks',environment_name='main',create_if_missing=True)


def root():
    out = Path('/runs')/RUN
    out.mkdir(parents=True,exist_ok=True)
    return out


@app.function(image=image,volumes=mounts,cpu=10,memory=16384,timeout=7200)
def prepare():
    import numpy as np
    from pilot_data import HLCPairs, MEMBERS
    from pilot_fit_stats import fit
    from pilot_train import save_json
    verification = json.loads(Path('/data/_transfer/verification.json').read_text())
    assert verification['status']=='complete' and verification['sizes_and_nanosecond_timestamps_verified']
    for split in ('train','val'):
        ds = HLCPairs('/data',split=split)
        for i in (0,len(ds)//2,len(ds)-1):
            sample = ds[i]
            assert sample['mask'].any() and np.isfinite(sample['synthetic_coarse_c']).all()
            assert sample['HLS'].shape == (6,224,224) and sample['synthetic_coarse_c'].shape == (1,7,7)
    shutil.copytree(MEMBERS,root()/'membership',dirs_exist_ok=True)
    save_json(root()/'preflight.json',{'status':'passed','upload_verified':True,'scope':'pilot','test_used':False})
    stats = fit('/data')
    save_json(root()/'normalization.json',stats)
    runs_volume.commit()
    return {'phase':'prepared','records':stats['records']}


@app.function(image=image,volumes=mounts,gpu='H200',cpu=12,memory=65536,
              timeout=28800,max_containers=2,single_use_containers=True)
def gpu_job(variant:str,stage:str):
    from pilot_train import run, save_json
    runs_volume.reload()
    assert variant in ('x0','epsilon') and stage in ('smoke','smoke_resume','train')
    stats = json.loads((root()/'normalization.json').read_text())
    if stage != 'train':
        return run('/data',root()/'smoke'/variant,stats,variant,smoke=True,
                   stop_after_updates=2 if stage=='smoke' else None,commit=runs_volume.commit)
    assert json.loads((root()/'smoke'/variant/'status.json').read_text())['phase']=='smoke_passed'
    return run('/data',root()/variant,stats,variant,commit=runs_volume.commit)


@app.function(image=image,volumes=mounts,cpu=2,memory=8192,timeout=43200,
              max_containers=1,single_use_containers=True)
def launch():
    from pilot_train import save_json
    from pilot_report import build
    if not locks.put(RUN,True,skip_if_exists=True):
        return {'phase':'already_active'}
    def phase(name, **extra):
        runs_volume.reload()
        save_json(root()/'pipeline.json',dict(phase=name,scope='pilot',**extra))
        runs_volume.commit()
    try:
        runs_volume.reload()
        def saved_phase(path):
            return json.loads(path.read_text()).get('phase') if path.exists() else None
        if all(saved_phase(root()/v/'status.json')=='training_complete' for v in ('x0','epsilon')):
            result = build(root())
            phase('complete',comparison=result)
            return result
        if not (root()/'normalization.json').exists():
            phase('preparing')
            prepare.remote()
        phase('smoke')
        pending_smoke = [v for v in ('x0','epsilon') if saved_phase(root()/'smoke'/v/'status.json')!='smoke_passed']
        calls = [gpu_job.spawn(v,'smoke') for v in pending_smoke]
        for call in calls:
            assert call.get()['phase'] in ('checkpointed','smoke_passed')
        calls = [gpu_job.spawn(v,'smoke_resume') for v in pending_smoke]
        for call in calls:
            assert call.get()['phase']=='smoke_passed'
        phase('training')
        calls = [gpu_job.spawn(v,'train') for v in ('x0','epsilon') if saved_phase(root()/v/'status.json')!='training_complete']
        for call in calls:
            assert call.get()['phase']=='training_complete'
        phase('reporting')
        result = build(root())
        phase('complete',comparison=result)
        return result
    except Exception as exc:
        phase('failed',error=repr(exc))
        raise
    finally:
        locks.pop(RUN,None)
