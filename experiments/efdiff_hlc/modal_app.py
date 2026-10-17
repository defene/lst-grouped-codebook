"""Deployed finite HLC experiment queue; no schedules or public endpoints."""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
APP_NAME = 'efdiff-hlc-synthetic-20260915'
RUN = 'hlc-synthetic-x0-epsilon-5epoch-v1'
app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name('hls-lst-datasets-20260914',environment_name='main')
runs_volume = modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main',create_if_missing=True)
locks = modal.Dict.from_name(APP_NAME+'-locks',environment_name='main',create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
 .apt_install('libgl1','libglib2.0-0')
 .pip_install('torch==2.8.0','torchvision==0.23.0','numpy==2.2.6','scipy==1.16.2',
              'PyYAML==6.0.2','zstandard==0.24.0','terratorch==1.2.13')
 .env({'PYTHONPATH':'/code/efdiff_hlc:/data','HF_HOME':'/runs/huggingface',
       'HF_XET_CACHE':'/tmp/hf-xet','OMP_NUM_THREADS':'1'})
 .add_local_dir(HERE,'/code/efdiff_hlc',ignore=['__pycache__','*.pyc'])
 .add_local_dir(HERE.parent/'efdiff/upstream','/code/efdiff/upstream',ignore=['.git','__pycache__','*.pyc']))
mounts = {'/data':data_volume,'/runs':runs_volume}


def root():
    out = Path('/runs')/RUN
    out.mkdir(parents=True,exist_ok=True)
    return out


def smoke_dir(variant):
    import hashlib
    from data import HERE
    h = hashlib.sha256()
    for name in ('data.py','fit_stats.py','train.py',f'configs/{variant}.yaml'):
        h.update((HERE/name).read_bytes())
    return root()/'smoke'/h.hexdigest()[:12]/variant


@app.function(image=image,volumes=mounts,cpu=4,memory=8192,timeout=3600)
def preflight():
    import hashlib
    import numpy as np
    from data import HLCPairs,COUNTS,PROTOCOL
    from train import save_json
    verification = json.loads(Path('/data/_transfer/verification.json').read_text())
    assert verification['status'] == 'complete' and verification['sizes_and_nanosecond_timestamps_verified']
    plan = json.loads(Path('/data/_transfer/plan.json').read_text())
    assert hashlib.sha256(Path('/data/_transfer/files.jsonl').read_bytes()).hexdigest() == plan['manifest_sha256']
    sets,tiles,checks = {},{},[]
    for split in ('train','val','test'):
        ds = HLCPairs('/data',split=split)
        sets[split] = set(map(int,ds.lst.indices[:,0]))
        tiles[split] = {r[0] for r in ds.lst._connect().execute('SELECT DISTINCT tile FROM records WHERE split=?',(split,))}
        for i in (0,1,len(ds)//2,len(ds)-1):
            sample = ds[i]
            assert sample['mask'].any() and np.isfinite(sample['synthetic_coarse_c']).all()
            assert sample['synthetic_coarse_c'].shape == (1,7,7)
        checks.append({'split':split,'patch_months':len(ds),'hls_records':len(ds.hls),
                       'tiles':len(tiles[split]),'fingerprint':ds.fingerprint})
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not sets[a]&sets[b] and not tiles[a]&tiles[b]
    result = {'status':'passed','protocol':PROTOCOL,'dataset':'HLC','real_modis_input':False,
              'upload_verification':verification['status'],'splits':checks}
    save_json(root()/'preflight.json',result)
    runs_volume.commit()
    return result


@app.function(image=image,volumes=mounts,cpu=10,memory=16384,timeout=21600,max_containers=8)
def stats_shard(shard:int,shards:int=8):
    from fit_stats import fit_shard
    from train import save_json
    result = fit_shard('/data',shard,shards)
    directory = root()/'stats_shards'
    directory.mkdir(exist_ok=True)
    save_json(directory/f'{shard:02d}.json',result)
    runs_volume.commit()
    return result


@app.function(image=image,volumes=mounts,gpu='H200',cpu=12,memory=65536,
              timeout=28800,max_containers=2,single_use_containers=True)
def gpu_job(variant:str,stage:str='smoke',split:str='val',start:int=0,end:int=0):
    import numpy as np
    from torch.utils.data import DataLoader,Subset
    from data import HLCPairs,PROTOCOL
    from train import run,final_evaluation_chunk,save_json,config
    runs_volume.reload()
    out = root()
    if stage in ('smoke','smoke_resume'):
        ds = HLCPairs('/data',load_hls=False)
        sums = np.zeros((2,2),dtype=np.float64)
        ids = np.linspace(0,len(ds)-1,32,dtype=int).tolist()
        for batch in DataLoader(Subset(ds,ids),batch_size=4,num_workers=4):
            for i,mask in enumerate(batch['mask'].numpy()):
                for j,key in enumerate(('condition_c','residual_c')):
                    v = batch[key][i].numpy()[mask].astype(np.float64)
                    sums[j] += [v.mean(),np.square(v).mean()]
        stats = {'protocol':PROTOCOL,'fit_split':'train','records':32,'scope':'technical smoke only'}
        for key,(mean,second) in zip(('lst','residual'),sums/32):
            stats[key] = {'mean':float(mean),'std':float(np.sqrt(max(second-mean*mean,1e-12)))}
        result = run('/data',smoke_dir(variant),stats,variant=variant,smoke=True,
                     stop_after_updates=2 if stage=='smoke' else None,commit=runs_volume.commit)
        if result['phase'] == 'smoke_passed':
            save_json(out/f'smoke_{variant}.json',dict(result,evidence=str(smoke_dir(variant).relative_to(out))))
            runs_volume.commit()
        return result
    assert json.loads((out/'preflight.json').read_text())['status'] == 'passed'
    assert json.loads((smoke_dir(variant)/'status.json').read_text())['phase'] == 'smoke_passed'
    stats = json.loads((out/'normalization.json').read_text())
    destination = out/variant
    if stage == 'train':
        return run('/data',destination,stats,variant=variant,commit=runs_volume.commit)
    assert stage == 'evaluate' and split in ('val','test')
    return final_evaluation_chunk('/data',destination,stats,variant,split,start,end,commit=runs_volume.commit)


@app.function(image=image,volumes=mounts,cpu=1,memory=4096,timeout=86400,max_containers=1)
def prepare():
    from fit_stats import merge
    from train import save_json
    out = root()
    try:
        save_json(out/'pipeline.json',{'phase':'preflight'})
        runs_volume.commit()
        preflight.remote()
        calls = [stats_shard.spawn(i,8) for i in range(8)]
        smoke_calls = [gpu_job.spawn(v,'smoke') for v in ('x0','epsilon')]
        for call in smoke_calls:
            assert call.get()['phase'] == 'checkpointed'
        resume_calls = [gpu_job.spawn(v,'smoke_resume') for v in ('x0','epsilon')]
        for call in resume_calls:
            assert call.get()['phase'] == 'smoke_passed'
        result = merge([call.get() for call in calls])
        runs_volume.reload()
        save_json(out/'normalization.json',result)
        save_json(out/'pipeline.json',{'phase':'prepared','formal_training_started':False})
        runs_volume.commit()
        return {'phase':'prepared','normalization_records':result['records']}
    except Exception as exc:
        runs_volume.reload()
        save_json(out/'pipeline.json',{'phase':'preparation_failed','error':repr(exc)})
        runs_volume.commit()
        raise


@app.function(image=image,volumes=mounts,cpu=1,memory=4096,timeout=86400,max_containers=2,
              single_use_containers=True)
def drive(variant:str):
    """At most two bounded GPU calls, then a finite continuation until budgets finish."""
    import math
    from data import COUNTS
    from train import save_json,config
    assert variant in ('x0','epsilon')
    key = RUN+':'+variant
    if not locks.put(key,{'active':True},skip_if_exists=True):
        return {'phase':'already_active','variant':variant}
    needs_continuation = False
    try:
        runs_volume.reload()
        out = root()
        assert (out/'normalization.json').exists()
        for _ in range(2):
            runs_volume.reload()
            status_path = out/variant/'status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get('phase') != 'training_complete':
                result = gpu_job.remote(variant,'train')
                print(variant,json.dumps(result),flush=True)
                needs_continuation = True
                continue
            directory = out/variant/'final_evaluation'
            chunk = config(variant)['evaluation']['chunk_samples']
            missing = [(s,a,min(a+chunk,COUNTS[s])) for s in ('val','test')
                       for a in range(0,COUNTS[s],chunk) if not (directory/s/f'{a:07d}.json').exists()]
            if missing:
                s,a,b = missing[0]
                gpu_job.remote(variant,'evaluate',s,a,b)
                needs_continuation = True
                continue
            final = {'phase':'complete','variant':variant,'checkpoint':'fixed_epoch5_ema','splits':{}}
            for split in ('val','test'):
                chunks = [json.loads((directory/split/f'{a:07d}.json').read_text()) for a in range(0,COUNTS[split],chunk)]
                assert chunks[0]['start'] == 0 and chunks[-1]['end'] == COUNTS[split]
                assert all(a['end'] == b['start'] for a,b in zip(chunks,chunks[1:]))
                n = sum(c['valid_pixels'] for c in chunks)
                records = sum(c['records'] for c in chunks)
                assert records == COUNTS[split]
                final['splits'][split] = {'records':records,'valid_pixels':n,
                  'rmse_c':math.sqrt(sum(c['squared_error_sum'] for c in chunks)/n),
                  'mae_c':sum(c['absolute_error_sum'] for c in chunks)/n,
                  'bicubic_rmse_c':math.sqrt(sum(c['baseline_squared_error_sum'] for c in chunks)/n)}
            save_json(out/f'final_{variant}.json',final)
            runs_volume.commit()
            needs_continuation = False
            return final
    except Exception as exc:
        runs_volume.reload()
        save_json(root()/f'failure_{variant}.json',{'error':repr(exc),'phase':'stopped_on_error'})
        runs_volume.commit()
        raise
    finally:
        locks.pop(key,None)
    if needs_continuation:
        call = drive.spawn(variant)
        print('Continuation',variant,call.object_id,flush=True)
        return {'phase':'continuing','variant':variant,'call_id':call.object_id}
