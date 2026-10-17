"""One A100: bounded smoke then fresh v-pred baseline; no H200 or duplicate launch."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import modal
HERE = Path(__file__).resolve().parent
runs_volume = modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
data_volume = modal.Volume.from_name('hls-lst-datasets-20260914',environment_name='main')
mounts = {'/data':data_volume,'/runs':runs_volume}
base_image = (modal.Image.debian_slim(python_version='3.12')
 .apt_install('libgl1','libglib2.0-0')
 .pip_install('torch==2.8.0','torchvision==0.23.0','numpy==2.2.6','scipy==1.16.2',
              'PyYAML==6.0.2','zstandard==0.24.0','terratorch==1.2.13')
 .env({'PYTHONPATH':'/code/efdiff_hlc:/data','HF_HOME':'/runs/huggingface',
       'HF_XET_CACHE':'/tmp/hf-xet','OMP_NUM_THREADS':'1'})
 .add_local_dir(HERE,'/code/efdiff_hlc',ignore=['__pycache__','*.pyc'])
 .add_local_dir(HERE.parent/'efdiff/upstream','/code/efdiff/upstream',ignore=['.git','__pycache__','*.pyc']))

HERE = Path(__file__).resolve().parent
APP = 'efdiff-hlc-raw-mmdit-v-a100-20261014'
RUN = 'hlc-raw-mmdit-v-b128-fullval-epoch2-20261014-v1'
app = modal.App(APP)
locks = modal.Dict.from_name(APP+'-locks',environment_name='main',create_if_missing=True)
if modal.is_local():
    assets = HERE.parents[1]/'output/efdiff_hlc_dit_timegeo_20261004'
    image = (base_image.add_local_dir(assets/'metadata','/timegeo_assets/metadata')
             .add_local_file(assets/'normalization.json','/timegeo_assets/normalization.json'))
else:
    image = base_image


def root():
    p=Path('/runs')/RUN
    p.mkdir(parents=True,exist_ok=True)
    return p


@app.function(image=image,volumes=mounts,gpu='A100-40GB',cpu=24,memory=98304,
              timeout=7200,max_containers=1,single_use_containers=True,retries=0)
def job(scope:str='smoke',segment:int=0):
    from epoch1_train import save_json
    assert scope in ('smoke','train') and 0<=segment<100
    runs_volume.reload()
    assert (root()/'preflight.json').exists()
    if scope=='train':
        smoke=json.loads((root()/'smoke/v/complete.json').read_text())
        assert smoke['phase']=='smoke_passed' and smoke['training_records']==388 and smoke['tail_records']==4
    out=root()/scope/'v';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        return dict(phase='already_complete',scope=scope)
    child=None
    try:
        cmd=[sys.executable,'/code/efdiff_hlc/raw_mmdit_train.py','--variant','v','--output',str(out),
             '--root','/data','--assets',str(root()/'assets')]
        if scope=='smoke':
            cmd+=['--smoke']
            if segment==0:cmd+=['--stop-after','2']
        save_json(out/'pipeline.json',dict(phase='starting',scope=scope,segment=segment,unix=time.time()))
        runs_volume.commit()
        with (out/'console.log').open('a',buffering=1) as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))
            while child.poll() is None:
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True).strip()
                processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                    '--format=csv,noheader,nounits'],text=True).strip()
                save_json(out/'gpu.json',dict(unix=time.time(),gpu=gpu,processes=processes,child_pid=child.pid,
                                             scope=scope,segment=segment))
                runs_volume.commit()
                time.sleep(20)
            if child.returncode!=0:raise RuntimeError(f'Trainer exited {child.returncode}; see console.log')
        result=json.loads((out/'segment_result.json').read_text())
        if result['phase']=='checkpointed':
            next_scope=scope
        elif result['phase']=='smoke_passed':
            assert result['training_records']==388 and result['updates']==4
            assert result['tail_records']==4 and result['sweep']['final_5']['records']==132
            next_scope='train'
        elif result['phase']=='complete':
            assert result['training_records']==5217544 and result['updates']==40764
            next_scope=None
        else:raise ValueError(f'Unexpected segment result: {result}')
        if next_scope:
            assert segment+1<100, 'Continuation limit reached; inspect before any further work'
            call=job.spawn(next_scope,segment+1)
            status=dict(phase='queued_continuation',call_id=call.object_id,next_scope=next_scope,
                        result=result,unix=time.time())
            if next_scope!=scope:
                save_json(root()/'train_launch.json',dict(phase='submitted',calls={'v':call.object_id},unix=time.time()))
        else:status=dict(phase='complete',result=result,unix=time.time())
        save_json(out/'pipeline.json',status);runs_volume.commit()
        return status
    except Exception as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        save_json(out/'pipeline.json',dict(phase='failed',scope=scope,segment=segment,error=repr(exc),unix=time.time()))
        runs_volume.commit()
        raise


@app.function(image=image,volumes=mounts,cpu=4,memory=16384,timeout=1200,max_containers=1)
def launch():
    import numpy as np
    from raw_mmdit_data import HLCPairs, check_stats, EpochSampler
    from epoch1_train import save_json
    from timegeo_metadata import sha256
    runs_volume.reload()
    if (root()/'submission.json').exists():
        return json.loads((root()/'submission.json').read_text())
    plan=json.loads((Path('/code/efdiff_hlc')/'raw_mmdit_train_plan.json').read_text())
    assert plan['run_id']==RUN and plan['variants']==['v'] and plan['epochs']==2 and plan['batch_size']==128
    if not locks.put(RUN,dict(unix=time.time()),skip_if_exists=True):
        return dict(phase='already_submitted_or_launch_in_progress',run_id=RUN)
    assets=root()/'assets';assets.mkdir(exist_ok=True)
    shutil.copytree('/timegeo_assets/metadata',assets/'metadata',dirs_exist_ok=True)
    shutil.copyfile('/timegeo_assets/normalization.json',assets/'normalization.json')
    stats=json.loads((assets/'normalization.json').read_text())
    datasets={};samples=[]
    for split in ('train','val'):
        ds=HLCPairs('/data',split=split,stats=stats,augment=False,seed=plan['seed'],metadata_dir=assets/'metadata')
        if split=='train':check_stats(stats,ds)
        for i in (0,len(ds)-1):
            sample=ds[i]
            assert sample['HLS'].shape==(6,224,224) and sample['HLS_valid'].shape==(1,224,224)
            assert np.isfinite(sample['HLS']).all() and np.isfinite(sample['geo_latlon']).all()
            samples.append(dict(split=split,index=i,month=sample['month'],geo=sample['geo_latlon'].tolist(),
                                hls_valid_fraction=float(sample['HLS_valid'].mean())))
        datasets[split]=ds
    assert len(datasets['train'])==2608772 and len(datasets['val'])==141711
    assert not set(map(int,datasets['train'].lst.indices[:,0])) & set(map(int,datasets['val'].lst.indices[:,0]))
    orders={str(ep):hashlib.sha256(EpochSampler(datasets['train'],ep).order.astype('<i4').tobytes()).hexdigest() for ep in range(2)}
    save_json(root()/'preflight.json',dict(phase='passed',plan=plan,samples=samples,
        fingerprints={k:v.fingerprint for k,v in datasets.items()},train_order_sha256=orders,
        test_used=False,unix=time.time(),normalization_sha256=sha256(assets/'normalization.json')))
    runs_volume.commit()
    call=job.spawn('smoke',0)
    result=dict(phase='submitted',scope='smoke_then_train',calls={'v':call.object_id},run_id=RUN,unix=time.time())
    save_json(root()/'submission.json',result);save_json(root()/'smoke_launch.json',result)
    runs_volume.commit()
    return result


@app.function(image=image,volumes=mounts,cpu=1,memory=2048,timeout=180,max_containers=1)
def status():
    runs_volume.reload()
    result={}
    for name in ('submission.json','preflight.json','smoke_launch.json','train_launch.json'):
        p=root()/name
        if p.exists():result[name]=json.loads(p.read_text())
    for scope in ('smoke','train'):
        result[scope]={'v':{}}
        for name in ('status.json','pipeline.json','gpu.json','complete.json'):
            p=root()/scope/'v'/name
            if p.exists():result[scope]['v'][name]=json.loads(p.read_text())
        for p in sorted((root()/scope/'v').glob('*.progress.json')):
            result[scope]['v'][p.name]=json.loads(p.read_text())
        p=root()/scope/'v'/'console.log'
        if p.exists():
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-5000));result[scope]['v']['log_tail']=f.read().decode(errors='replace')
    return result
