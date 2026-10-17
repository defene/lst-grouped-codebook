"""One A100 only, inference smoke then resumable fixed full-test evaluation."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
APP='hlc-unet-mmdit-fulltest-20261015'
RUN='hlc-unet-mmdit-fulltest-20261015-v1'
app=modal.App(APP)
vol=modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
data=modal.Volume.from_name('hls-lst-datasets-20260914',environment_name='main')
locks=modal.Dict.from_name(APP+'-locks',environment_name='main',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.12')
 .apt_install('libgl1','libglib2.0-0')
 .pip_install('torch==2.8.0','torchvision==0.23.0','numpy==2.2.6','scipy==1.16.2','PyYAML==6.0.2','zstandard==0.24.0','terratorch==1.2.13')
 .env({'PYTHONPATH':'/code/efdiff_hlc:/data','OMP_NUM_THREADS':'1'})
 .add_local_dir(HERE,'/code/efdiff_hlc',ignore=['__pycache__','*.pyc'])
 .add_local_dir(HERE.parent/'efdiff/upstream','/code/efdiff/upstream',ignore=['.git','__pycache__','*.pyc']))
if modal.is_local():image=image.add_local_dir(HERE.parents[1]/'output/hlc_unet_mmdit_test_20261015/metadata','/test_metadata')

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)

@app.function(image=image,volumes={'/runs':vol,'/data':data},gpu='A100-40GB',cpu=16,memory=98304,timeout=7200,max_containers=1,retries=0,single_use_containers=True)
def job(scope='smoke',segment=0):
    from test_comparison_20261015 import run
    assert scope in ('smoke','test') and segment<50
    vol.reload();root=Path('/runs')/RUN
    save(root/'pipeline.json',dict(phase='running',scope=scope,segment=segment,unix=time.time()));vol.commit()
    try:
        result=run(scope,vol.commit)
        if scope=='smoke':
            assert result['phase']=='complete' and result['records']==44
            next_scope='test'
        elif result['phase']=='checkpointed':next_scope='test'
        else:next_scope=None
        vol.commit()
        if next_scope:
            call=job.spawn(next_scope,segment+1)
            status=dict(phase='queued_continuation',scope=next_scope,segment=segment+1,call_id=call.object_id,result=result,unix=time.time())
        else:status=dict(phase='complete',result=result,unix=time.time())
        save(root/'pipeline.json',status);vol.commit();return status
    except Exception as exc:
        save(root/'pipeline.json',dict(phase='failed',scope=scope,segment=segment,error=repr(exc),unix=time.time()));vol.commit();raise

@app.function(image=image,volumes={'/runs':vol},cpu=1,memory=1024,timeout=180)
def launch():
    vol.reload();root=Path('/runs')/RUN
    if (root/'submission.json').exists():return json.loads((root/'submission.json').read_text())
    if not locks.put(RUN,dict(unix=time.time()),skip_if_exists=True):return dict(phase='already_submitted')
    call=job.spawn('smoke',0);r=dict(phase='submitted',call_id=call.object_id,run=RUN,unix=time.time())
    save(root/'submission.json',r);vol.commit();return r

@app.function(image=image,volumes={'/runs':vol},cpu=1,memory=1024,timeout=180)
def status():
    vol.reload();root=Path('/runs')/RUN
    return {str(p.relative_to(root)):json.loads(p.read_text()) for p in root.rglob('*.json') if p.parent.name!='chunks'}

@app.local_entrypoint()
def main():
    print(json.dumps(launch.remote(),indent=2))
