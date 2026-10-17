"""One bounded H200, sequential fresh processes; no training chain or extra GPUs."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import modal
from modal_app import image,mounts,runs_volume

APP='efdiff-hlc-dit-flow-epsilon-probe-20260930'
RUN='hlc-dit-flow-epsilon-probe-20260930-v1'
app=modal.App(APP)


@app.function(image=image,volumes=mounts,gpu='H200',cpu=24,memory=98304,
              timeout=1200,max_containers=1,single_use_containers=True,retries=0)
def probe():
    import hashlib
    import torch
    from torch.utils.data import DataLoader
    from center_data import HLCPairs,EpochSampler,check_stats
    from data import HERE
    out=Path('/runs')/RUN;out.mkdir(parents=True,exist_ok=True)
    def save(name,data):
        (out/name).write_text(json.dumps(data,indent=2)+'\n');runs_volume.commit()
    if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    save('pipeline.json',dict(phase='loading_real_train_data',gpu=gpu,unix=time.time()))
    try:
        plan=json.loads((HERE/'dit_plan.json').read_text())
        stats=json.loads(Path(plan['normalization_path']).read_text())
        ds=HLCPairs('/data',stats=stats,augment=False,seed=plan['seed']);check_stats(stats,ds)
        batches=list(DataLoader(ds,sampler=EpochSampler(ds,0,limit=132),batch_size=64,num_workers=4))
        assert [len(b['HR']) for b in batches]==[64,64,4]
        data_path='/tmp/dit_probe_batches.pt';torch.save(batches,data_path)
        sources={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
                 ['dit_model.py','dit_objectives.py','dit_probe.py','dit_probe_modal.py','dit_plan.json','center_data.py']}
        save('provenance.json',dict(plan=plan,gpu=gpu,fingerprint=ds.fingerprint,source_sha256=sources,
              source_indices=[int(v) for b in batches for v in b['index']],
              stats_sha256=hashlib.sha256(json.dumps(stats,sort_keys=True).encode()).hexdigest(),test_used=False))
        del batches,ds
        results=[]
        for ckpt in [True,False]:
            for variant in ['flow','epsilon']:
                name=f'{variant}_checkpoint{int(ckpt)}'
                save('pipeline.json',dict(phase='probing',case=name,gpu=gpu,unix=time.time()))
                with (out/f'{name}.log').open('w') as log:
                    env=dict(os.environ,PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                    child=subprocess.Popen([sys.executable,str(HERE/'dit_probe.py'),'--variant',variant,
                        '--checkpointing',str(int(ckpt)),'--data',data_path,'--result',str(out/f'{name}.json')],
                        stdout=log,stderr=subprocess.STDOUT,env=env)
                    try:
                        while child.poll() is None:
                            snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
                            save('gpu.json',dict(unix=time.time(),gpu=snapshot,child_pid=child.pid,case=name))
                            time.sleep(5)
                    finally:
                        if child.poll() is None:child.kill();child.wait()
                    if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}: '+(out/f'{name}.log').read_text()[-3000:])
                result=json.loads((out/f'{name}.json').read_text());results.append(result)
                save('progress.json',dict(results=results,gpu=gpu))
        passed=[r for r in results if r['phase']=='passed']
        assert len({r['initial_dit_sha256'] for r in passed})<=1
        assert len({r['frozen_encoder_sha256'] for r in passed})<=1
        result=dict(phase='complete',gpu=gpu,results=results,single_gpu_only=True,formal_training_launched=False)
        save('result.json',result);save('pipeline.json',dict(phase='complete',gpu=gpu,unix=time.time()))
        return result
    except Exception as exc:
        save('pipeline.json',dict(phase='failed',error=repr(exc),gpu=gpu,unix=time.time()))
        raise
