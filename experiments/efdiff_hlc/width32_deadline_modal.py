"""Finite single-GPU queue: smoke, equal-budget stages, validation, then stop."""
import json
import time
from pathlib import Path
import modal
from modal_app import image,mounts,runs_volume

APP='efdiff-hlc-width32-single-12h-20260923'
RUN='hlc-width32-single-12h-20260923-v1'
app=modal.App(APP)


def root():
    p=Path('/runs')/RUN
    p.mkdir(parents=True,exist_ok=True)
    return p


def write_pipeline(value):
    from epoch1_train import save_json
    save_json(root()/'pipeline.json',value)
    runs_volume.commit()
    return value


def child(variant,records,smoke=False,stop_after=None):
    import subprocess
    import sys
    out=root()/('smoke' if smoke else 'train')/variant
    command=[sys.executable,'/code/efdiff_hlc/width32_deadline_train.py',
             '--variant',variant,'--records',str(records),'--output',str(out)]
    if smoke: command.append('--smoke')
    if stop_after is not None: command+=['--stop-after',str(stop_after)]
    subprocess.run(command,check=True)
    return json.loads((out/'segment_result.json').read_text())


@app.function(image=image,volumes=mounts,cpu=2,memory=8192,timeout=600,
              max_containers=1,single_use_containers=True)
def report(records:int):
    from width32_deadline_report import build
    runs_volume.reload()
    result=build(root(),records)
    runs_volume.commit()
    return result


@app.function(image=image,volumes=mounts,gpu='H200',cpu=24,memory=65536,timeout=7200,
              max_containers=1,single_use_containers=True)
def job(mode:str='smoke',stage:int=0,variant_index:int=0):
    from epoch1_train import save_json
    from data import HERE
    runs_volume.reload()
    plan=json.loads((HERE/'width32_12h_plan.json').read_text())
    save_json(root()/'plan.json',plan)
    if time.time()>=plan['deadline_unix']-1800:
        return write_pipeline({'phase':'deadline_stopped','test_used':False,'deadline':plan['deadline_utc']})
    try:
        if mode=='smoke':
            write_pipeline({'phase':'smoke','deadline':plan['deadline_utc'],'gpu_count':1})
            for variant in ('x0','epsilon'):
                first=child(variant,131,smoke=True,stop_after=2)
                assert first['phase'] in ('checkpointed','smoke_passed')
                if first['phase']=='checkpointed':
                    assert first['presented']==128 and first['step']==2
                result=child(variant,131,smoke=True)
                assert result['phase']=='smoke_passed' and result['updates']==3 and result['training_records']==131
            call=job.spawn('train',0,0)
            return write_pipeline({'phase':'training_queued','smoke_resume':'both passed',
                'call_id':call.object_id,'deadline':plan['deadline_utc'],'plan':plan})
        assert mode=='train'
        for variant in ('x0','epsilon'):
            assert json.loads((root()/'smoke'/variant/'stage_131.json').read_text())['phase']=='smoke_passed'
        records=plan['stage_records'][stage]
        variant=('x0','epsilon')[variant_index]
        if stage==0 and variant_index==0:
            assert time.time()<plan['deadline_unix']-plan['predicted_first_stage_pair_seconds']-1800
        write_pipeline({'phase':'training','stage':stage,'variant':variant,'target_records':records,
            'deadline':plan['deadline_utc'],'gpu_count':1})
        result=child(variant,records)
        if result['phase']=='deadline_stop':
            return write_pipeline(dict(result,deadline=plan['deadline_utc']))
        if result['phase']=='checkpointed':
            call=job.spawn('train',stage,variant_index)
            return write_pipeline(dict(result,phase='training_queued',stage=stage,call_id=call.object_id))
        assert result['phase']=='stage_complete'
        if variant_index==0:
            call=job.spawn('train',stage,1)
            return write_pipeline({'phase':'training_queued','stage':stage,'variant':'epsilon','call_id':call.object_id})
        comparison=report.remote(records)
        runs_volume.reload()
        if stage+1<len(plan['stage_records']):
            extra_updates=(plan['stage_records'][stage+1]-records)/64
            needed=extra_updates*plan['conservative_pair_seconds_per_update']+1800
            if time.time()+needed<plan['deadline_unix']:
                call=job.spawn('train',stage+1,0)
                return write_pipeline({'phase':'training_queued','stage':stage+1,'variant':'x0',
                    'call_id':call.object_id,'latest_comparison':comparison})
        return write_pipeline({'phase':'complete','latest_comparison':comparison,
            'deadline':plan['deadline_utc'],'test_used':False,'gpu_count':1})
    except Exception as exc:
        write_pipeline({'phase':'failed','mode':mode,'stage':stage,'variant_index':variant_index,'error':repr(exc)})
        raise


@app.function(image=image,volumes=mounts,cpu=1,memory=1024,timeout=120,max_containers=1)
def status():
    runs_volume.reload()
    out={}
    for name in ('pipeline.json','plan.json','latest_comparison.json'):
        p=root()/name
        if p.exists():out[name]=json.loads(p.read_text())
    for v in ('x0','epsilon'):
        p=root()/'train'/v/'status.json'
        if p.exists():out[v]=json.loads(p.read_text())
    return out
