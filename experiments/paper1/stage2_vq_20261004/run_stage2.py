"""Bound preflight and formal array runner for both new VQ experiment families."""
import json
import os
from pathlib import Path
import subprocess
import sys
from experiments.paper1.lst_restoration.config import write_json
from experiments.paper1.lst_restoration.runtime import source_hashes, utc, execution_guard


def main(root, mode, index):
    manifest = json.loads((root/'manifest.json').read_text())
    assert source_hashes() == manifest['source_hashes'], 'Source changed after freeze'
    import torch
    assert torch.cuda.device_count() == 1 and 'L40S' in torch.cuda.get_device_name(0)
    job = manifest['jobs'][index]
    name, config_path = job['name'], Path(job['config'])
    output = root/'checks'/name if mode == 'preflight' else root/'runs'/name
    config = json.loads(config_path.read_text())
    write_json(output/'environment.json', dict(utc=utc(), torch=str(torch.__version__),
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), slurm_job_id=os.environ['SLURM_JOB_ID']))
    def cli(*args):
        subprocess.run([sys.executable,'-B','-u','-m','experiments.paper1.lst_restoration',*map(str,args)],check=True)
    if mode == 'preflight':
        for part, filename in [('components','components'),('data-budget','data_budget'),('model','model')]:
            cli('preflight','--config',config_path,'--part',part,'--device','cuda:0','--output',output)
            r = json.loads((output/(filename+'.json')).read_text())
            assert r['status'] in ('passed','synthetic_model_checks_complete_other_gates_pending'), r
            if part == 'model':
                assert all(r['gates'][g] for g in ('model_contract','backward_finite','parameter_rate_audit','resource_profile')), r
        from experiments.paper1.lst_restoration.stage2_audit import run
        run(config, output/'stage2.json')
        write_json(output/'accepted.json',dict(status='passed',utc=utc()))
        return
    for row in manifest['jobs']:
        assert json.loads((root/'checks'/row['name']/'accepted.json').read_text())['status']=='passed'
    reports = root/'checks'/name
    cli('assemble-preflight','--config',config_path,'--reports',
        *[reports/(n+'.json') for n in ('components','data_budget','model','stage2')],
        '--output',output/'preflight.json')
    config['execution']['enabled']=True
    execution_guard(config)
    write_json(output/'config.enabled.json',config)
    write_json(output/'launch_status.json',dict(status='accepted_starting',utc=utc()))
    cli('run','--config',output/'config.enabled.json','--output',output/'training')
    write_json(output/'launch_status.json',dict(status='training_and_validation_complete',utc=utc()))


if __name__=='__main__':
    root,mode,index=Path(sys.argv[1]).resolve(),sys.argv[2],int(sys.argv[3])
    try:
        main(root,mode,index)
    except BaseException as exc:
        write_json(root/'failures'/f'{mode}_{index}.json',dict(utc=utc(),error=repr(exc)))
        raise
