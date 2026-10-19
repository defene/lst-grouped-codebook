"""Fail-closed array worker: acceptance or one exact-budget formal run."""
import json
import os
from pathlib import Path
import subprocess
import sys
from experiments.paper1.lst_restoration.config import write_json
from experiments.paper1.lst_restoration.runtime import source_hashes, utc, execution_guard


def main(root, mode, index):
    manifest = json.loads((root/'manifest.json').read_text())
    if source_hashes() != manifest['source_hashes']:
        raise RuntimeError('Source changed since freeze')
    import torch
    if torch.cuda.device_count() != 1 or 'L40S' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Exactly one assigned L40S required')
    jobs = manifest['jobs'] + (manifest['audits'] if mode == 'preflight' else [])
    job = jobs[index]
    name, config_path = job['name'], Path(job['config'])
    output = root/'checks'/name if mode == 'preflight' else root/'runs'/name
    write_json(output/'environment.json', {'utc':utc(), 'torch':str(torch.__version__),
        'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0),
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'), 'mode':mode})
    def cli(*args):
        subprocess.run([sys.executable,'-B','-u','-m','experiments.paper1.lst_restoration',*map(str,args)],check=True)
    if mode == 'preflight':
        parts = ['components','data-budget','model'] if job['formal'] else ['model']
        for part in parts:
            cli('preflight','--config',config_path,'--part',part,'--device','cuda:0','--output',output)
            filename = {'components':'components','data-budget':'data_budget','model':'model'}[part]
            report = json.loads((output/f'{filename}.json').read_text())
            if report['status'] not in ('passed','synthetic_model_checks_complete_other_gates_pending'):
                raise RuntimeError(f"{name}: {part} failed; inspect {output}")
            if part == 'model' and not all(report['gates'][g] for g in
                    ('model_contract','backward_finite','parameter_rate_audit','resource_profile')):
                raise RuntimeError(f'{name}: incomplete model acceptance')
        write_json(output/'accepted.json', {'status':'passed','utc':utc()})
        return
    # The entire 21-element acceptance array must have succeeded first.
    for row in manifest['jobs'] + manifest['audits']:
        if json.loads((root/'checks'/row['name']/'accepted.json').read_text())['status'] != 'passed':
            raise RuntimeError('Stage1 group acceptance incomplete')
    reports = root/'checks'/name
    if job['model'] == 'vq':
        channels = 'R' if job['representation'] == 'R' else 'G'
        aligned = [reports/'model.json'] + [root/'checks'/f'aux_{kind}_{channels}'/'model.json' for kind in ('fsq','lfq','bsq')]
        cli('alignment-audit','--reports',*aligned,'--output',output/'alignment.json')
    cli('assemble-preflight','--config',config_path,'--reports',reports/'components.json',
        reports/'data_budget.json',reports/'model.json','--output',output/'preflight.json')
    config = json.loads(config_path.read_text())
    config['execution']['enabled'] = True
    execution_guard(config)
    write_json(output/'config.enabled.json', config)
    write_json(output/'launch_status.json', {'status':'accepted_starting','utc':utc()})
    cli('run','--config',output/'config.enabled.json','--output',output/'training')
    write_json(output/'launch_status.json', {'status':'training_and_validation_complete','utc':utc()})


if __name__ == '__main__':
    root, mode, index = Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3])
    try:
        main(root, mode, index)
    except BaseException as exc:
        write_json(root/'failures'/f'{mode}_{index}.json', {'utc':utc(),'error':repr(exc)})
        raise
