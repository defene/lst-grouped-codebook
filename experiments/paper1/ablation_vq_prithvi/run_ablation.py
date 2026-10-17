"""Run/resume eight independent, equally budgeted, scratch-trained codecs."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eo_data.locking import build_lock
from eo_data.core import json_write, sha256_file


def now():
    return datetime.now(timezone.utc).isoformat()


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    config_paths = [HERE/'configs'/f'pilot_{m}_{b}_{n}_seed17.json'
                    for m in ('hls','lst') for b in ('conv','prithvi') for n in ('clean','noisy')]
    hashes = {p.name: sha256_file(p) for p in config_paths}
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['config_hashes'] != hashes:
            raise ValueError('Suite configuration changed; use a new output directory')
    else:
        json_write(manifest_path, {'created_utc':now(), 'config_hashes':hashes,
            'purpose':'first_equal_budget_ablation_pilot', 'pretrained_weights':False,
            'steps_per_run':1000, 'batch_per_run':8, 'training_views':1,
            'train_membership':'all 230659 source pairs; first 8000 shuffled pairs consumed per run',
            'training_noise':'0% or 10% QA-shape missing; original valid image is always target',
            'evaluation':'same fixed 256 val pairs, two realizations, clean and 10% missing',
            'checkpoint_selection':'fixed last.pt at step1000 for every comparison',
            'main_losses':'sample-equal normalized MSE + 0.1 gradient MSE + VQ loss; S=C=0',
            'parameter_reference':'VAR VQ-VAE count 108948355 is ceiling only',
            'architecture':'official Prithvi-EO-2.0 encoder/decoder Blocks, reduced dimensions, no time; common VQ bottleneck',
            'independent_test':False, 'seeds':[17]})
    completed = []
    with build_lock(output):
        for config in config_paths:
            name = config.stem
            destination = output/name
            json_write(output/'status.json', {'status':'running','current':name,'completed':completed,'updated_utc':now()})
            status_path = destination/'status.json'
            finished = status_path.exists() and json.loads(status_path.read_text())['status'] == 'complete'
            if not finished:
                command = [sys.executable,'-m','eo_denoise','train','--config',str(config),'--output',str(destination)]
                if (destination/'last.pt').exists():
                    command += ['--resume',str(destination/'last.pt')]
                with (output/(name+'.stdout.log')).open('a',encoding='utf-8') as log:
                    print(json.dumps({'event':'train','name':name,'utc':now()}),flush=True)
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            for label, coverage in [('clean',0),('noisy10',.1)]:
                target = destination/('eval_'+label)
                if (target/'metrics.json').exists():
                    continue
                command = [sys.executable,'-m','eo_denoise','evaluate','--checkpoint',str(destination/'last.pt'),
                           '--output',str(target),'--coverage',str(coverage),'--limit','256','--noise-kind','bank_missing']
                with (output/(name+'.stdout.log')).open('a',encoding='utf-8') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            completed.append(name)
            print(json.dumps({'event':'completed','name':name,'utc':now()}),flush=True)
        json_write(output/'status.json', {'status':'training_and_evaluation_complete','completed':completed,'updated_utc':now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/paper1/ablation_vq_prithvi_20260910')
    args = parser.parse_args()
    try:
        run(args.output.resolve())
    except Exception as error:
        json_write(args.output/'failure.json',{'utc':now(),'error':repr(error)})
        raise
