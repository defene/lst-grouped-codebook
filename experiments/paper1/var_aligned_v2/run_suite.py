"""Run the eight independent codecs serially; persist status and stop on any failure."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from eo_data.core import json_write,sha256_file
from eo_data.locking import build_lock
from eo_denoise.engine import source_hashes

def now():return datetime.now(timezone.utc).isoformat()

def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.phase=='full':
        gate=json.loads(args.pilot_gate.read_text())
        if gate.get('status')!='passed' or gate.get('source_hashes')!=source_hashes():
            raise ValueError('Full training requires a passed small-test gate for the current source')
    configs=[HERE/'configs'/args.phase/f'{args.phase}_{m}_{b}_{n}_seed17.json'
             for m in ('hls','lst') for b in ('conv','prithvi') for n in ('clean','noisy')]
    identity=dict(config_hashes={p.name:sha256_file(p) for p in configs},source_hashes=source_hashes())
    manifest=output/'manifest.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        if any(previous[k]!=identity[k] for k in identity):raise ValueError('Suite identity changed; use a new output')
    else:
        json_write(manifest,dict(**identity,created_utc=now(),phase=args.phase,pretrained_weights=False,
            steps_per_run=128 if args.phase=='pilot' else 10813,
            effective_batch=16 if args.phase=='pilot' else 64,
            train_pairs=64 if args.phase=='pilot' else 230659,
            final_val_pairs=64 if args.phase=='pilot' else 11757,
            architecture_reference='FoundationVision/VAR@78b95394fc5896192e3a003e4b295f8ea743c48f',
            model_comparison='actual VAR VQVAE vs same residual quantizer with spatial Prithvi encoder/decoder',
            checkpoint_selection='fixed final last.pt; full validation is for final comparison',
            pilot_gate=str(args.pilot_gate.resolve()) if args.phase=='full' else None))
    completed=[]
    with build_lock(output):
        for config in configs:
            name=config.stem;destination=output/name
            for stage in ('train','eval_clean','eval_noisy10'):
                state=dict(status='running',phase=args.phase,current=name,stage=stage,completed=completed,
                    supervisor_pid=os.getpid(),updated_utc=now())
                if stage=='train':
                    status=destination/'status.json'
                    if status.exists() and json.loads(status.read_text())['status']=='complete':continue
                    command=[sys.executable,'-m','eo_denoise','train','--config',str(config),'--output',str(destination)]
                    if (destination/'last.pt').exists():command+=['--resume',str(destination/'last.pt')]
                else:
                    target=destination/stage
                    if (target/'metrics.json').exists():continue
                    command=[sys.executable,'-m','eo_denoise','evaluate','--checkpoint',str(destination/'last.pt'),
                        '--output',str(target),'--coverage','0' if stage=='eval_clean' else '.1',
                        '--limit','64' if args.phase=='pilot' else '11757','--noise-kind','bank_missing']
                json_write(output/'status.json',state)
                with (output/(name+'.stdout.log')).open('a',encoding='utf-8') as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    state['child_pid']=child.pid;json_write(output/'status.json',state)
                    print(json.dumps(state),flush=True)
                    returncode=child.wait()
                    if returncode:raise RuntimeError(f'{name}/{stage} exited {returncode}; see {name}.stdout.log')
            completed.append(name)
        json_write(output/'status.json',dict(status='training_and_evaluation_complete',phase=args.phase,
            completed=completed,updated_utc=now()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','full'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot-gate',type=Path,default=ROOT/'runs/paper1/var_aligned_v2_pilot/gate.json')
    args=p.parse_args()
    try:run(args)
    except BaseException as error:
        state=dict(status='failed',error=repr(error),updated_utc=now())
        json_write(args.output/'failure.json',state);json_write(args.output/'status.json',state)
        raise
