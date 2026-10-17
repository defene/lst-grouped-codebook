"""Submit or inspect the finite full-training-set one-epoch comparison."""
import argparse
import json
from pathlib import Path
import modal

APP='efdiff-hlc-full-epoch1-20260916'
RUN='hlc-synthetic-full-epoch1-20260916-v1'
parser=argparse.ArgumentParser()
parser.add_argument('action',choices=['start','status','sync'])
args=parser.parse_args()
out=Path(__file__).resolve().parents[2]/'output/efdiff_hlc_full_epoch1_20260916'
out.mkdir(parents=True,exist_ok=True)
volume=modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
if args.action=='start':
    call=modal.Function.from_name(APP,'launch',environment_name='main').spawn()
    value={'run_id':RUN,'launch_call_id':call.object_id}
    (out/'launch.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))
else:
    names=['pipeline.json','preflight.json','normalization.json','comparison.json','REPORT.md']
    for v in ('x0','epsilon'):
        names += [f'{v}/status.json',f'{v}/validation_epoch_1.json',f'{v}/provenance.json',
                  f'{v}/metrics.jsonl',f'{v}/predictions_epoch1.npz',f'smoke/{v}/status.json',
                  f'failure_{v}.json',f'continuation_{v}.json']
    if args.action=='status':
        names=['pipeline.json']+[f'{v}/status.json' for v in ('x0','epsilon')]+[
            f'smoke/{v}/status.json' for v in ('x0','epsilon')]+[f'failure_{v}.json' for v in ('x0','epsilon')]
    else:
        names += [f'comparison_{i}.png' for i in range(1,5)]
    for name in names:
        try:
            raw=b''.join(volume.read_file(f'{RUN}/{name}'))
        except FileNotFoundError:
            continue
        if args.action=='sync':
            target=out/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
        else:
            print(name,raw.decode())
