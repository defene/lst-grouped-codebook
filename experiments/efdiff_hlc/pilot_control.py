"""Launch or fetch the finite fixed-subset pilot."""
import argparse
import json
from pathlib import Path
import modal

APP_NAME = 'efdiff-hlc-pilot-20260915'
RUN = 'hlc-synthetic-pilot-16384-512-v1'
parser = argparse.ArgumentParser()
parser.add_argument('action',choices=['start','status','sync'])
args = parser.parse_args()
volume = modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
out = Path(__file__).resolve().parents[2]/'output/efdiff_hlc_pilot_20260915'
out.mkdir(parents=True,exist_ok=True)
if args.action=='start':
    call = modal.Function.from_name(APP_NAME,'launch',environment_name='main').spawn()
    result = {'run_id':RUN,'call_id':call.object_id,'scope':'pilot'}
    (out/'launch.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
else:
    names = ['pipeline.json','preflight.json','normalization.json','comparison.json','REPORT.md']
    names += [f'comparison_{i}.png' for i in range(1,5)]
    names += [f'membership/{n}' for n in ('summary.json','train.csv','val.csv','train.npy','val.npy')]
    for v in ('x0','epsilon'):
        names += [f'{v}/{n}' for n in ('status.json','metrics.jsonl','provenance.json','predictions_epoch5.npz')]
        names += [f'{v}/validation_epoch_{i}.json' for i in range(1,6)]
        names += [f'smoke/{v}/status.json']
    if args.action=='status':
        names = ['pipeline.json','x0/status.json','epsilon/status.json','smoke/x0/status.json','smoke/epsilon/status.json']
    for name in names:
        try:
            raw = b''.join(volume.read_file(f'{RUN}/{name}'))
        except FileNotFoundError:
            continue
        if args.action=='sync':
            target = out/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
        else:
            print(name,raw.decode())
