"""Submit/read the deployed finite experiment; no recurring automation."""
import argparse
import json
from pathlib import Path
import modal

APP_NAME = 'efdiff-hlc-synthetic-20260915'
RUN = 'hlc-synthetic-x0-epsilon-5epoch-v1'
parser = argparse.ArgumentParser()
parser.add_argument('action',choices=['prepare','start','status','sync'])
args = parser.parse_args()
volume = modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
out = Path(__file__).resolve().parents[2]/'output/efdiff_hlc_synthetic_20260915'
out.mkdir(parents=True,exist_ok=True)
if args.action in ('prepare','start'):
    if args.action == 'prepare':
        calls = {'prepare':modal.Function.from_name(APP_NAME,'prepare',environment_name='main').spawn().object_id}
    else:
        state = json.loads(b''.join(volume.read_file(f'{RUN}/pipeline.json')))
        assert state['phase'] == 'prepared',state
        calls = {v:modal.Function.from_name(APP_NAME,'drive',environment_name='main').spawn(v).object_id for v in ('x0','epsilon')}
    (out/f'{args.action}_calls.json').write_text(json.dumps(calls,indent=2))
    print(json.dumps(calls))
else:
    names = ['pipeline.json','preflight.json','normalization.json','smoke_x0.json','smoke_epsilon.json']
    for v in ('x0','epsilon'):
        names += [f'{v}/status.json',f'{v}/metrics.jsonl',f'{v}/provenance.json',f'failure_{v}.json',f'final_{v}.json']
    for name in names:
        try:
            raw = b''.join(volume.read_file(f'{RUN}/{name}'))
        except FileNotFoundError:
            continue
        if args.action == 'sync':
            target = out/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
        elif name.endswith(('metrics.jsonl','provenance.json')) or name == 'preflight.json':
            continue
        elif name == 'normalization.json':
            j = json.loads(raw)
            print(name,json.dumps({k:j[k] for k in ('records','lst','residual')}))
        else:
            print(name,raw.decode())
