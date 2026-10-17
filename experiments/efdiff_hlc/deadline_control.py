"""Submit or inspect the finite deadline-constrained run."""
import argparse
import json
from pathlib import Path
import modal

APP = 'efdiff-hlc-epoch1-deadline-20260916'
RUN = 'hlc-synthetic-full-epoch1-ddp2-20260916-v2'
parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['start','status','sync'])
args = parser.parse_args()
out = Path(__file__).resolve().parents[2]/'output/efdiff_hlc_epoch1_deadline_20260916'
out.mkdir(parents=True, exist_ok=True)
volume = modal.Volume.from_name('efdiff-hlc-synthetic-runs', environment_name='main')
if args.action == 'start':
    call = modal.Function.from_name(APP,'launch',environment_name='main').spawn()
    value = {'run_id':RUN,'launch_call_id':call.object_id}
    (out/'launch.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))
else:
    names = ['pipeline.json','throughput_gate.json']
    for v in ('x0','epsilon'):
        names += [f'train/{v}/status.json',f'smoke/{v}/status.json',f'failure_{v}.json']
    if args.action == 'sync':
        names += ['normalization.json','probe/epsilon/metrics.jsonl','train/comparison.json','train/REPORT.md']
        names += [f'train/comparison_{i}.png' for i in range(1,5)]
        for v in ('x0','epsilon'):
            names += [f'train/{v}/{f}' for f in ('provenance.json','metrics.jsonl','validation_epoch_1.json')]
    for name in names:
        try:
            raw = b''.join(volume.read_file(f'{RUN}/{name}'))
        except FileNotFoundError:
            continue
        if args.action == 'sync':
            target = out/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        else:
            print(name,raw.decode())
