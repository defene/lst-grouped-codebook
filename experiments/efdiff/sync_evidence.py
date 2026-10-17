"""Download compact run evidence; leave multi-GB model checkpoints on Modal."""
from pathlib import Path
import modal

root = Path(__file__).resolve().parents[2]/'output/efdiff_hlp_v2_20260915'
volume = modal.Volume.from_name('efdiff-hlp-v2-runs', environment_name='main')
run = 'hlp-v2-x0-5epoch-20260915'
names = ['pipeline.json', 'preflight.json', 'normalization.json',
         'smoke/status.json', 'smoke/provenance.json', 'smoke/validation_epoch_1.json',
         'formal/status.json', 'formal/provenance.json', 'formal/metrics.jsonl', 'formal/pip-freeze.txt']
names += [f'formal/validation_epoch_{epoch}.json' for epoch in range(1,6)]
for name in names:
    try:
        raw = b''.join(volume.read_file(f'{run}/{name}'))
    except FileNotFoundError:
        continue
    target = root/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    print(f'{name}: {len(raw)} bytes')
