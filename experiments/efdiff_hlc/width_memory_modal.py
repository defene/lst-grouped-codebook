"""One bounded two-H200 allocation, four fresh subprocess comparisons."""
import json
from pathlib import Path
import modal
from modal_app import image, mounts

app = modal.App('efdiff-hlc-width-memory-20260923')


@app.function(image=image, volumes=mounts, gpu='H200:2', cpu=16, memory=65536,
              timeout=900, max_containers=1, single_use_containers=True)
def probe():
    import subprocess
    import sys
    import time
    started = time.perf_counter()
    out = Path('/tmp/width_memory')
    for variant in ('x0', 'epsilon'):
        for width in (64,32):
            subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=2', '/code/efdiff_hlc/width_memory_probe.py',
                '--variant',variant, '--width',str(width), '--output',str(out)], check=True)
    return {'allocation_seconds':time.perf_counter()-started,
            'results':[json.loads(p.read_text()) for p in sorted(out.glob('*.json'))]}


@app.local_entrypoint()
def main():
    result = probe.remote()
    out = Path('output/efdiff_hlc_width_memory_20260923')
    out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'saved':str(out/'results.json'), 'allocation_seconds':result['allocation_seconds']}))
