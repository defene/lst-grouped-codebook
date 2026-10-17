"""One finite A100 inference job; no training or automatic continuation."""
import json
import traceback
from pathlib import Path
import modal
from modal_app import image, mounts, runs_volume

APP = 'efdiff-hlc-dit-aligned-val512-20261002'
RUN = 'hlc-dit-aligned-val512-20261002-v1'
app = modal.App(APP)
if modal.is_local():
    image = image.add_local_file(Path(__file__).resolve().parents[2]/'output/efdiff_hlc_width_comparison_20260926/w64_x0.npz','/code/alignment_reference.npz')

@app.function(image=image, volumes=mounts, gpu='A100-40GB', cpu=16, memory=65536,
              max_containers=1, timeout=7200, retries=0, scaledown_window=2)
def job():
    from dit_aligned_eval import run
    try:
        return run(runs_volume.commit)
    except Exception:
        out = Path('/runs')/RUN
        out.mkdir(exist_ok=True)
        (out/'error.txt').write_text(traceback.format_exc())
        runs_volume.commit()
        raise

@app.local_entrypoint()
def launch():
    out = Path(__file__).resolve().parents[2]/'output/efdiff_hlc_dit_aligned_val512_20261002'
    out.mkdir(exist_ok=True)
    guard = out/'submission.json'
    assert not guard.exists(), 'Already submitted: inspect existing call instead of spawning'
    call = job.spawn()
    guard.write_text(json.dumps(dict(app=APP,run=RUN,call_id=call.object_id),indent=2))
    print(guard.read_text())
