import json
from pathlib import Path
import modal
from modal_app import image,mounts
app=modal.App('efdiff-hlc-width32-single-probe-20260923')

@app.function(image=image,volumes=mounts,gpu='H200',cpu=24,memory=65536,
              timeout=1200,max_containers=1,single_use_containers=True)
def probe():
    from width32_single_probe import run
    return run()

@app.local_entrypoint()
def main():
    result=probe.remote()
    out=Path('output/efdiff_hlc_width32_12h_20260923')
    out.mkdir(parents=True,exist_ok=True)
    (out/'single_gpu_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'saved':str(out/'single_gpu_benchmark.json')}))
