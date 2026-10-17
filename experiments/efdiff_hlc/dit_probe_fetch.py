"""Read/download the one existing probe call. Never submits a GPU job."""
import json
import os
from pathlib import Path
os.environ['MODAL_PROFILE']='runlongyu-target'
os.environ['MODAL_ENVIRONMENT']='main'
import modal

OUT=Path(__file__).resolve().parents[2]/'output/efdiff_hlc_dit_flow_epsilon_probe_20260930'
launch=json.loads((OUT/'launch.json').read_text())
fn=modal.Function.from_name(launch['app'],'probe',environment_name='main')
s=fn.get_current_stats()
result={'stats':dict(backlog=s.backlog,runners=s.num_total_runners,running=s.num_running_inputs)}
try:
    returned=modal.FunctionCall.from_id(launch['call_id']).get(timeout=0)
    result['call_phase']=returned['phase']
except TimeoutError:result['call_phase']='pending'
except Exception as exc:result['call_error']=repr(exc)
volume=modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
files=['pipeline.json','gpu.json','progress.json','provenance.json','result.json']
files += [f'{v}_checkpoint{c}.{ext}' for v in ('flow','epsilon') for c in (0,1) for ext in ('json','log')]
for name in files:
    try:raw=b''.join(volume.read_file(launch['run']+'/'+name))
    except FileNotFoundError:continue
    (OUT/name).write_bytes(raw)
    if name in ('pipeline.json','gpu.json'):result[name]=json.loads(raw)
    if name.endswith('.log'):
        result[name]=raw.decode(errors='replace')[-500:]
    if name=='progress.json':
        result['completed_cases']=[{k:r[k] for k in ('variant','checkpointing','phase','mean_update_seconds','peak_train_allocated_GB') if k in r} for r in json.loads(raw)['results']]
(OUT/'fetch_status.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
