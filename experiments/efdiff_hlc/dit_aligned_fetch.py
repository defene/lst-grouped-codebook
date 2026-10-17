"""Fetch the existing alignment call; never spawn any compute."""
import os
os.environ['MODAL_PROFILE']='runlongyu-target'
os.environ['MODAL_ENVIRONMENT']='main'
import json
from pathlib import Path
import modal

OUT=Path(__file__).resolve().parents[2]/'output/efdiff_hlc_dit_aligned_val512_20261002'
s=json.loads((OUT/'submission.json').read_text())
result={}
if s.get('function_id'):
    import asyncio
    from modal.client import _Client
    from modal_proto import api_pb2
    async def stats_for_id():
        client=await _Client.from_env()
        stats=await client.stub.FunctionGetCurrentStats(api_pb2.FunctionGetCurrentStatsRequest(function_id=s['function_id']))
        return dict(backlog=stats.backlog,runners=stats.num_total_tasks,running=stats.num_running_inputs)
    result['job_stats']=asyncio.run(stats_for_id())
call=modal.FunctionCall.from_id(s['call_id'])
try: result['returned']=call.get(timeout=0)
except TimeoutError: result['state']='pending'
except Exception as e: result['error']=repr(e)
volume=modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
for entry in volume.iterdir(s['run']):
    dest=OUT/Path(entry.path).name
    if dest.suffix not in ('.json','.txt','.npz'): continue
    if dest.suffix=='.npz' and dest.exists() and dest.stat().st_size==entry.size: continue
    dest.write_bytes(b''.join(volume.read_file(entry.path)))
(OUT/'fetch_status.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
if (OUT/'status.json').exists():print((OUT/'status.json').read_text())
if (OUT/'alignment_audit.json').exists():
    a=json.loads((OUT/'alignment_audit.json').read_text())
    print(json.dumps({k:{f:v[f] for f in ('rmse_c','mae_c')} for k,v in a['results'].items()}))
if (OUT/'error.txt').exists():print((OUT/'error.txt').read_text())
