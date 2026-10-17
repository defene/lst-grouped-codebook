"""Read and download an existing test job; never launch or spawn GPU work."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MODAL_PROFILE','runlongyu-target')
import modal

APP='hlc-unet-mmdit-fulltest-20261015'
RUN='hlc-unet-mmdit-fulltest-20261015-v1'
OUT=Path(__file__).resolve().parents[2]/'output/hlc_unet_mmdit_test_20261015'

def main():
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');args=p.parse_args()
    state=modal.Function.from_name(APP,'status',environment_name='main').remote()
    stats=modal.Function.from_name(APP,'job',environment_name='main').get_current_stats()
    state['job_stats']={'runners':stats.num_total_runners,'running':stats.num_running_inputs,'backlog':stats.backlog}
    call=state.get('pipeline.json',{}).get('call_id') or state.get('submission.json',{}).get('call_id')
    if call:
        for _ in range(52):
            state['tracked_call_id']=call
            try:
                result=modal.FunctionCall.from_id(call).get(timeout=0)
                state['call_result']=result
                if isinstance(result,dict) and result.get('phase')=='queued_continuation':
                    call=result['call_id'];continue
            except (modal.exception.TimeoutError,TimeoutError):state['call_result']='pending'
            except Exception as e:state['call_result']={'error':repr(e)}
            break
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'latest_status.json').write_text(json.dumps(state,indent=2)+'\n')
    for name,value in state.items():
        if name.endswith('.json'):
            dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(value,indent=2)+'\n')
    if args.download:
        vol=modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
        for entry in vol.iterdir(RUN,recursive=True):
            if not entry.path.endswith(('.json','.csv','.npz')):continue
            relative=entry.path.removeprefix('/').removeprefix(RUN+'/');dest=OUT/relative
            if '/chunks/' in '/'+relative and dest.exists() and dest.stat().st_size==entry.size:continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.with_suffix(dest.suffix+'.download').open('wb') as f:
                for b in vol.read_file(entry.path):f.write(b)
            dest.with_suffix(dest.suffix+'.download').replace(dest)
    print(json.dumps({k:v for k,v in state.items() if not k.endswith('protocol.json')},indent=2))

if __name__=='__main__':main()
