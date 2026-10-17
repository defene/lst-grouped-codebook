"""Explicit launch or read-only fetch for the isolated v/score comparison."""
import argparse
import json
import os
from pathlib import Path
os.environ['MODAL_PROFILE']='runlongyu-target'
os.environ['MODAL_ENVIRONMENT']='main'
import modal

APP='efdiff-hlc-wide-vscore-20260927'
RUN='hlc-wide-vscore-20260927-v1'
OUT=Path(__file__).resolve().parents[2]/'output/efdiff_hlc_vscore_20260927'


def main(action):
    OUT.mkdir(parents=True,exist_ok=True)
    if action in ('smoke','start'):
        result=modal.Function.from_name(APP,'launch',environment_name='main').remote(smoke=action=='smoke')
        (OUT/f'{action}_launch.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        return
    fn=modal.Function.from_name(APP,'job',environment_name='main')
    stats=fn.get_current_stats()
    result=modal.Function.from_name(APP,'status',environment_name='main').remote()
    result['job_stats']=dict(backlog=stats.backlog,num_total_runners=stats.num_total_runners,
                             num_running_inputs=stats.num_running_inputs)
    (OUT/'status.json').write_text(json.dumps(result,indent=2)+'\n')
    if action=='download':
        volume=modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
        names=['preflight.json','train_launch.json']
        for v in ('v','score'):
            names += [f'train/{v}/{f}' for f in ('provenance.json','complete.json','metrics.jsonl','status.json')]
            for sel in ('best','final'):
                for nfe in (50,20,10,5):
                    names += [f'train/{v}/{sel}_nfe{nfe}.{ext}' for ext in ('json','npz')]
        for name in names:
            try:raw=b''.join(volume.read_file(f'{RUN}/{name}'))
            except FileNotFoundError:continue
            dest=OUT/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(raw)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['smoke','start','status','download'])
    main(parser.parse_args().action)
