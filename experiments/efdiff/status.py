"""Read current Modal run evidence without launching a GPU."""
import json
import modal

volume = modal.Volume.from_name('efdiff-hlp-v2-runs', environment_name='main')
run = 'hlp-v2-x0-5epoch-20260915'
for name in ['pipeline.json','preflight.json','smoke/status.json','normalization.json','formal/status.json']:
    try:
        raw = b''.join(volume.read_file(f'{run}/{name}'))
        value = json.loads(raw)
        if name == 'normalization.json':
            value = {k:value[k] for k in ('protocol','fit_split','records','lst','residual')}
        print(json.dumps({'artifact':name,'data':value}, ensure_ascii=False))
    except FileNotFoundError:
        print(json.dumps({'artifact':name,'data':None}))
