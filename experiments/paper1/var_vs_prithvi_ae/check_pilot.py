"""Validate all eight small runs before releasing the full-data queue."""
from pathlib import Path
import argparse
import json
import math
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from eo_data.core import json_write,sha256_file
from eo_denoise.engine import source_hashes

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def main(output):
    suite=read(output/'status.json')
    if suite['status']!='training_and_evaluation_complete':raise ValueError('Small suite is incomplete')
    manifest=read(output/'manifest.json')
    if manifest['source_hashes']!=source_hashes():raise ValueError('Source changed since small suite')
    rows=[];failures=[];reference_ids=None;val_reference=None
    for m in ('hls','lst'):
        for b in ('conv','prithvi'):
            for n in ('clean','noisy'):
                name=f'pilot_{m}_{b}_{n}_seed17';run=output/name
                status=read(run/'status.json');model=read(run/'model.json');c=read(run/'config.json')
                logs=[json.loads(s) for s in (run/'train.jsonl').read_text().splitlines()]
                ids=np.load(run/'train_record_ids.npy')
                if reference_ids is None:reference_ids=ids
                if not np.array_equal(ids,reference_ids):failures.append(name+': train subset differs')
                if status['status']!='complete' or status['step']!=128 or len(logs)!=128:failures.append(name+': budget incomplete')
                if not all(math.isfinite(x[k]) for x in logs for k in ('total','reconstruction','gradient_norm','bottleneck')):
                    failures.append(name+': nonfinite training value')
                early=float(np.mean([x['reconstruction'] for x in logs[:16]]))
                late=float(np.mean([x['reconstruction'] for x in logs[-16:]]))
                if late>=.9*early:failures.append(name+': reconstruction did not improve by 10%')
                seen={s for x in logs for s in x['sample_ids']}
                if len(seen)!=64:failures.append(name+': not all small-subset records used')
                if model['pretrained_weights']:failures.append(name+': pretrained weights')
                if b=='prithvi':
                    if c['model']['backbone']!='prithvi_eo2_continuous' or c['model']['quantizer']!='ae':
                        failures.append(name+': Prithvi is not continuous AE')
                    if any(x['bottleneck']!=0 or x['kl']!=0 for x in logs):failures.append(name+': nonzero VQ/KL loss')
                evaluations={}
                for condition in ('clean','noisy10'):
                    folder=run/('eval_'+condition);metrics=read(folder/'metrics.json')
                    val_ids={json.loads(s)['sample_id'] for s in (folder/'samples.jsonl').read_text().splitlines()}
                    if val_reference is None:val_reference=val_ids
                    if val_ids!=val_reference or len(val_ids)!=64:failures.append(name+': validation subset differs')
                    if not math.isfinite(metrics['selection_score']):failures.append(name+': nonfinite evaluation')
                    if b=='prithvi' and metrics['tokens']:failures.append(name+': discrete token metrics on continuous AE')
                    if b=='conv' and metrics['tokens'][m]['used_codes']<2:failures.append(name+': codebook collapsed to one code')
                    evaluations[condition]=dict(rmse=metrics['regions'][m+'/all']['rmse'],
                        unit=metrics['regions'][m+'/all']['unit'],tokens=metrics['tokens'].get(m))
                rows.append(dict(name=name,parameters=model['parameters'],early_reconstruction=early,
                    late_reconstruction=late,reduction_percent=100*(1-late/early),
                    peak_allocated_gib=max(x['memory']['peak_allocated_gib'] for x in logs),
                    train_unique_records=len(seen),evaluation=evaluations))
    gpu=[json.loads(s) for s in (output/'gpu.jsonl').read_text().splitlines()]
    gate=dict(status='failed' if failures else 'passed',source_hashes=source_hashes(),
        configs=manifest['config_hashes'],rows=rows,failures=failures,
        peak_device_used_gib=max(x['used_gib'] for x in gpu),
        validates='finite training and evaluation, learning on small real subset, no VQ/KL in Prithvi, common data, parallel VRAM',
        does_not_establish='convergence, superiority, or matched compression rate',
        source_manifest_sha256=sha256_file(output/'manifest.json'))
    json_write(output/'gate.json',gate)
    print(json.dumps(gate,indent=2))
    if failures:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
