"""One-time scaffolding for the separate pilot; full-run source is preserved."""
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
for variant in ('x0','epsilon'):
    cfg = yaml.safe_load((HERE/'configs'/f'{variant}.yaml').read_text())
    cfg['name'] = f'efdiff_{variant}_hlc_synthetic_pilot_16384_5epoch'
    cfg['scope'] = 'fixed subset pilot; not a full-data reproduction'
    cfg['data'].update(train_samples=16384, val_samples=512, test_samples=0,
                       membership='pilot_membership/summary.json')
    cfg['normalization']['fit_samples'] = 16384
    cfg['train'].update(updates_per_epoch=256, tail_records=0, total_updates=1280,
                        record_presentations=81920, warmup_steps=100,
                        ema_decay=0.99, ema_start_step=100, checkpoint_every=128)
    cfg['evaluation'].update(monitor_samples=512, monitor_selection='all fixed 512 pilot validation members',
                            final_splits=['val'], final_full_membership=False, prediction_examples=16)
    (HERE/'configs'/f'pilot_{variant}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))

source = (HERE/'train.py').read_text()
source = source.replace('from data import HLCPairs', 'from pilot_data import HLCPairs')
source = source.replace("f'{variant}.yaml'", "f'pilot_{variant}.yaml'")
source = source.replace("HERE/'fit_stats.py', HERE/'train.py'", "HERE/'pilot_data.py', HERE/'pilot_fit_stats.py', HERE/'pilot_train.py', HERE/'pilot_membership/summary.json', HERE/'pilot_membership/train.npy', HERE/'pilot_membership/val.npy'")
source = source.replace("'fixed epoch5 EMA; 4096 val months monitored per epoch; full val/test afterwards'", "'PILOT fixed epoch5 EMA; all fixed 512 val months each epoch; no test evaluation'")
source = source.replace('    # Keep the upstream 5000-update warmup; five epochs is an explicit budget change.',
                        '    # Pilot-only shared short warmup and EMA, recorded in YAML.')
source = source.replace('indices=None, seed=20260915):', 'indices=None, seed=20260915, capture_path=None):')
source = source.replace('    squared = absolute = baseline = pixels = records = 0',
'''    squared = absolute = baseline = pixels = records = 0
    captures = {key:[] for key in ('target_c','condition_c','mask','prediction_c','index','record_id','month')}
    capture_ids = set(int(ds.members[i]) for i in np.linspace(0,len(ds)-1,16,dtype=int)) if capture_path else set()''')
source = source.replace("            mask = b['mask']", """            mask = b['mask']
            for i, source_index in enumerate(raw['index'].tolist()):
                if source_index in capture_ids:
                    for key in captures:
                        value = prediction[i] if key=='prediction_c' else raw[key][i]
                        captures[key].append(value.detach().cpu().numpy() if torch.is_tensor(value) else value)""")
source = source.replace('    net.train()\n    return', """    if capture_path:
        assert len(captures['index']) == 16
        np.savez_compressed(capture_path, **{k:np.asarray(v) for k,v in captures.items()})
    net.train()
    return""")
source = source.replace("metrics = evaluate(net, val, stats, micro, workers, limit=opt['smoke']['validation_records'] if smoke else opt['evaluation']['monitor_samples'])",
"""metrics = evaluate(net, val, stats, micro, workers,
            limit=opt['smoke']['validation_records'] if smoke else opt['evaluation']['monitor_samples'],
            capture_path=output/'predictions_epoch5.npz' if not smoke and epoch+1==epochs else None)""")
source = source[:source.index('\ndef final_evaluation_chunk')]
(HERE/'pilot_train.py').write_text(source)
