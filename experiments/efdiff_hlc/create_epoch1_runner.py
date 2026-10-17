"""One-time scaffold for the separately authorized full-data one-epoch run."""
import json
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
budget = {'run_id':'hlc-synthetic-full-epoch1-20260916-v1','scope':'user requested one complete full-training-set epoch',
          'epochs':1,'train_records':2608772,'validation_records':512,'effective_batch':64,
          'updates':40763,'tail_records':4,'record_presentations_per_model':2608772,
          'models':['x0','epsilon'],'initialization':'fresh denoiser + frozen pretrained Prithvi; no pilot weights',
          'test_used':False,'does_not_change_other_experiments_default_epochs':True}
(HERE/'epoch1_budget.json').write_text(json.dumps(budget,indent=2)+'\n')
for variant in ('x0','epsilon'):
    cfg = yaml.safe_load((HERE/'configs'/f'{variant}.yaml').read_text())
    cfg['name'] = f'efdiff_{variant}_hlc_full_epoch1_20260916'
    cfg['scope'] = 'full train membership, one epoch, fixed pilot validation membership'
    cfg['initialization'] = budget['initialization']
    cfg['data'].update(val_samples=512,test_samples=0,validation_membership='pilot_membership/val.npy')
    cfg['train'].update(epochs=1,total_updates=40763,record_presentations=2608772,
                        segment_seconds=14400,log_every=50,checkpoint_every=500)
    cfg['evaluation'].update(monitor_samples=512,monitor_selection='same fixed 512 pilot validation members',
        primary_checkpoint='fixed_epoch1_ema',final_splits=['val'],final_full_membership=False,prediction_examples=16)
    (HERE/'configs'/f'epoch1_{variant}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
source = (HERE/'pilot_train.py').read_text()
source = source.replace('from pilot_data import HLCPairs, COUNTS, PROTOCOL, HERE, UPSTREAM, check_stats, EpochSampler',
    'from data import HLCPairs, COUNTS, PROTOCOL, HERE, UPSTREAM, check_stats, EpochSampler\nfrom pilot_data import HLCPairs as ValidationPairs')
source = source.replace("f'pilot_{variant}.yaml'","f'epoch1_{variant}.yaml'")
source = source.replace("val = HLCPairs(root, split='val', stats=stats)","val = ValidationPairs(root, split='val', stats=stats)")
source = source.replace("(Path(root)/'experiments/paper1/training_budget.json')","(HERE/'epoch1_budget.json')")
source = source.replace("assert budget['epochs'] == opt['train']['epochs'] == 5", "assert budget['epochs'] == opt['train']['epochs'] == 1\n    assert budget['train_records'] == len(ds) and budget['updates'] == opt['train']['total_updates']")
source = source.replace("HERE/'pilot_fit_stats.py', HERE/'pilot_train.py', HERE/'pilot_membership/summary.json', HERE/'pilot_membership/train.npy'",
                        "HERE/'fit_stats.py', HERE/'epoch1_stats.py', HERE/'epoch1_train.py', HERE/'epoch1_budget.json', HERE/'pilot_membership/summary.json'")
source = source.replace('Pilot-only shared short warmup and EMA, recorded in YAML.', 'Full-run upstream warmup/EMA settings are recorded in YAML.')
source = source.replace("'PILOT fixed epoch5 EMA; all fixed 512 val months each epoch; no test evaluation'", "'Full-training-set epoch1 EMA; same fixed 512 pilot val members; no test evaluation'")
source = source.replace("output/'predictions_epoch5.npz'", "output/'predictions_epoch1.npz'")
# Preserve epoch updates and cumulative timing over bounded GPU segments.
source = source.replace('    best = math.inf','    best = math.inf\n    previous_seconds = 0.0')
source = source.replace("        del state\n    def save", "        previous_seconds = state.get('accumulated_seconds',0.0)\n        del state\n    def save")
source = source.replace("                 'variant': variant, 'smoke': smoke}","                 'variant': variant, 'smoke': smoke,\n                 'accumulated_seconds': previous_seconds+time.time()-started}")
source = source.replace("'elapsed_seconds': time.time()-started", "'elapsed_seconds': previous_seconds+time.time()-started")
(HERE/'epoch1_train.py').write_text(source)
