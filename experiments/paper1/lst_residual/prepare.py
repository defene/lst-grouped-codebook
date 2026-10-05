"""Freeze the four matched arms; all use the existing LST data protocol."""
from pathlib import Path
import copy
import json
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.paper1.var_vs_prithvi_ae.run_suite import json_write
from eo_data.core import sha256_file
HERE = Path(__file__).resolve().parent
ARMS = [('absolute', False, 0), ('absolute_spatial4', False, 3),
        ('residual', True, 0), ('residual_spatial4', True, 3)]


def main():
    basepath = ROOT/'runs/paper1/codebook1024/full/full_lst_conv_noisy_seed17/config.json'
    base = json.loads(basepath.read_text(encoding='utf-8'))
    for phase in ('smoke', 'pilot'):
        folder = HERE/'configs'/phase
        folder.mkdir(parents=True, exist_ok=True)
        for name, residual, weight in ARMS:
            c = copy.deepcopy(base)
            c['name'] = f'{phase}_lst_{name}_seed17'
            c['residual_experiment'] = dict(id='A04', predict_residual=residual, spatial_extra_weight=weight)
            c['train'].update(steps=8 if phase == 'smoke' else 2000, batch_size=4,
                              accumulation=2 if phase == 'smoke' else 16, workers=2,
                              warmup=2 if phase == 'smoke' else 100,
                              eval_every=8 if phase == 'smoke' else 500,
                              save_every=8 if phase == 'smoke' else 250,
                              log_every=1 if phase == 'smoke' else 20, memory_limit_gib=12)
            c['eval'].update(batch_size=4, bootstrap_replicates=100 if phase == 'smoke' else 500)
            c['data'].update(train_limit=64 if phase == 'smoke' else None, val_limit=16 if phase == 'smoke' else 256)
            path = folder/(c['name']+'.json')
            if path.exists():
                assert json.loads(path.read_text(encoding='utf-8')) == c
            else:
                json_write(path, c)
    json_write(HERE/'derivation.json', dict(base=str(basepath.relative_to(ROOT)), base_sha256=sha256_file(basepath),
        arms=ARMS, matched_training='10% known-mask missing; same initialization and record order; scratch',
        pilot_steps=2000, effective_batch=64, full_training_pool=230659, sample_presentations=128000,
        pilot_is_not_full_convergence=True, train_normalization='unchanged frozen LST normalization'))


if __name__ == '__main__':
    main()
