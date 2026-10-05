from pathlib import Path
import copy
import json
import math
import numpy as np
from eo_data import open_dataset
from eo_data.core import json_write, sha256_file
from .data import PREPARED, COUNTS, DatasetV2
from .mask_bank import build_bank

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT/'runs/paper1/lst_v2_5epoch_20260915'
ARMS = ['absolute', 'absolute_spatial4', 'residual', 'residual_spatial4']


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ids, tiles, stats, prints = {}, {}, {}, {}
    for split in COUNTS:
        ds = open_dataset('prithvi', 'lst', split, root=ROOT, prepared=ROOT/PREPARED)
        ids[split] = set(map(int, ds.indices[:, 0]))
        assert len(ids[split]) == COUNTS[split]
        stats[split], prints[split] = ds.stats, ds.fingerprint
        tiles[split] = set((ROOT/f'data_inspection/hlp_split_20260914/{split}_tiles.txt').read_text().splitlines())
        ds.close()
    assert stats['train'] == stats['val'] == stats['test']
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not ids[a]&ids[b] and not tiles[a]&tiles[b]
    for split in COUNTS:
        if not (OUTPUT/f'mask_banks/{split}.json').exists():
            build_bank(ROOT, OUTPUT/'mask_banks', split)
    configs = {}
    for phase in ('smoke', 'full'):
        for arm in ARMS:
            old = ROOT/f'runs/paper1/lst_residual_20260913/pilot/pilot_lst_{arm}_seed17/config.json'
            c = json.loads(old.read_text(encoding='utf-8'))
            c['name'] = f'{phase}_lst_v2_{arm}_seed17'
            c['data'].update(root=str(ROOT), prepared=PREPARED,
                train_limit=67 if phase=='smoke' else None,
                val_limit=8 if phase=='smoke' else None, test_limit=8 if phase=='smoke' else None)
            c['noise']['bank_root'] = str(OUTPUT/'mask_banks')
            epochs = 2 if phase=='smoke' else 5
            n = 67 if phase=='smoke' else COUNTS['train']
            c['train'].update(steps=epochs*math.ceil(n/64), batch_size=4, accumulation=16, workers=2,
                warmup=1 if phase=='smoke' else 500, log_every=1 if phase=='smoke' else 20,
                save_every=1 if phase=='smoke' else 250, eval_every=math.ceil(n/64), memory_limit_gib=12)
            c['eval'].update(batch_size=4, realizations=2, bootstrap_replicates=50 if phase=='smoke' else 500)
            c['experiment'] = {'id': 'A05', 'phase': phase, 'epochs': epochs, 'output_root': str(OUTPUT),
                'pretrained_weights': False, 'primary_checkpoint': 'last_after_exact_epochs',
                'no_test_model_selection': True, 'epoch_tail': 'flush_actual_sample_weight_no_padding',
                'schedule': 'cosine_with_500_update_warmup_for_full'}
            dest = HERE/'configs'/phase/(c['name']+'.json')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                assert json.loads(dest.read_text(encoding='utf-8')) == c
            else:
                json_write(dest, c)
            configs[str(dest.relative_to(HERE))] = sha256_file(dest)
            for split in COUNTS:
                ds = DatasetV2(c, split, .1)
                assert ds.sources['lst'].stats == stats['train']
                sample = ds[0]
                assert np.isfinite(sample['lst']['image']).all()
                assert not sample['lst']['image'][~sample['lst']['input_mask'].astype(bool)].any()
                ds.close()
    json_write(OUTPUT/'data_preflight.json', {'status': 'passed', 'counts': COUNTS,
        'tile_counts': {k: len(v) for k, v in tiles.items()}, 'disjoint_ids_and_tiles': True,
        'normalization': stats['train'], 'fingerprints': prints, 'configs': configs,
        'formal_records_per_model': COUNTS['train']*5, 'formal_updates': 17110, 'tail_records': 31})
    print('PASS: v2 splits, train-only normalization, three split-local mask banks and 8 configs.')


if __name__ == '__main__':
    main()
