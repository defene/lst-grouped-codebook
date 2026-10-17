"""Verify the complete evaluation from saved sample errors and source membership."""
from pathlib import Path
import argparse
from collections import defaultdict
import json
import math
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eo_data import open_dataset
from eo_data.core import sha256_file, json_write
from experiments.paper1.lst_full_eval.data import PREPARED, COUNTS
from experiments.paper1.lst_full_eval.run import OUTPUT, ARMS


def main(output):
    status = json.loads((output/'status.json').read_text(encoding='utf-8'))
    assert status['status'] == 'complete'
    digests = {}
    summaries = []
    checkpoint_hashes = {}
    expected = {}
    for split in ('val', 'test'):
        ds = open_dataset('prithvi', 'lst', split, root=ROOT, prepared=ROOT/PREPARED)
        expected[split] = set(map(int, ds.indices[:, 0]))
        ds.close()
    for arm, _, _ in ARMS:
        for split in ('val', 'test'):
            for condition in ('clean', 'noisy10'):
                folder = output/arm/split/condition
                result = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
                cp = result['checkpoint_path']
                if cp not in checkpoint_hashes:
                    checkpoint_hashes[cp] = sha256_file(Path(cp))
                assert checkpoint_hashes[cp] == result['checkpoint_sha256']
                assert sha256_file(folder/'samples.jsonl') == result['samples_sha256']
                assert result['evaluated_pairs'] == COUNTS[split]
                assert set(np.load(folder/'record_ids.npy').tolist()) == expected[split]
                assert digests.setdefault((split, condition), result['input_digest']) == result['input_digest']
                rows = [json.loads(s) for s in (folder/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
                all_rows = [r for r in rows if r['region'] == 'all']
                groups = defaultdict(list)
                for r in all_rows:
                    assert all(math.isfinite(r[field]) for field in ('mse_physical', 'bias_physical', 'mae_physical'))
                    groups[int(r['sample_id'].rsplit(':', 1)[1])].append(r)
                assert set(groups) == expected[split]
                assert all(sorted(r['view'] for r in rs) == [0, 1] for rs in groups.values())
                mse = np.mean([r['mse_physical'] for r in all_rows])
                mean_mse = np.mean([r['bias_physical']**2 for r in all_rows])
                assert np.isclose(result['regions']['lst/all']['rmse']**2, mse, rtol=1e-10)
                assert np.isclose(result['mean_rmse']**2, mean_mse, rtol=1e-10)
                assert np.isclose(result['spatial_rmse']**2+mean_mse, mse, rtol=1e-10)
                summaries.append({'model': arm, 'split': split, 'condition': condition,
                                  'pairs_verified': len(groups), 'views_verified': len(all_rows),
                                  'rmse': float(mse**.5), 'metrics_sha256': sha256_file(folder/'metrics.json')})
    json_write(output/'completion_review.json', {'status': 'passed', 'units': summaries,
        'complete_membership': True, 'same_inputs_across_models': True, 'physical_rmse_recomputed': True,
        'checkpoint_and_sample_hashes': True, 'mean_spatial_decomposition': True})
    print('PASS: 16 full evaluation units, complete membership, matching inputs, hashes, and physical metrics.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    main(parser.parse_args().output)

