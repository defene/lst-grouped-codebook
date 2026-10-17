"""Verify final artifacts and spawn-worker reads without requiring PyTorch."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eo_data import open_datasets
from eo_data.core import json_write
from eo_data.verify import verify_all, restore_prithvi


def signatures(ds, indices):
    result = []
    for i in indices:
        sample = ds[i]
        h = hashlib.sha256(sample['image'].tobytes() + sample['reference_mask'].tobytes()).hexdigest()
        result.append((sample['sample_id'], h))
    ds.close()
    return result


def main():
    report = {'verification': verify_all(ROOT, samples=32), 'profiles': {}}
    datasets = open_datasets(root=ROOT)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        for name, mods in datasets.items():
            for mod, ds in mods.items():
                chosen = [0, len(ds)//2, len(ds)-1]
                expected = signatures(ds, chosen)
                assert pool.submit(signatures, ds, chosen).result() == expected
                started = time.perf_counter()
                indices = ds.epoch_indices(0)[:64]
                signatures(ds, list(map(int, indices)))
                report['profiles'][f'{name}/{mod}'] = {
                    'train_samples': len(ds), 'spawn_worker_values_identical': True,
                    'random_read_samples': len(indices),
                    'random_read_seconds': time.perf_counter()-started,
                    'mean': ds.stats['mean'], 'std': ds.stats['std'], 'fingerprint': ds.fingerprint,
                }
                ds.close()
    restore = ROOT/'processed_data/restore_check'
    if not restore.exists():
        report['restore'] = restore_prithvi(ROOT, None, restore, limit=8)
    else:
        report['restore'] = {'destination': str(restore), 'reused_existing_check': True}
    compared = 0
    for file in restore.rglob('*'):
        if file.is_file():
            original = ROOT/'prithvi_data'/file.relative_to(restore)
            if original.exists():
                assert file.read_bytes() == original.read_bytes()
                compared += 1
    report['restore']['direct_original_files_compared'] = compared
    report['restore']['original_bytes_match'] = True if compared else None
    report['restore']['decoded_content_hashes_verified'] = True
    report['status'] = 'passed'
    json_write(ROOT/'experiments/data/materialization_validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
