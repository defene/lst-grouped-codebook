"""Read-only migration smoke check for current HLP v2 and HLC v1 assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dataset', choices=('hlp', 'hlc', 'all'), default='all')
    parser.add_argument('--verify-hlp-archive', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    from eo_data import open_dataset

    catalog = json.loads((root / 'experiments/data/dataset_catalog.json').read_text(encoding='utf-8'))
    report = {'root': str(root), 'scope': 'frozen asset hashes and one sample per split/modality',
              'checks': [], 'errors': []}
    for spec in catalog['datasets']:
        short = spec['short_name'].lower()
        if args.dataset not in ('all', short):
            continue
        prepared = root / Path(spec['prepared_directory']).parent
        for split, counts in spec['splits'].items():
            for modality, count_key in [('hls', 'hls_samples'), ('lst', 'fine_lst_samples')]:
                ds = None
                try:
                    ds = open_dataset(spec['reader_id'], modality, counts['reader_split'],
                                      root=root, prepared=prepared)
                    if len(ds) != counts[count_key]:
                        raise ValueError(f'Count mismatch: {len(ds)} != {counts[count_key]}')
                    sample = ds[0]
                    mask = sample['reference_mask']
                    if not np.any(mask) or not np.isfinite(sample['image']).all():
                        raise ValueError('First sample has invalid normalized values or empty mask')
                    check = {'dataset': short, 'split': split, 'modality': modality,
                             'records': len(ds), 'shape': list(sample['image'].shape),
                             'first_sample_valid_pixels': int(mask.sum()), 'fingerprint': ds.fingerprint}
                    report['checks'].append(check)
                except Exception as exc:
                    report['errors'].append({'dataset': short, 'split': split,
                                             'modality': modality, 'error': str(exc)})
                finally:
                    if ds is not None:
                        ds.close()
    if args.verify_hlp_archive:
        try:
            storage = json.loads((root / 'processed_data/hlp_split_v2_20260914/prithvi/storage.json').read_text(encoding='utf-8'))
            digest = hashlib.sha256()
            with (root / storage['archive']).open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != storage['archive_sha256']:
                raise ValueError('HLP archive SHA-256 mismatch')
            report['hlp_archive_sha256'] = digest.hexdigest()
        except Exception as exc:
            report['errors'].append({'dataset': 'hlp', 'error': str(exc)})
    report['status'] = 'passed' if not report['errors'] and report['checks'] else 'failed'
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
