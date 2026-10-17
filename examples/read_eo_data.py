"""One command to open all four independent training profiles."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eo_data import open_datasets


def main():
    data = open_datasets()
    for name, modalities in data.items():
        for modality, ds in modalities.items():
            sample = ds[0]
            print(json.dumps({
                'dataset': name, 'modality': modality, 'samples': len(ds),
                'shape': list(sample['image'].shape), 'dtype': str(sample['image'].dtype),
                'valid_pixels': sample['valid_pixels'], 'sample_id': sample['sample_id'],
                'normalization_mean': ds.stats['mean'], 'normalization_std': ds.stats['std'],
                'fingerprint': ds.fingerprint,
            }, ensure_ascii=False))
            ds.close()


if __name__ == '__main__':
    main()
