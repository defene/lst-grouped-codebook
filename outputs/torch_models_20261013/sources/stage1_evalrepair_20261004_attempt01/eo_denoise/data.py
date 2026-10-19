from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np
from torch.utils.data import Dataset, Sampler
from eo_data import open_dataset
from .noise import Corruptor, MaskBank, stable_seed
from .config import fingerprint


class PairedDataset(Dataset):
    """Join original record IDs, preserving per-modality v1 normalization."""
    def __init__(self, config, split='train', coverage=None):
        self.config, self.split, self.coverage = config, split, coverage
        root = Path(config['data']['root']).resolve()
        self.modalities = tuple(config['data']['modalities'])
        self.sources = {m: open_dataset('prithvi', m, split, root=root) for m in self.modalities}
        first = self.sources[self.modalities[0]]
        record_ids = np.asarray(first.indices[:, 0])
        if len(np.unique(record_ids)) != len(record_ids):
            raise ValueError('Expected one observation per Prithvi record')
        for ds in self.sources.values():
            record_ids = np.intersect1d(record_ids, ds.indices[:, 0], assume_unique=True)
        self.positions = {}
        for m, ds in self.sources.items():
            ids = np.asarray(ds.indices[:, 0])
            order = np.argsort(ids)
            self.positions[m] = order[np.searchsorted(ids[order], record_ids)]
        self.record_ids = record_ids
        limit = config['data']['train_limit' if split == 'train' else 'val_limit']
        # Full membership is unchanged; subsets use a fixed dataset seed shared
        # by every model and every training seed.
        if limit is not None and limit < len(record_ids):
            hashes = np.array([stable_seed(config['data']['subset_seed'], split, int(r)) for r in record_ids], dtype=np.uint64)
            selected = np.sort(np.argsort(hashes, kind='stable')[:limit])
            self.record_ids = self.record_ids[selected]
            self.positions = {m: p[selected] for m, p in self.positions.items()}
        if not len(self.record_ids):
            raise ValueError('No eligible samples')
        bank = None
        if config['noise']['kind'] == 'bank_missing':
            bank = MaskBank(root/config['noise']['bank_root'], split, first.fingerprint['source_index'],
                            first.bundle['samples'][f'{split}/hls']['sha256'])
        self.corruptor = Corruptor(config['noise'], split, bank)
        self.fingerprint = {
            'paired_record_ids_sha256': hashlib.sha256(self.record_ids.astype('<i8').tobytes()).hexdigest(),
            'sources': {m: ds.fingerprint for m, ds in self.sources.items()},
            'normalization': 'frozen_independent_modality_stats',
            'mask_bank_sha256': bank.info['sha256'] if bank else None,
            'mask_bank_manifest_sha256': bank.manifest_sha256 if bank else None,
            'samples': len(self), 'alignment': config['data']['alignment'],
            'dataset_config': fingerprint(config['data']),
        }

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, index):
        epoch, index = index if isinstance(index, tuple) else (0, index)
        index = int(index)
        anchor = self.modalities[0]
        first = self.sources[anchor]
        position = int(self.positions[anchor][index])
        meta = first.get_metadata(position)
        hls_id = meta['hls_image_id']
        hls_tile, timestamp = hls_id.rsplit('_', 1)
        hls_time = datetime.strptime(timestamp, '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
        observed = datetime.fromisoformat(meta['datetime'].replace('Z', '+00:00'))
        delta = (hls_time-observed).total_seconds()
        exact_metadata_match = hls_tile == meta['tile'] and abs(delta) < 1
        pair_id = f"prithvi:{self.split}:{int(self.record_ids[index])}"
        result = {'sample_id': pair_id, 'tile_id': meta['tile'], 'record_id': int(self.record_ids[index]),
                  'source_tile_match': hls_tile == meta['tile'], 'time_delta_seconds': delta,
                  'cross_view_eligible': exact_metadata_match if self.config['data']['alignment'] != 'all_pairs' else True}
        views = self.config['data']['train_views'] if self.split == 'train' else self.config['eval']['realizations']
        for m, ds in self.sources.items():
            clean = ds[int(self.positions[m][index])]
            noisy = [self.corruptor(clean['image'], clean['reference_mask'], pair_id, m, epoch, v, self.coverage)
                     for v in range(views)]
            result[m] = {'target': clean['target'], 'reference_mask': clean['reference_mask'],
                         'image': np.stack([v['image'] for v in noisy]),
                         'input_mask': np.stack([v['input_mask'] for v in noisy]),
                         'corruption_mask': np.stack([v['corruption_mask'] for v in noisy]),
                         'actual_coverage': np.array([v['actual_coverage'] for v in noisy], np.float32),
                         'requested_coverage': np.array([v['requested_coverage'] for v in noisy], np.float32),
                         'noise_seed': [v['noise_seed'] for v in noisy],
                         'noise_kind': [v['noise_kind'] for v in noisy],
                         'shape_ids': [v['shape_ids'] for v in noisy]}
        return result

    def close(self):
        for ds in self.sources.values():
            ds.close()


class EpochSampler(Sampler):
    """Epoch travels with the index, including into persistent worker processes."""
    def __init__(self, length, seed, epoch=0, start=0):
        self.length, self.seed, self.epoch, self.start = length, seed, epoch, start

    def __iter__(self):
        rng = np.random.default_rng(stable_seed('sample_order', self.seed, self.epoch))
        for i in rng.permutation(self.length)[self.start:]:
            yield self.epoch, int(i)

    def __len__(self):
        return self.length-self.start
