"""Explicit HLP v2 datasets with split-local noise and train-only statistics."""
from pathlib import Path
import hashlib
import numpy as np
from eo_data import open_dataset
from eo_denoise.data import PairedDataset
from eo_denoise.noise import MaskBank, Corruptor, stable_seed

PREPARED = 'processed_data/hlp_split_v2_20260914'
COUNTS = {'train': 218975, 'val': 11684, 'test': 11757}


class DatasetV2(PairedDataset):
    def __init__(self, config, split='train', coverage=None):
        assert split in COUNTS and config['data']['prepared'] == PREPARED
        self.config, self.split, self.coverage = config, split, coverage
        self.modalities = ('lst',)
        root = Path(config['data']['root'])
        ds = open_dataset('prithvi', 'lst', split, root=root, prepared=root/PREPARED)
        self.sources = {'lst': ds}
        self.record_ids = np.asarray(ds.indices[:, 0])
        assert len(self.record_ids) == COUNTS[split] == len(np.unique(self.record_ids))
        bank = MaskBank(root/config['noise']['bank_root'], split, ds.fingerprint['source_index'],
                        ds.bundle['samples'][f'{split}/hls']['sha256'])
        assert {s['record_id'] for s in bank.info['shapes']} <= set(map(int, self.record_ids))
        self.positions = {'lst': np.arange(len(ds))}
        limit = config['data'].get(split+'_limit')
        if limit is not None:
            assert config['experiment']['phase'] == 'smoke'
            hashes = np.array([stable_seed(config['data']['subset_seed'], split, int(r)) for r in self.record_ids], dtype=np.uint64)
            select = np.sort(np.argsort(hashes, kind='stable')[:limit])
            self.record_ids = self.record_ids[select]
            self.positions['lst'] = self.positions['lst'][select]
        self.corruptor = Corruptor(config['noise'], split, bank)
        self.fingerprint = {'prepared': PREPARED, 'split': split,
            'record_ids_sha256': hashlib.sha256(self.record_ids.astype('<i8').tobytes()).hexdigest(),
            'source': ds.fingerprint, 'stats': ds.stats, 'samples': len(self),
            'mask_bank_sha256': bank.info['sha256'], 'mask_manifest_sha256': bank.manifest_sha256}


def epoch_updates(iterator, accumulation):
    """Yield at most accumulation microbatches; flush the tail without advancing epoch."""
    assert accumulation > 0
    while True:
        batches = []
        for _ in range(accumulation):
            try:
                batches.append(next(iterator))
            except StopIteration:
                break
        if not batches:
            return
        yield batches
