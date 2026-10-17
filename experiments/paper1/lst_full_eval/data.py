"""Full HLP v2 evaluation membership and normalization, isolated from frozen training."""
from pathlib import Path
import hashlib
import numpy as np
from eo_data import open_dataset
from eo_denoise.data import PairedDataset
from eo_denoise.noise import Corruptor, MaskBank

PREPARED = 'processed_data/hlp_split_v2_20260914'
COUNTS = {'train': 218975, 'val': 11684, 'test': 11757}


class FullDataset(PairedDataset):
    def __init__(self, config, split, coverage, bank_root):
        if split not in ('val', 'test'):
            raise ValueError('Evaluation only')
        self.config, self.split, self.coverage = config, split, coverage
        self.modalities = ('lst',)
        root = Path(config['data']['root']).resolve()
        ds = open_dataset('prithvi', 'lst', split, root=root, prepared=root/PREPARED)
        self.sources = {'lst': ds}
        self.record_ids = np.asarray(ds.indices[:, 0])
        assert len(self.record_ids) == COUNTS[split] == len(np.unique(self.record_ids))
        self.positions = {'lst': np.arange(len(ds))}
        bank = MaskBank(bank_root, split, ds.fingerprint['source_index'],
                        ds.bundle['samples'][f'{split}/hls']['sha256'])
        assert {s['record_id'] for s in bank.info['shapes']} <= set(self.record_ids.tolist())
        self.corruptor = Corruptor(config['noise'], split, bank)
        self.fingerprint = {
            'prepared': PREPARED, 'split': split, 'sources': {'lst': ds.fingerprint},
            'paired_record_ids_sha256': hashlib.sha256(self.record_ids.astype('<i8').tobytes()).hexdigest(),
            'normalization': ds.stats, 'samples': len(self),
            'mask_bank_sha256': bank.info['sha256'],
            'mask_bank_manifest_sha256': bank.manifest_sha256,
            'noise_seed_split': split, 'evaluation_views': config['eval']['realizations'],
        }
