"""New metadata-conditioned entry point; leaves frozen readers/data untouched."""
import json
from pathlib import Path
import numpy as np
from center_data import HLCPairs as CenterPairs, EpochSampler, check_stats
from timegeo_metadata import PROTOCOL, sha256


class HLCPairs(CenterPairs):
    def __init__(self, root, split='train', stats=None, augment=False, load_hls=True,
                 seed=20260915, *, metadata_dir):
        if split not in ('train', 'val'):
            raise ValueError('Time/geo entry point is prepared only for train and val')
        super().__init__(root, split, stats, augment, load_hls, seed)
        directory = Path(metadata_dir)
        self.metadata_audit = json.loads((directory/'audit.json').read_text(encoding='utf-8'))
        a = self.metadata_audit
        if a['protocol'] != PROTOCOL or a['order'] != ['latitude','longitude'] or a['units'] != 'degrees':
            raise ValueError('Unsupported time/geo metadata protocol')
        path = directory/'latlon_by_record.npy'
        if sha256(path) != a['artifact_sha256'] or a['source_hashes']['index.sqlite'] != self.lst.fingerprint['record_index']:
            raise ValueError('Time/geo sidecar or source index hash mismatch')
        if a['splits'][split]['sample_index_sha256'] != self.lst.fingerprint['sample_index']:
            raise ValueError('Time/geo split mismatch')
        self.latlon_by_record = np.load(path, mmap_mode='r', allow_pickle=False)
        ids = np.unique(self.lst.indices[:,0])
        coords = self.latlon_by_record[ids]
        if not np.isfinite(coords).all() or not (np.abs(coords) <= [90,180]).all():
            raise ValueError('Missing or invalid crop-center coordinates')
        self.fingerprint['conditioning'] = dict(protocol=PROTOCOL,
                artifact_sha256=a['artifact_sha256'], audit_sha256=sha256(directory/'audit.json'),
                data_code_sha256=sha256(__file__), calendar='target LST month; not HLS month')

    def __getitem__(self, item):
        sample = super().__getitem__(item)
        sample['geo_latlon'] = self.latlon_by_record[sample['record_id']].astype(np.float32)
        return sample
