"""Fixed-center HLC protocol; historical data.py remains frozen for replay."""
import hashlib
from pathlib import Path

from data import HLCPairs as HistoricalPairs, EpochSampler, check_stats as check_historical_stats

PROTOCOL = 'efdiff-hlc-fixed-center224-no-flip-20260930-v1'


class HLCPairs(HistoricalPairs):
    """All months and epochs use [16:240,16:240], without spatial flips."""
    def __init__(self, root, split='train', stats=None, augment=False,
                 load_hls=True, seed=20260915):
        if augment:
            raise ValueError('Fixed-center protocol forbids random crops and flips')
        super().__init__(root, split, stats, False, load_hls, seed)
        self.fingerprint.update(
            protocol=PROTOCOL,
            spatial_policy={'crop': [16, 240, 16, 240], 'flip': False,
                            'shared_across_months': True, 'shared_across_epochs': True},
            center_data_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def __getitem__(self, item):
        if self.augment:
            raise ValueError('Fixed-center protocol forbids enabling augmentation after construction')
        return super().__getitem__(item)


def check_stats(stats, ds):
    # Historical full-train statistics were already fitted with augment=False.
    # Retain their original protocol/hash instead of relabelling their provenance.
    check_historical_stats(stats, ds)
    if ds.augment or ds.fingerprint['protocol'] != PROTOCOL:
        raise ValueError('Expected fixed-center dataset')
