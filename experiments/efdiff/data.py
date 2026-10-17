"""HLP v2 adapter; frozen eo_data and upstream EFDiff remain unchanged."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE / 'upstream'))
from util.downsampling import walds_protocol_downsample
from eo_data import open_dataset

PREPARED = 'processed_data/hlp_split_v2_20260914'
COUNTS = {'train': 218975, 'val': 11684, 'test': 11757}
PROTOCOL = 'efdiff-hlp-v2-masked-224-wald32-v1'


class HLPPairs(Dataset):
    def __init__(self, root, split='train', stats=None, augment=False, load_hls=True):
        root = Path(root).resolve()
        self.lst = open_dataset('prithvi', 'lst', split, root=root, prepared=root / PREPARED)
        self.hls = open_dataset('prithvi', 'hls', split, root=root, prepared=root / PREPARED) if load_hls else None
        assert len(self.lst) == COUNTS[split]
        if self.hls is not None:
            assert np.array_equal(self.lst.indices[:, :2], self.hls.indices[:, :2]), 'HLS/LST pairing drift'
        self.stats = stats
        self.augment = augment
        self.split = split
        self.fingerprint = {'protocol': PROTOCOL, 'lst': self.lst.fingerprint,
                            'hls': self.hls.fingerprint if self.hls else None}

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, item):
        epoch, index = item if isinstance(item, tuple) else (0, int(item))
        record = self.lst[index]
        fine = self.lst.denormalize(record['image'])[0] / 100.0
        mask = record['reference_mask'][0]
        # Match upstream fill-before-crop convention; filled pixels never enter loss/metrics.
        fine = np.where(mask, fine, fine[mask].mean())
        hls = None
        if self.hls is not None:
            optical = self.hls[index]
            assert optical['tile_id'] == record['tile_id']
            hls = np.clip(self.hls.denormalize(optical['image']), 0, 1)
            hls = np.where(optical['reference_mask'], hls, 0)
        rng = np.random.default_rng(np.random.SeedSequence([20260914, epoch, index]))
        top, left = rng.integers(0, 33, 2) if self.augment else (16, 16)
        fine = fine[top:top+224, left:left+224]
        mask = mask[top:top+224, left:left+224]
        if hls is not None:
            hls = hls[:, top:top+224, left:left+224]
        if self.augment:
            for axis in (0, 1):
                if rng.random() < .5:
                    fine = np.flip(fine, axis)
                    mask = np.flip(mask, axis)
                    if hls is not None:
                        hls = np.flip(hls, axis+1)
        coarse = walds_protocol_downsample(np.ascontiguousarray(fine), 32)
        sr = torch.nn.functional.interpolate(torch.from_numpy(coarse.copy())[None, None],
                    size=(224, 224), mode='bicubic', align_corners=False)[0].numpy()
        residual = fine[None] - sr
        result = {'target_c': fine[None].copy(), 'condition_c': sr.copy(),
                  'mask': mask[None].copy(), 'index': index}
        if self.stats is not None:
            for name, value, key in [('HR', residual, 'residual'), ('SR', sr, 'lst')]:
                s = self.stats[key]
                result[name] = np.clip((value-s['mean'])/(3*s['std']), -1, 1).astype(np.float32)
        else:
            result['residual_c'] = residual.copy()
        if hls is not None:
            result['HLS'] = hls.copy()
        return result


def epoch_order(dataset, epoch):
    return dataset.lst.epoch_indices(epoch, seed=20260914).tolist()


def fingerprint_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check_stats(stats, dataset):
    assert stats['protocol'] == PROTOCOL
    assert stats['fit_split'] == 'train' and stats['records'] == COUNTS['train']
    assert stats['train_fingerprint'] == dataset.lst.fingerprint
    for key in ('lst', 'residual'):
        assert np.isfinite(stats[key]['mean']) and stats[key]['std'] > 0
