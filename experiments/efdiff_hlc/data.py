"""HLC calendar-month samples with synthetic (never MODIS) coarse LST."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent/'efdiff/upstream'
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(UPSTREAM))
from eo_data import open_dataset
from util.downsampling import walds_protocol_downsample

PROTOCOL = 'efdiff-hlc-v1-synthetic-wald32-224-v1'
COUNTS = {'train': 2608772, 'val': 141711, 'test': 141524}


class HLCPairs(Dataset):
    def __init__(self, root, split='train', stats=None, augment=False, load_hls=True, seed=20260915):
        root = Path(root).resolve()
        self.lst = open_dataset('lstsr_tb', 'lst', split, root=root, prepared=root/'processed_data/v1')
        self.hls = open_dataset('lstsr_tb', 'hls', split, root=root, prepared=root/'processed_data/v1') if load_hls else None
        assert len(self.lst) == COUNTS[split]
        rids = self.lst.indices[:,0]
        assert np.all(rids[1:] >= rids[:-1])
        self.starts = np.r_[0, np.flatnonzero(rids[1:] != rids[:-1])+1, len(rids)]
        if self.hls is not None:
            hids = self.hls.indices[:,0]
            assert len(np.unique(hids)) == len(hids) and np.all(hids[1:] > hids[:-1])
            self.hls_positions = np.searchsorted(hids, rids)
            assert np.all(self.hls_positions < len(hids))
            assert np.array_equal(hids[self.hls_positions], rids), 'Month-to-HLS pairing drift'
        self.stats, self.augment, self.seed, self.split = stats, augment, seed, split
        self.fingerprint = {'protocol': PROTOCOL, 'lst': self.lst.fingerprint,
                            'hls': self.hls.fingerprint if self.hls else None}

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, item):
        epoch, index = item if isinstance(item, tuple) else (0, int(item))
        source = self.lst[index]
        fine = self.lst.denormalize(source['image'])[0]/100.0
        mask = source['reference_mask'][0]
        fine = np.where(mask, fine, fine[mask].mean())
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index]))
        top,left = rng.integers(0,33,2) if self.augment else (16,16)
        fine, mask = fine[top:top+224,left:left+224], mask[top:top+224,left:left+224]
        optical = None
        if self.hls is not None:
            hs = self.hls[int(self.hls_positions[index])]
            assert hs['tile_id'] == source['tile_id']
            optical = np.where(hs['reference_mask'], np.clip(self.hls.denormalize(hs['image']),0,1),0)
            optical = optical[:,top:top+224,left:left+224]
        if self.augment:
            for axis in (0,1):
                if rng.random() < .5:
                    fine,mask = np.flip(fine,axis),np.flip(mask,axis)
                    if optical is not None:
                        optical = np.flip(optical,axis+1)
        coarse = walds_protocol_downsample(np.ascontiguousarray(fine),32).astype(np.float32)
        condition = torch.nn.functional.interpolate(torch.from_numpy(coarse.copy())[None,None],
                size=(224,224),mode='bicubic',align_corners=False)[0].numpy()
        residual = fine[None]-condition
        result = {'target_c':fine[None].copy(), 'condition_c':condition.copy(),
                  'mask':mask[None].copy(), 'index':index, 'month':source['month'],
                  'record_id':int(self.lst.indices[index,0]), 'synthetic_coarse_c':coarse[None].copy()}
        if self.stats:
            for key,value,kind in [('HR',residual,'residual'),('SR',condition,'lst')]:
                s = self.stats[kind]
                result[key] = np.clip((value-s['mean'])/(3*s['std']),-1,1).astype(np.float32)
        else:
            result['residual_c'] = residual.copy()
        if optical is not None:
            result['HLS'] = optical.copy()
        return result


class EpochSampler(Sampler):
    """Shuffle bundles and their valid months, visit every month exactly once."""
    def __init__(self, ds, epoch, start=0, limit=None):
        self.ds,self.epoch,self.start,self.limit = ds,epoch,start,limit
        rng = np.random.default_rng(np.random.SeedSequence([ds.seed,epoch,321]))
        pieces = [rng.permutation(np.arange(ds.starts[b],ds.starts[b+1],dtype=np.int32))
                  for b in rng.permutation(len(ds.starts)-1)]
        self.order = np.concatenate(pieces)[:limit]

    def __iter__(self):
        return ((self.epoch,int(i)) for i in self.order[self.start:])

    def __len__(self):
        return len(self.order)-self.start


def check_stats(stats, ds):
    assert stats['protocol'] == PROTOCOL and stats['fit_split'] == 'train'
    assert stats['records'] == COUNTS['train'] and stats['train_fingerprint'] == ds.lst.fingerprint
    assert stats['data_code_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for key in ('lst','residual'):
        assert np.isfinite(stats[key]['mean']) and np.isfinite(stats[key]['std']) and stats[key]['std'] > 0
