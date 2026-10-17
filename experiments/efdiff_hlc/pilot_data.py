"""Versioned, tile-balanced HLC pilot membership, fixed before any model fitting."""
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from data import HLCPairs as FullPairs, EpochSampler, HERE, UPSTREAM

PROTOCOL = 'efdiff-hlc-synthetic-pilot-16384-512-v1'
COUNTS = {'train': 16384, 'val': 512}
MEMBERS = HERE/'pilot_membership'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(root):
    """Equal quotas by tile; random remainder tiles and uniform months within tile."""
    bundle = Path(root)/'processed_data/v1/lstsr_tb'
    MEMBERS.mkdir(exist_ok=True)
    summary = {'protocol': PROTOCOL, 'seed': 20260915,
               'method': 'equal tile quotas plus seeded random remainder; uniform months without replacement within tile',
               'test_used': False, 'splits': {}}
    identities = {}
    with sqlite3.connect(f'file:{(bundle/"index.sqlite").as_posix()}?mode=ro', uri=True) as db:
        for split, count in COUNTS.items():
            index = np.load(bundle/'indices'/f'{split}.lst_hr.npy')
            rid_tile = dict(db.execute('SELECT id,tile FROM records WHERE split=?', (split,)))
            tiles = sorted(set(rid_tile.values()))
            tile_num = {t:i for i,t in enumerate(tiles)}
            labels = np.array([tile_num[rid_tile[int(r)]] for r in index[:,0]], dtype=np.int32)
            grouped = np.argsort(labels, kind='stable')
            boundaries = np.r_[0, np.cumsum(np.bincount(labels, minlength=len(tiles)))]
            rng = np.random.default_rng(np.random.SeedSequence([20260915, 0 if split=='train' else 1]))
            capacities = np.diff(boundaries)
            quotas = np.zeros(len(tiles), dtype=int)
            remaining = count
            while remaining:
                available = rng.permutation(np.flatnonzero(quotas < capacities))[:remaining]
                assert len(available)
                quotas[available] += 1
                remaining -= len(available)
            chosen = np.sort(np.concatenate([
                rng.choice(grouped[boundaries[i]:boundaries[i+1]], int(q), replace=False)
                for i,q in enumerate(quotas)] )).astype(np.int64)
            assert len(chosen) == count == len(np.unique(chosen))
            path = MEMBERS/f'{split}.npy'
            np.save(path, chosen)
            with (MEMBERS/f'{split}.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['source_index','record_id','month_slot','tile_id'])
                writer.writerows((int(i),int(index[i,0]),int(index[i,1]),rid_tile[int(index[i,0])]) for i in chosen)
            identities[split] = (set(map(int,index[chosen,0])), {rid_tile[int(r)] for r in index[chosen,0]})
            summary['splits'][split] = {'samples': count, 'tiles': len(tiles),
                'min_months_per_tile': int(quotas.min()), 'max_months_per_tile': int(quotas.max()),
                'membership_sha256': sha(path), 'csv_sha256': sha(MEMBERS/f'{split}.csv'),
                'source_index_sha256': sha(bundle/'indices'/f'{split}.lst_hr.npy')}
    assert not identities['train'][0] & identities['val'][0]
    assert not identities['train'][1] & identities['val'][1]
    (MEMBERS/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


class HLCPairs(FullPairs):
    def __init__(self, root, split='train', **kwargs):
        assert split in COUNTS, 'Pilot does not use test data'
        super().__init__(root, split=split, **kwargs)
        self.members = np.load(MEMBERS/f'{split}.npy')
        expected = json.loads((MEMBERS/'summary.json').read_text())['splits'][split]
        assert sha(MEMBERS/f'{split}.npy') == expected['membership_sha256']
        assert len(self.members) == COUNTS[split]
        assert np.all(self.members[1:] > self.members[:-1])
        source_path = Path(root)/'processed_data/v1/lstsr_tb/indices'/f'{split}.lst_hr.npy'
        assert sha(source_path) == expected['source_index_sha256']
        rids = self.lst.indices[self.members,0]
        self.starts = np.r_[0, np.flatnonzero(rids[1:] != rids[:-1])+1, len(rids)]
        self.fingerprint.update(protocol=PROTOCOL, membership_sha256=expected['membership_sha256'],
                                samples=len(self.members))

    def __len__(self):
        return len(self.members)

    def __getitem__(self, item):
        epoch, position = item if isinstance(item, tuple) else (0, int(item))
        return super().__getitem__((epoch, int(self.members[position])))


def check_stats(stats, ds):
    assert stats['protocol'] == PROTOCOL and stats['fit_split'] == 'train'
    assert stats['records'] == COUNTS['train']
    assert stats['train_fingerprint'] == ds.lst.fingerprint
    assert stats['membership_sha256'] == ds.fingerprint['membership_sha256']
    assert stats['data_code_sha256'] == {p:sha(HERE/p) for p in ('data.py','pilot_data.py','pilot_fit_stats.py')}
    for key in ('lst','residual'):
        assert np.isfinite(stats[key]['mean']) and np.isfinite(stats[key]['std']) and stats[key]['std'] > 0


if __name__ == '__main__':
    print(json.dumps(generate(HERE.parents[1]), indent=2))
