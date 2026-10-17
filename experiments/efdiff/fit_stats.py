"""Train-only EFDiff residual statistics, merged from disjoint full-coverage shards."""
import numpy as np
from torch.utils.data import DataLoader, Subset
from data import HLPPairs, COUNTS, PROTOCOL


def fit_shard(root, shard, shards, workers=8):
    ds = HLPPairs(root, load_hls=False)
    indices = list(range(shard, len(ds), shards))
    sums = np.zeros((2, 2), dtype=np.float64)
    count = 0
    for batch in DataLoader(Subset(ds, indices), batch_size=16, num_workers=workers):
        masks = batch['mask'].numpy()
        for row in range(len(masks)):
            mask = masks[row]
            assert mask.any(), f'No valid target in center crop {batch["index"][row]}'
            for j, key in enumerate(('condition_c', 'residual_c')):
                values = batch[key][row].numpy()[mask].astype(np.float64)
                sums[j] += [values.mean(), np.square(values).mean()]
        count += len(masks)
        if count % 2048 == 0:
            print(f'stats shard {shard}: {count}/{len(indices)}', flush=True)
    return {'shard': shard, 'shards': shards, 'records': count, 'sums': sums.tolist(),
            'train_fingerprint': ds.lst.fingerprint}


def merge(parts):
    shards = parts[0]['shards']
    assert sorted(p['shard'] for p in parts) == list(range(shards))
    assert all(p['train_fingerprint'] == parts[0]['train_fingerprint'] for p in parts)
    assert all(p['records'] == len(range(p['shard'], COUNTS['train'], shards)) for p in parts)
    n = sum(p['records'] for p in parts)
    assert n == COUNTS['train']
    moments = np.sum([p['sums'] for p in parts], axis=0) / n
    result = {'protocol': PROTOCOL, 'fit_split': 'train', 'records': n,
              'weighting': 'equal records, equal valid target pixels within record',
              'crop': 'center224 for fit; seeded random224 + flips for train',
              'train_fingerprint': parts[0]['train_fingerprint']}
    for key, (mean, second) in zip(('lst', 'residual'), moments):
        result[key] = {'mean': float(mean), 'std': float(np.sqrt(max(second-mean*mean, 1e-12)))}
    return result
