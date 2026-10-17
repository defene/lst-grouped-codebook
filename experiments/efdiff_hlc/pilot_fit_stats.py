import numpy as np
from torch.utils.data import DataLoader
from pilot_data import HLCPairs, PROTOCOL, HERE, sha


def fit(root, workers=8):
    ds = HLCPairs(root, load_hls=False)
    sums = np.zeros((2,2), dtype=np.float64)
    count = 0
    for batch in DataLoader(ds, batch_size=32, num_workers=workers):
        for i, mask in enumerate(batch['mask'].numpy()):
            assert mask.any()
            for j,key in enumerate(('condition_c','residual_c')):
                v = batch[key][i].numpy()[mask].astype(np.float64)
                sums[j] += [v.mean(), np.square(v).mean()]
        count += len(batch['index'])
        if count % 1024 == 0:
            print(f'Pilot normalization {count}/{len(ds)}', flush=True)
    assert count == len(ds)
    result = {'protocol': PROTOCOL, 'fit_split': 'train', 'records': count,
        'train_fingerprint': ds.lst.fingerprint,
        'membership_sha256': ds.fingerprint['membership_sha256'],
        'data_code_sha256': {p:sha(HERE/p) for p in ('data.py','pilot_data.py','pilot_fit_stats.py')},
        'weighting': 'equal patch-months; equal valid pixels within each patch-month', 'crop': 'center224'}
    for key,(mean,second) in zip(('lst','residual'),sums/count):
        result[key] = {'mean': float(mean), 'std': float(np.sqrt(max(second-mean*mean,1e-12)))}
    return result
