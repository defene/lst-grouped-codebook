import hashlib
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Subset
from data import HLCPairs, COUNTS, PROTOCOL, HERE


def fit_shard(root, shard, shards, workers=8):
    ds = HLCPairs(root, load_hls=False)
    boundaries = np.linspace(0,len(ds.starts)-1,shards+1,dtype=int)
    start,end = map(int,ds.starts[boundaries[shard:shard+2]])
    sums = np.zeros((2,2),dtype=np.float64)
    count = 0
    for batch in DataLoader(Subset(ds,range(start,end)),batch_size=64,num_workers=workers):
        for i, mask in enumerate(batch['mask'].numpy()):
            assert mask.any()
            for j,key in enumerate(('condition_c','residual_c')):
                v = batch[key][i].numpy()[mask].astype(np.float64)
                sums[j] += [v.mean(),np.square(v).mean()]
        count += len(batch['index'])
        if count % 16384 == 0:
            print(f'HLC stats {shard}: {count}/{end-start}',flush=True)
    return {'shard':shard,'shards':shards,'start':start,'end':end,'records':count,
            'sums':sums.tolist(),'train_fingerprint':ds.lst.fingerprint,
            'data_code_sha256':hashlib.sha256((HERE/'data.py').read_bytes()).hexdigest()}


def merge(parts):
    parts = sorted(parts,key=lambda p:p['shard'])
    assert [p['shard'] for p in parts] == list(range(parts[0]['shards']))
    assert parts[0]['start'] == 0 and parts[-1]['end'] == COUNTS['train']
    assert all(a['end'] == b['start'] for a,b in zip(parts,parts[1:]))
    assert all(p['end']-p['start'] == p['records'] for p in parts)
    assert all(p['train_fingerprint'] == parts[0]['train_fingerprint'] for p in parts)
    assert all(p['data_code_sha256'] == parts[0]['data_code_sha256'] for p in parts)
    moments = np.sum([p['sums'] for p in parts],axis=0)/COUNTS['train']
    result = {k:parts[0][k] for k in ('train_fingerprint','data_code_sha256')}
    result.update(protocol=PROTOCOL,fit_split='train',records=COUNTS['train'],
                  weighting='equal patch-months, equal valid pixels within patch-month',crop='center224')
    for key,(mean,second) in zip(('lst','residual'),moments):
        result[key] = {'mean':float(mean),'std':float(np.sqrt(max(second-mean*mean,1e-12)))}
    return result
