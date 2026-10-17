"""Small full-data shards, cached individually and merged with complete coverage checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader,Subset
from data import HLCPairs,HERE
from fit_stats import merge as merge_base


def fit(root,shard,shards=32,workers=8):
    ds=HLCPairs(root,load_hls=False)
    boundaries=np.linspace(0,len(ds.starts)-1,shards+1,dtype=int)
    start,end=map(int,ds.starts[boundaries[shard:shard+2]])
    count=0
    sums=np.zeros((2,2),dtype=np.float64)
    for batch in DataLoader(Subset(ds,range(start,end)),batch_size=64,num_workers=workers):
        for i,mask in enumerate(batch['mask'].numpy()):
            assert mask.any()
            for j,key in enumerate(('condition_c','residual_c')):
                x=batch[key][i].numpy()[mask].astype(np.float64)
                sums[j]+=[x.mean(),np.square(x).mean()]
        count+=len(batch['index'])
        if count%8192==0:
            print(f'Full epoch1 stats {shard}/{shards}: {count}/{end-start}',flush=True)
    assert count==end-start
    return {'shard':shard,'shards':shards,'start':start,'end':end,'records':count,'sums':sums.tolist(),
            'train_fingerprint':ds.lst.fingerprint,
            'data_code_sha256':hashlib.sha256((HERE/'data.py').read_bytes()).hexdigest(),
            'stats_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def merge(parts):
    assert all(p['stats_code_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest() for p in parts)
    result=merge_base(parts)
    result['stats_code_sha256']=parts[0]['stats_code_sha256']
    return result
