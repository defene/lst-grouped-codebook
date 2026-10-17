import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pilot_data import HLCPairs, EpochSampler, MEMBERS, COUNTS, generate, sha
from pilot_train import config
ROOT = Path(__file__).resolve().parents[2]


def test_fixed_membership_reproducibility_and_isolation():
    before = (MEMBERS/'summary.json').read_bytes()
    generate(ROOT)
    assert before == (MEMBERS/'summary.json').read_bytes()
    train,val = HLCPairs(ROOT),HLCPairs(ROOT,split='val')
    assert len(train)==16384 and len(val)==512
    assert not set(train.lst.indices[train.members,0]) & set(val.lst.indices[val.members,0])
    for ds in (train,val):
        rid_tile = dict(ds.lst._connect().execute('SELECT id,tile FROM records WHERE split=?',(ds.split,)))
        covered = {rid_tile[int(r)] for r in ds.lst.indices[ds.members,0]}
        assert len(covered) == (2249 if ds.split=='train' else 118)
        assert len(np.unique(ds.members)) == len(ds)
    with pytest.raises(AssertionError):
        HLCPairs(ROOT,split='test')


def test_subset_mapping_order_resume_and_augmentation():
    ds = HLCPairs(ROOT,augment=True)
    order = EpochSampler(ds,2).order
    np.testing.assert_array_equal(np.sort(order),np.arange(16384))
    assert list(EpochSampler(ds,2,start=128,limit=131)) == [(2,int(i)) for i in order[128:131]]
    source_index = int(ds.members[100])
    sample = ds[(2,100)]
    assert sample['index']==source_index
    assert sample['record_id']==int(ds.lst.indices[source_index,0])
    np.testing.assert_array_equal(sample['target_c'],ds[(2,100)]['target_c'])
    assert sample['HLS'].shape==(6,224,224) and sample['synthetic_coarse_c'].shape==(1,7,7)


def test_pilot_budget_and_shared_controls():
    a,b = config('x0'),config('epsilon')
    assert a['data']==b['data'] and a['normalization']==b['normalization']
    assert a['model']['unet']==b['model']['unet'] and a['model']['prithvi']==b['model']['prithvi']
    for c in (a,b):
        assert c['train']['total_updates']==5*(16384//64)==1280
        assert c['train']['record_presentations']==81920
        assert c['train']['warmup_steps']==c['train']['ema_start_step']==100
        assert c['train']['ema_decay']==.99
        assert c['evaluation']['monitor_samples']==512 and c['evaluation']['final_splits']==['val']
        assert not c['data']['real_modis_input'] and c['data']['test_samples']==0
