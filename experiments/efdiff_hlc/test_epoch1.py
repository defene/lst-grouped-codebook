import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from epoch1_train import config,apply_update
from data import HLCPairs,EpochSampler,HERE
from pilot_data import HLCPairs as ValidationPairs
from epoch1_stats import merge
ROOT=HERE.parents[1]


def test_full_membership_final_tail_and_resume():
    ds=HLCPairs(ROOT,load_hls=False)
    order=EpochSampler(ds,0).order
    assert len(ds)==2608772 and np.array_equal(np.sort(order),np.arange(len(ds)))
    remaining=list(EpochSampler(ds,0,start=2608768))
    assert len(remaining)==4
    assert [i for _,i in remaining]==order[-4:].tolist()
    assert all(epoch==0 for epoch,_ in remaining)
    val=ValidationPairs(ROOT,split='val',load_hls=False)
    assert len(val)==512 and val.fingerprint['membership_sha256']=='d5dc287867c7eb7aed1bf6a8641679ae8ad7d9926f62e9a6ef78247718629f78'


def test_exact_one_epoch_budget_and_shared_controls():
    a,b=config('x0'),config('epsilon')
    assert a['data']==b['data'] and a['normalization']==b['normalization']
    assert a['model']['unet']==b['model']['unet']
    budget=json.loads((HERE/'epoch1_budget.json').read_text())
    for cfg in (a,b):
        assert cfg['train']['epochs']==budget['epochs']==1
        assert cfg['train']['total_updates']==budget['updates']==40763
        assert cfg['train']['record_presentations']==2608772
        assert cfg['train']['tail_records']==4
        assert cfg['evaluation']['final_splits']==['val']
        assert cfg['evaluation']['primary_checkpoint']=='fixed_epoch1_ema'
        assert cfg['train']['segment_seconds']==14400
        assert cfg['data']['real_modis_input'] is False


def test_four_sample_tail_uses_actual_weight():
    actual=torch.nn.Linear(1,1,bias=False)
    expected=torch.nn.Linear(1,1,bias=False)
    actual.weight.data.fill_(.1)
    expected.load_state_dict(actual.state_dict())
    x=torch.tensor([[.1],[.2],[.3],[.4]])
    opt=torch.optim.SGD(actual.parameters(),lr=.1)
    ref=torch.optim.SGD(expected.parameters(),lr=.1)
    (actual(x).square().mean()*4).backward()
    apply_update(SimpleNamespace(denoise_fn=actual),opt,4)
    expected(x).square().mean().backward()
    ref.step()
    torch.testing.assert_close(actual.weight,expected.weight)


def test_statistics_reject_missing_shards():
    import hashlib
    part={'stats_code_sha256':hashlib.sha256((HERE/'epoch1_stats.py').read_bytes()).hexdigest(),
          'shard':0,'shards':32,'start':0,'end':81920,'records':81920}
    with pytest.raises(AssertionError):
        merge([part])
