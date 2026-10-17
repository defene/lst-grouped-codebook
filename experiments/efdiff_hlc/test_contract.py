import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data import HLCPairs, EpochSampler, COUNTS
from train import config, MaskedLoss, apply_update
ROOT = Path(__file__).resolve().parents[2]


def test_pairing_months_and_exact_order():
    ds = HLCPairs(ROOT,augment=True)
    assert len(ds) == 2608772
    assert len(ds.hls) == 250449
    a,b = ds[(0,0)],ds[(0,1)]
    assert a['record_id'] == b['record_id'] and a['month'] != b['month']
    assert np.array_equal(ds[(2,100)]['target_c'],ds[(2,100)]['target_c'])
    order = EpochSampler(ds,0).order
    assert np.array_equal(np.sort(order),np.arange(len(ds)))
    np.testing.assert_array_equal(EpochSampler(ds,0,start=128).order,order)
    assert list(EpochSampler(ds,0,start=128,limit=131)) == [(0,int(i)) for i in order[128:131]]


def test_synthetic_coarse_from_reference_and_masked_loss():
    from util.downsampling import walds_protocol_downsample
    ds = HLCPairs(ROOT,split='val')
    sample = ds[100]
    expected = walds_protocol_downsample(sample['target_c'][0],32)
    np.testing.assert_allclose(sample['synthetic_coarse_c'][0],expected,rtol=1e-6)
    assert sample['HLS'].shape == (6,224,224)
    loss = MaskedLoss('l2')
    loss.mask = torch.tensor([[[[True,False]]]])
    pred = torch.tensor([[[[2.,999.]]]],requires_grad=True)
    loss(pred,torch.zeros_like(pred)).backward()
    assert pred.grad[0,0,0,1] == 0


def test_identical_data_budget_and_architecture_between_variants():
    a,b = config('x0'),config('epsilon')
    assert a['data'] == b['data'] and a['normalization'] == b['normalization']
    assert a['model']['unet'] == b['model']['unet'] and a['model']['prithvi'] == b['model']['prithvi']
    for c in (a,b):
        n = COUNTS['train']
        assert c['train']['total_updates'] == 5*((n+63)//64) == 203815
        assert n%64 == c['train']['tail_records'] == 4
        assert c['train']['record_presentations'] == 5*n == 13043860
        assert not c['data']['real_modis_input']


def test_partial_accumulation():
    from types import SimpleNamespace
    actual,expected = torch.nn.Linear(1,1,bias=False),torch.nn.Linear(1,1,bias=False)
    actual.weight.data.fill_(.1)
    expected.load_state_dict(actual.state_dict())
    opt = torch.optim.SGD(actual.parameters(),lr=.01)
    ref = torch.optim.SGD(expected.parameters(),lr=.01)
    x = torch.linspace(.01,.3,19).reshape(-1,1)
    for micro in x.split(16):
        (actual(micro).square().mean()*len(micro)).backward()
    apply_update(SimpleNamespace(denoise_fn=actual),opt,len(x))
    expected(x).square().mean().backward()
    ref.step()
    torch.testing.assert_close(actual.weight,expected.weight)
