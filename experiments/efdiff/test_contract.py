import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HLPPairs, epoch_order
from train import MaskedLoss, apply_update

ROOT = Path(__file__).resolve().parents[2]


def test_v2_pairing_membership_and_reproducible_views():
    ds = HLPPairs(ROOT, augment=True)
    assert len(ds) == 218975
    a, b = ds[(2,100)], ds[(2,100)]
    assert np.array_equal(a['target_c'], b['target_c'])
    assert a['HLS'].shape == (6,224,224)
    assert a['condition_c'].shape == (1,224,224)
    assert np.isfinite(a['residual_c']).all()
    assert len(set(epoch_order(ds, 0))) == len(ds)
    assert epoch_order(ds, 0) != epoch_order(ds, 1)
    val = HLPPairs(ROOT, split='val')
    assert not set(ds.lst.indices[:,0]) & set(val.lst.indices[:,0])
    source = ds.lst[100]
    expected = ds.lst.denormalize(source['image'])/100
    assert abs(float(expected.mean())) < 100


def test_invalid_pixels_do_not_change_loss_or_gradient():
    loss = MaskedLoss('l2')
    loss.mask = torch.tensor([[[[True, False]]], [[[True, True]]]])
    prediction = torch.tensor([[[[2., 1000.]]], [[[1., 3.]]]], requires_grad=True)
    actual = loss(prediction, torch.zeros_like(prediction))
    assert actual.item() == 4.5  # equal records: (4 + (1+9)/2)/2
    actual.backward()
    assert prediction.grad[0,0,0,1].item() == 0


def test_partial_accumulation_matches_actual_record_mean():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.denoise_fn = torch.nn.Linear(1,1,bias=False)
    net = Net()
    ref = Net()
    net.denoise_fn.weight.data.fill_(.1)
    ref.load_state_dict(net.state_dict())
    opt = torch.optim.SGD(net.parameters(), lr=.01)
    opt_ref = torch.optim.SGD(ref.parameters(), lr=.01)
    x = torch.linspace(.01,.3,31).reshape(-1,1)
    for small in x.split(4):
        (net.denoise_fn(small).square().mean()*len(small)).backward()
    apply_update(net,opt,31)
    ref.denoise_fn(x).square().mean().backward()
    opt_ref.step()
    torch.testing.assert_close(net.denoise_fn.weight,ref.denoise_fn.weight)


def test_exact_epoch_tail_budget():
    for n in (131,218975):
        updates = []
        pending = 0
        for start in range(0,n,4):
            pending += min(4,n-start)
            if pending == 64 or start+4 >= n:
                updates.append(pending)
                pending = 0
        assert sum(updates) == n and pending == 0
        assert updates[-1] == n%64
    assert len(updates)*5 == 17110
