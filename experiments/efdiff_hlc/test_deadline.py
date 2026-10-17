import numpy as np
import torch
from types import SimpleNamespace
from deadline_train import RankSampler, config
from data import EpochSampler, HLCPairs, HERE
from epoch1_train import apply_update


def test_distributed_membership_and_tail():
    ds = HLCPairs(HERE.parents[1], load_hls=False)
    original = EpochSampler(ds, 0).order
    ranks = [RankSampler(ds, r).indices for r in (0,1)]
    assert all(len(x) == 1304386 for x in ranks)
    reconstructed = np.concatenate([np.concatenate([x[a:a+32] for x in ranks])
        for a in range(0,len(ranks[0]),32)])
    np.testing.assert_array_equal(reconstructed, original)
    assert len(np.unique(reconstructed)) == len(ds)
    for r in (0,1):
        np.testing.assert_array_equal(RankSampler(ds,r,start=2608768).indices,
                                      original[-4+2*r:][:2])


def test_global_four_record_gradient_weight():
    models = [torch.nn.Linear(1,1,bias=False) for _ in range(3)]
    for m in models:
        m.weight.data.fill_(.1)
    x = torch.tensor([[.1],[.2],[.3],[.4]])
    for r in (0,1):
        (models[r](x[r*2:(r+1)*2]).square().mean()*2).backward()
    # DDP averages local gradient sums; runner divides by actual local count.
    models[0].weight.grad = (models[0].weight.grad+models[1].weight.grad)/2
    apply_update(SimpleNamespace(denoise_fn=models[0]),
                 torch.optim.SGD(models[0].parameters(),lr=.1),2)
    models[2](x).square().mean().backward()
    torch.optim.SGD(models[2].parameters(),lr=.1).step()
    torch.testing.assert_close(models[0].weight,models[2].weight)


def test_shared_distributed_budget():
    a,b = config('x0'),config('epsilon')
    assert a['data'] == b['data'] and a['normalization'] == b['normalization']
    for c in (a,b):
        assert c['train']['world_size'] == 2
        assert c['train']['micro_batch']*c['train']['gradient_accumulation_per_rank']*2 == 64
        assert c['train']['record_presentations'] == 2608772
        assert c['train']['total_updates'] == 40763
        assert c['smoke']['records'] == 132
