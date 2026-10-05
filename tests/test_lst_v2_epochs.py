import math
import numpy as np
import torch
from eo_denoise.data import EpochSampler
from experiments.paper1.lst_v2_5epoch.data import epoch_updates


def test_full_five_epochs_exact_membership_and_tail():
    n = 218975
    steps = 0
    for epoch in range(5):
        order = [i for e, i in EpochSampler(n, 17, epoch)]
        micros = [order[i:i+4] for i in range(0, n, 4)]
        updates = list(epoch_updates(iter(micros), 16))
        flat = [i for batches in updates for b in batches for i in b]
        assert flat == order and sorted(flat) == list(range(n))
        assert len(updates) == 3422
        assert sum(map(len, updates[-1])) == 31
        steps += len(updates)
    assert steps == 17110


def test_resume_order_at_optimizer_boundary_and_small_tail():
    order = [i for _, i in EpochSampler(67, 17, 1)]
    tail = [i for _, i in EpochSampler(67, 17, 1, 64)]
    assert order[64:] == tail
    updates = list(epoch_updates(iter([order[i:i+4] for i in range(0, 67, 4)]), 16))
    assert [sum(map(len, b)) for b in updates] == [64, 3]


def test_partial_accumulation_actual_sample_weight_matches_full_loss():
    x = torch.arange(31, dtype=torch.float64)/31
    full = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    ((full*x-1)**2).mean().backward()
    micro = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    for values in x.split(4):
        ((((micro*values-1)**2).mean())*len(values)/len(x)).backward()
    assert torch.allclose(full.grad, micro.grad, atol=1e-12, rtol=0)
