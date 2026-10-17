import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vscore_diffusion import coefficients, predictions, objective_loss, time_grid, VScoreDiffusion


@pytest.mark.parametrize('objective', ['v', 'score'])
def test_oracle_reconstruction_and_zero_loss(objective):
    torch.manual_seed(7)
    x0, eps = torch.randn(4, 1, 7, 7), torch.randn(4, 1, 7, 7)
    a = torch.tensor([.9999995, .95, .5, .081]).view(4, 1, 1, 1)
    s = (1 - a.square()).sqrt()
    xt = a * x0 + s * eps
    raw = a * eps - s * x0 if objective == 'v' else -eps / s
    pred_x, pred_eps = predictions(raw, xt, a, s, objective)
    torch.testing.assert_close(pred_x, x0, atol=5e-6, rtol=1e-5)
    torch.testing.assert_close(pred_eps, eps, atol=5e-6, rtol=1e-5)
    assert objective_loss(raw, x0, eps, a, s, torch.ones_like(x0).bool(), objective) < 1e-12


def test_score_weighting_and_gradient_identity():
    torch.manual_seed(11)
    raw = torch.randn(3, 1, 8, 8, requires_grad=True)
    eps = torch.randn_like(raw)
    s = torch.tensor([.001, .3, .99]).view(3, 1, 1, 1)
    a = (1 - s.square()).sqrt()
    actual = objective_loss(raw, eps, eps, a, s, torch.ones_like(raw).bool(), 'score')
    expected = (s.square() * (raw + eps / s).square()).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, raw)[0],
                               torch.autograd.grad(expected, raw)[0])


@pytest.mark.parametrize('objective', ['v', 'score'])
def test_mask_equal_record_weight_and_finite_backward(objective):
    raw = torch.ones(2, 1, 4, 4, requires_grad=True)
    x0, eps = torch.zeros_like(raw), torch.zeros_like(raw)
    mask = torch.ones_like(raw).bool()
    mask[0, :, 1:] = False
    a = torch.tensor([.8, .6]).view(2, 1, 1, 1)
    s = (1 - a.square()).sqrt()
    loss = objective_loss(raw, x0, eps, a, s, mask, objective)
    expected = torch.tensor(1.) if objective == 'v' else s.square().mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    assert raw.grad[~mask].count_nonzero() == 0


@pytest.mark.parametrize('steps', [1, 5, 10, 20, 50, 1000])
def test_exact_nfe_grid(steps):
    grid = time_grid(steps)
    assert len(grid) == steps and grid[0][0] == 999 and grid[-1][1] == -1
    assert all(t > p for t, p in grid)


@pytest.mark.parametrize('objective', ['v', 'score'])
def test_actual_sampler_oracle_and_call_count(objective):
    class Oracle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, inputs, level):
            self.calls += 1
            cond, xt = inputs.chunk(2, 1)
            a = level.view(-1, 1, 1, 1)
            ids = (net.alpha_bar.sqrt()[:, None] - level[:, 0]).abs().argmin(0)
            s = (1 - net.alpha_bar[ids]).sqrt().view(-1, 1, 1, 1)
            eps = (xt - a * cond) / s
            return a * eps - s * cond if objective == 'v' else -eps / s
    denoiser = Oracle()
    net = VScoreDiffusion(denoiser, None, objective)
    target = torch.rand(2, 1, 4, 4) * 1.8 - .9
    for steps in [5, 10, 20, 50]:
        denoiser.calls = 0
        result = net.sample({'SR': target}, steps, torch.randn_like(target))
        assert denoiser.calls == steps
        torch.testing.assert_close(result, target, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('objective', ['v', 'score'])
def test_saved_optimizer_and_rng_resume_matches_uninterrupted(tmp_path, objective):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Conv2d(2, 1, 1)
        def forward(self, x, level):
            return self.layer(x) + level.view(-1, 1, 1, 1) * .01
    torch.manual_seed(27)
    net = VScoreDiffusion(Tiny(), None, objective)
    opt = torch.optim.Adam(net.parameters(), lr=2e-5)
    batch = dict(HR=torch.rand(4,1,4,4), SR=torch.rand(4,1,4,4), mask=torch.ones(4,1,4,4).bool())
    def update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch)
        loss.backward()
        optimizer.step()
        return loss.detach()
    update(net, opt)
    path = tmp_path/'state.pt'
    torch.save(dict(model=net.state_dict(),opt=opt.state_dict(),rng=torch.get_rng_state()), path)
    expected = update(net, opt)
    resumed = VScoreDiffusion(Tiny(), None, objective)
    new_opt = torch.optim.Adam(resumed.parameters(),lr=2e-5)
    state = torch.load(path,weights_only=False)
    resumed.load_state_dict(state['model'])
    new_opt.load_state_dict(state['opt'])
    torch.set_rng_state(state['rng'])
    actual = update(resumed,new_opt)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for key in net.state_dict():
        torch.testing.assert_close(net.state_dict()[key],resumed.state_dict()[key],rtol=0,atol=0)


def test_actual_full_membership_and_tail():
    from types import SimpleNamespace
    import numpy as np
    # Exercise the production epoch-order algorithm without decoding data.
    from data import EpochSampler
    n = 2608772
    ds = SimpleNamespace(seed=20260927,starts=np.r_[np.arange(0,n,12),n])
    full = EpochSampler(ds,0).order
    assert len(full)==n and np.array_equal(np.sort(full),np.arange(n))
    tail = list(EpochSampler(ds,0,start=2608768))
    assert len(tail)==4 and [i for ep,i in tail]==full[-4:].tolist()
