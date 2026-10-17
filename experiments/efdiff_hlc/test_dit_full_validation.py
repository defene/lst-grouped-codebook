import contextlib
import math
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dit_full_validation as validation


def test_full_validation_resume_preserves_rng_members_and_tail(tmp_path, monkeypatch):
    generator = torch.Generator
    randn = torch.randn
    monkeypatch.setattr(torch, "randn", lambda *a, **k: randn(*a, **dict(k, device="cpu")))
    loader = torch.utils.data.DataLoader
    monkeypatch.setattr(torch, 'Generator', lambda **kwargs: generator(device='cpu'))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch, 'autocast', lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch.random, 'fork_rng', lambda **k: contextlib.nullcontext())
    monkeypatch.setattr(validation, 'move', lambda b: b)
    def load(ds, **kwargs):
        kwargs.update(num_workers=0, pin_memory=False)
        return loader(ds, **kwargs)
    monkeypatch.setattr(validation, 'DataLoader', load)
    class Dataset:
        fingerprint = {'test': '131-distinct-members'}
        def __len__(self): return 131
        def __getitem__(self, i):
            return dict(HR=torch.zeros(1,2,2), condition_c=torch.zeros(1,2,2),
                        target_c=torch.zeros(1,2,2), mask=torch.ones(1,2,2).bool(),
                        index=i, record_id=i, month=i)
    class Net(torch.nn.Module):
        def sample(self, b, steps, noise): return noise
    common = dict(net=Net(), val=Dataset(), stats={'residual': {'std': 1/3, 'mean': 0}},
                  steps=5, limit=131, commit=lambda: None)
    expected = validation.evaluate(**common, capture_path=tmp_path/'all.npz',
                   progress_path=tmp_path/'all.pt', deadline=float('inf'))
    with pytest.raises(validation.EvaluationPending):
        validation.evaluate(**common, capture_path=tmp_path/'resumed.npz',
                            progress_path=tmp_path/'partial.pt', deadline=0)
    saved = torch.load(tmp_path/'partial.pt', weights_only=False)
    assert saved['records'] == 128
    actual = validation.evaluate(**common, capture_path=tmp_path/'resumed.npz',
                    progress_path=tmp_path/'partial.pt', deadline=float('inf'))
    for key in ('records', 'valid_pixels', 'rmse_c', 'mae_c', 'squared_error_sum', 'absolute_error_sum'):
        assert actual[key] == expected[key]
    a, b = np.load(tmp_path/'all.npz'), np.load(tmp_path/'resumed.npz')
    for key in a.files: np.testing.assert_array_equal(a[key], b[key])
    assert b['index'][-1] == 130


@pytest.mark.parametrize('old_steps', [2500, 3000, 3500, 4000])
def test_mixed_batch_epoch_budget(old_steps):
    n = 2608772
    offset = old_steps*64
    remaining = math.ceil((n-offset)/128)
    total_steps = old_steps+remaining+math.ceil(n/128)
    assert offset+(remaining-1)*128+4 == n
    assert (math.ceil(n/128)-1)*128+4 == n
    assert total_steps == 40764+old_steps//2
