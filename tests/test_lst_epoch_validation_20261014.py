"""CPU-only regression tests; no real training data or experiment submissions."""
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.paper1.lst_restoration_epochval_20261014 import config as cfg
from experiments.paper1.lst_restoration_epochval_20261014 import data, models, runtime
from experiments.paper1.lst_restoration_epochval_20261014 import epoch_validation as ev


class ToyDataset(torch.utils.data.Dataset):
    opened = []
    closed = []
    duplicate = False

    def __init__(self, config, split, evaluation_profile=None, view=0):
        self.config, self.split, self.view = config, split, view
        self.profile = evaluation_profile or {'kind': 'pixel10'}
        self.record_ids = np.arange(cfg.COUNTS[split])
        self.fingerprint = {'split': split, 'samples': len(self.record_ids), 'profile': self.profile}
        self.opened.append((split, view, self.profile['kind']))

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, item):
        epoch, index = item if isinstance(item, tuple) else (0, int(item))
        target = torch.zeros(1, 2, 2)
        valid = torch.ones_like(target, dtype=torch.bool)
        if index == 0:
            valid[0, 0, 0] = False
        # Vary masks and errors so averaging image RMSE is visibly incorrect.
        field = torch.full_like(target, .1 * index)
        return {'sample_id': f'{self.split}-{index}', 'record_id': 0 if self.duplicate else index,
                'tile_id': f'tile-{index}', 'temperature': target, 'input_field': field,
                'target_field': target, 'reference_valid': valid, 'observed_valid': valid,
                'mean_celsius': 0., 'center_celsius': 0., 'patch_missing': torch.zeros(1, 1, dtype=torch.bool),
                'profile': self.profile['kind']}

    def close(self):
        self.closed.append((self.split, self.view, self.profile['kind']))


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1.))
        self.sub = torch.nn.Dropout(.2)

    def forward(self, encoded_input, **kwargs):
        noise = torch.randn_like(encoded_input) * .01 if self.training else 0.
        return {'encoded_prediction': encoded_input + self.bias + noise}


@pytest.fixture
def tiny(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(cfg, 'COUNTS', {'train': 5, 'val': 3, 'test': 2})
    monkeypatch.setattr(data, 'LSTDataset', ToyDataset)
    monkeypatch.setattr(models, 'build_model', lambda config: ToyModel())
    monkeypatch.setattr(runtime, 'execution_guard', lambda config: None)
    monkeypatch.setattr(runtime, 'source_hashes', lambda: {'cpu-test': 'fixed'})
    ToyDataset.opened, ToyDataset.closed = [], []
    ToyDataset.duplicate = False
    config = cfg.default_config('ae', 'R')
    config['train'].update(device='cpu', workers=0, effective_batch=4, micro_batch=2,
                           epochs=5, updates=10, record_presentations=25, lr=.01,
                           checkpoint_every=1, amp=False)
    config['evaluation']['batch_size'] = 2
    config['task']['residual_std_celsius'] = 2.
    config['representation']['A_celsius'] = 1.
    return config


def test_full_val_pixel_weighting_views_and_mode_restoration(tiny, tmp_path):
    model = ToyModel()
    model.sub.eval()
    states = [m.training for m in model.modules()]
    before = copy.deepcopy(model.state_dict())
    cells = ev.measure_validation(model, tiny, torch.device('cpu'), tmp_path)
    expected = (3 * 1.**2 + 4 * 1.1**2 + 4 * 1.2**2) / 11
    for cell in cells:
        assert cell['raw']['pixel_weighted_rmse'] == pytest.approx(math.sqrt(expected))
        assert cell['loss_pixel'] == pytest.approx((1. + 1.21 + 1.44) / 3 / 4)
        assert cell['row_count'] == 6 and cell['unique_records'] == 3
        assert cell['valid_pixels'] == 22
    assert [m.training for m in model.modules()] == states
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(split == 'val' for split, _, _ in ToyDataset.opened)
    assert ToyDataset.opened == ToyDataset.closed


@pytest.mark.parametrize('fail', [False, True])
def test_inference_restores_all_rngs_even_on_error(fail):
    model = ToyModel()
    model.sub.eval()
    random.seed(14)
    np.random.seed(14)
    torch.manual_seed(14)
    expected = (random.random(), np.random.random(), torch.rand(2))
    random.seed(14)
    np.random.seed(14)
    torch.manual_seed(14)
    try:
        with ev.isolated_inference(model, torch.device('cpu'), 123):
            random.random(), np.random.random(), torch.rand(33)
            assert not model.training and not torch.is_grad_enabled()
            if fail:
                raise RuntimeError('injected')
    except RuntimeError:
        assert fail
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(2), expected[2])
    assert model.training and not model.sub.training


def test_duplicate_membership_fails_and_closes_dataset(tiny, tmp_path):
    ToyDataset.duplicate = True
    model = ToyModel()
    with pytest.raises(ValueError, match='membership'):
        ev.measure_validation(model, tiny, torch.device('cpu'), tmp_path)
    assert model.training
    assert ToyDataset.opened == ToyDataset.closed


def test_nonfinite_validation_never_commits_receipt(tiny, tmp_path):
    model = ToyModel()
    model.bias.data.fill_(float('nan'))
    (tmp_path / 'last.pt').write_bytes(b'checkpoint')
    with pytest.raises(FloatingPointError):
        ev.finish_epoch(model, tiny, torch.device('cpu'), tmp_path, 1, 2)
    assert not (tmp_path / 'selection.json').exists()
    assert not (tmp_path / 'epoch_validation/epoch_01.json').exists()


def test_five_epochs_exact_tail_and_best_last_retained(tiny, tmp_path):
    runtime.train(tiny, tmp_path)
    assert len(list((tmp_path / 'epochs').glob('*.pt'))) == 5
    assert len(list((tmp_path / 'epoch_validation').glob('*.json'))) == 5
    selection = json.loads((tmp_path / 'selection.json').read_text())
    assert selection['through_epoch'] == 5 and not selection['test_used']
    best = min(selection['history'], key=lambda x: (x['score'], x['epoch']))
    assert selection['best'] == best
    assert cfg.file_digest(tmp_path / 'best.pt') == best['checkpoint_sha256']
    state = torch.load(tmp_path / 'last.pt', weights_only=True)
    assert (state['epoch'], state['records'], state['step']) == (5, 25, 10)
    events = [json.loads(line) for path in tmp_path.glob('train_*.jsonl') for line in path.read_text().splitlines()]
    assert [r['actual_batch'] for r in events] == [4, 1] * 5
    assert len([r for r in ToyDataset.opened if r[0] == 'val']) == 20


@pytest.mark.parametrize('interruption_epoch', [1, 2, 5])
def test_resume_pending_epoch_validation_matches_uninterrupted(tiny, tmp_path, monkeypatch, interruption_epoch):
    complete, interrupted = tmp_path / 'complete', tmp_path / 'interrupted'
    runtime.train(tiny, complete)
    measure = ev.measure_validation
    calls = 0

    def fail_once(*args):
        nonlocal calls
        calls += 1
        if calls == interruption_epoch:
            raise RuntimeError('interrupted validation')
        return measure(*args)

    monkeypatch.setattr(ev, 'measure_validation', fail_once)
    with pytest.raises(RuntimeError, match='interrupted validation'):
        runtime.train(tiny, interrupted)
    state = torch.load(interrupted / 'last.pt', weights_only=True)
    assert state['epoch'] == interruption_epoch and state['consumed'] == 0
    assert not (interrupted / f'epoch_validation/epoch_{interruption_epoch:02d}.json').exists()
    monkeypatch.setattr(ev, 'measure_validation', measure)
    runtime.train(tiny, interrupted)
    a = torch.load(complete / 'last.pt', weights_only=True)
    b = torch.load(interrupted / 'last.pt', weights_only=True)
    assert torch.equal(a['model']['bias'], b['model']['bias'])
    assert torch.equal(a['rng_cpu'], b['rng_cpu'])
    assert (a['records'], a['step']) == (b['records'], b['step'])
    for epoch in range(1, 6):
        one = json.loads((complete / f'epoch_validation/epoch_{epoch:02d}.json').read_text())
        two = json.loads((interrupted / f'epoch_validation/epoch_{epoch:02d}.json').read_text())
        assert one['cells'] == two['cells']


def test_receipt_replay_repairs_best_without_reinference(tiny, tmp_path, monkeypatch):
    runtime.train(tiny, tmp_path)
    expected = json.loads((tmp_path / 'selection.json').read_text())['best']
    (tmp_path / 'best.pt').write_bytes(b'interrupted copy')
    (tmp_path / 'selection.json').unlink()
    monkeypatch.setattr(ev, 'measure_validation', lambda *a: pytest.fail('must reuse completed receipt'))
    runtime.train(tiny, tmp_path)
    assert cfg.file_digest(tmp_path / 'best.pt') == expected['checkpoint_sha256']


def test_mid_epoch_resume_keeps_prior_validation_and_exact_optimizer(tiny, tmp_path, monkeypatch):
    complete, interrupted = tmp_path / 'complete', tmp_path / 'interrupted'
    runtime.train(tiny, complete)
    update = runtime.train_update
    calls = 0

    def interrupted_update(*args):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError('interrupted before epoch 2 tail')
        return update(*args)

    monkeypatch.setattr(runtime, 'train_update', interrupted_update)
    with pytest.raises(RuntimeError, match='epoch 2 tail'):
        runtime.train(tiny, interrupted)
    state = torch.load(interrupted / 'last.pt', weights_only=True)
    assert state['epoch'] == 1 and state['consumed'] == 4
    monkeypatch.setattr(runtime, 'train_update', update)
    runtime.train(tiny, interrupted)
    a = torch.load(complete / 'last.pt', weights_only=True)
    b = torch.load(interrupted / 'last.pt', weights_only=True)
    assert torch.equal(a['model']['bias'], b['model']['bias'])
    for key in ('exp_avg', 'exp_avg_sq', 'step'):
        assert torch.equal(a['optimizer']['state'][0][key], b['optimizer']['state'][0][key])


def test_formal_evaluation_can_load_best_from_earlier_epoch(tiny, tmp_path, monkeypatch):
    from experiments.paper1.lst_restoration_epochval_20261014 import evaluate as evaluation
    runtime.train(tiny, tmp_path)
    first = tmp_path / 'epoch_validation/epoch_01.json'
    row = json.loads(first.read_text())
    row['cells'][1]['raw']['pixel_weighted_rmse'] = 0.
    cfg.write_json(first, row)
    ev.reconcile_selection(tiny, tmp_path, 5)
    expected = torch.load(tmp_path / 'epochs/epoch_01.pt', weights_only=True)['model']['bias']
    seen = []

    def fake_cell(model, config, device, split, profile, output, stop_root, weight_hash):
        seen.append(model.bias.detach().clone())
        return {'split': split, 'condition': evaluation.condition_id(profile), 'unique_records': 3,
                'row_count': 6, 'raw': {}, 'projected': {}, 'observation_mean_fill_baseline': {}}

    monkeypatch.setattr(evaluation, 'execution_guard', lambda config: None)
    monkeypatch.setattr(evaluation, 'COUNTS', cfg.COUNTS)
    monkeypatch.setattr(evaluation, 'evaluate_cell', fake_cell)
    cells = evaluation.evaluate(tiny, tmp_path / 'best.pt', tmp_path / 'best_validation_selected')
    assert len(cells) == len(tiny['evaluation']['profiles'])
    assert all(torch.equal(value, expected) for value in seen)


def test_missing_earlier_validation_or_changed_checkpoint_fails(tiny, tmp_path):
    runtime.train(tiny, tmp_path)
    receipt = tmp_path / 'epoch_validation/epoch_02.json'
    backup = receipt.read_bytes()
    receipt.unlink()
    with pytest.raises(RuntimeError, match='Missing completed validation'):
        ev.reconcile_selection(tiny, tmp_path, 5)
    receipt.write_bytes(backup)
    (tmp_path / 'epochs/epoch_03.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='provenance'):
        ev.reconcile_selection(tiny, tmp_path, 5)


def test_tie_selects_earliest_epoch(tiny, tmp_path):
    runtime.train(tiny, tmp_path)
    for path in (tmp_path / 'epoch_validation').glob('*.json'):
        row = json.loads(path.read_text())
        row['cells'][1]['raw']['pixel_weighted_rmse'] = .5
        cfg.write_json(path, row)
    selection = ev.reconcile_selection(tiny, tmp_path, 5)
    assert selection['best']['epoch'] == 1


def test_generated_39_configs_are_disabled_new_ids_and_val_only():
    root = Path(__file__).resolve().parents[1]
    directory = root / 'runs/paper1/lst_restoration_epochval_20261014/configs'
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert len(manifest['models']) == 39
    ids = set()
    for row in manifest['models']:
        config = json.loads((directory / row['config']).read_text())
        assert cfg.validate(config)['ok']
        assert config['execution']['enabled'] is False
        assert config['tracking']['enabled'] is False
        ids.add(config['tracking']['run_id'])
    assert len(ids) == 39


@pytest.mark.parametrize('mutation', ['test', 'one_view', 'other_metric', 'mixed_training'])
def test_selection_protocol_rejects_unregistered_changes(mutation):
    config = cfg.default_config()
    if mutation == 'test':
        config['epoch_validation']['split'] = 'test'
    elif mutation == 'one_view':
        config['epoch_validation']['views'] = 1
    elif mutation == 'other_metric':
        config['epoch_validation']['selection_metric'] = 'train/loss'
    else:
        config['corruption']['train_profiles'].append({'kind': 'clean', 'seed': 20260925})
    assert not cfg.validate(config)['ok']
