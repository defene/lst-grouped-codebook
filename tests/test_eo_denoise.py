import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from eo_data import prepare
from eo_denoise.config import DEFAULT, validate
from eo_denoise.data import PairedDataset, EpochSampler
from eo_denoise.engine import train, read_checkpoint
from eo_denoise.losses import masked_mean, gradient_loss, stability_loss, structure_loss
from eo_denoise.metrics import EvaluationMetrics
from eo_denoise.models import Codec, VectorQuantizer, FiniteScalarQuantizer
from eo_denoise.noise import Corruptor, MaskBank
from eo_denoise.mask_bank import build_bank
from eo_denoise.tokens import pack, unpack
from test_eo_data import sources

torch.set_num_threads(2)


def small_config():
    c = copy.deepcopy(DEFAULT)
    c['model'].update(base_width=4, blocks=1, latent_dim=4, codebook_size=8,
                      vit_dim=48, vit_heads=4, vit_depth=1, grid=8,
                      eo_dim=48, eo_depth=1, eo_heads=4,
                      eo_decoder_dim=32, eo_decoder_depth=1, eo_decoder_heads=4)
    c['noise'].update(kind='procedural_missing', clean_probability=0)
    c['train'].update(steps=4, batch_size=1, accumulation=1, workers=0,
                      warmup=1, eval_every=4, save_every=4, log_every=4, device='cpu', amp=False)
    c['eval'].update(batch_size=1, bootstrap_replicates=0)
    return c


@pytest.fixture
def paired_root(sources):
    for path in (sources/'prithvi_data').rglob('metadata.json'):
        meta = json.loads(path.read_text())
        meta.update(datetime='2020-01-01T10:00:00Z', hls_image_id=meta['tile']+'_20200101T100000')
        path.write_text(json.dumps(meta))
        f = np.zeros((256, 256), np.uint8)
        f[32:64, 32:96] = 2
        f[100:120, 100:160] = 8
        f[180:230, 10:20] = 4
        f[0, 0] = 255
        np.save(path.parent/'fmask.npy', f)
    prepare.prepare_dataset(sources, 'prithvi', sources/'processed_data/v1', workers=1)
    return sources


def test_noise_is_deterministic_and_cannot_reveal_hidden_values():
    c = small_config()['noise']
    x = np.ones((6, 256, 256), np.float32)
    mask = np.ones((1, 256, 256), bool)
    mask[:, :8] = False
    corrupt = Corruptor(c, 'train')
    a = corrupt(x, mask, 'example', 'hls', 0, 0)
    b = corrupt(x*7, mask, 'example', 'hls', 0, 0)
    assert np.array_equal(a['corruption_mask'], b['corruption_mask'])
    assert np.all(a['image'][:, a['corruption_mask'][0]] == 0)
    assert np.all(b['image'][:, b['corruption_mask'][0]] == 0)
    assert not a['corruption_mask'][:, :8].any()
    assert not np.array_equal(a['corruption_mask'], corrupt(x, mask, 'example', 'hls', 1, 0)['corruption_mask'])
    val = Corruptor(c, 'val')
    assert np.array_equal(val(x, mask, 'example', 'hls', 0, 0, .1)['image'],
                          val(x, mask, 'example', 'hls', 20, 0, .1)['image'])
    gaussian = Corruptor(dict(c, kind='gaussian'), 'train')(x, mask, 'e', 'hls', 0, 0)
    assert np.array_equal(gaussian['input_mask'], mask)
    assert np.all(gaussian['image'][:, :8] == 0)


def test_reference_masks_control_losses_and_structure_gate():
    p = torch.tensor([[[[1., 1000.], [1., 1000.]]]], requires_grad=True)
    t = torch.zeros_like(p)
    m = torch.tensor([[[[True, False], [True, False]]]])
    loss = masked_mean((p-t).square(), m)
    assert loss.item() == 1
    assert gradient_loss(p, t, m).item() == 0
    loss.backward()
    assert p.grad[0, 0, 0, 1] == 0
    empty = masked_mean(p.square(), torch.zeros_like(m))
    assert empty.item() == 0
    feature = torch.randn(2, 8, 4, 4, requires_grad=True)
    a, b = {'structure': feature}, {'structure': feature*3}
    mask = torch.ones(2, 1, 16, 16, dtype=torch.bool)
    assert structure_loss(a, b, mask, mask, torch.ones(2, dtype=torch.bool), 4, .75) < 1e-10
    different = {'structure': torch.randn_like(feature)}
    assert structure_loss(a, different, mask, mask, torch.zeros(2, dtype=torch.bool), 4, .75) == 0
    assert structure_loss(a, different, mask, mask, torch.ones(2, dtype=torch.bool), 4, .75) > .01


def test_vq_fsq_gradients_and_code_roundtrip():
    for q in (VectorQuantizer(4, 16), FiniteScalarQuantizer(4, [8, 8, 8])):
        z = torch.randn(2, 4, 8, 8, requires_grad=True)
        output = q(z)
        decoded = q.from_indices(output['indices'])
        torch.testing.assert_close(decoded, output['quantized'], atol=2e-6, rtol=0)
        (output['quantized'].square().mean()+output['bottleneck_loss']).backward()
        assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
        if isinstance(q, VectorQuantizer):
            assert q.embedding.weight.grad.abs().sum() > 0
            common = torch.ones(2, 1, 256, 256, dtype=torch.bool)
            assert stability_loss(output, output, common, .75).abs() < 1e-7
    q = FiniteScalarQuantizer(3, [8, 8, 8])
    all_ids = torch.arange(512).reshape(1, 16, 32)
    assert torch.isfinite(q.from_indices(all_ids)).all()


@pytest.mark.parametrize('backbone,kind', [('conv','ae'),('conv','vae'),('conv','vq'),('conv','fsq'),('prithvi_style','vae'),('prithvi_eo2_spatial','vq')])
def test_backbones_have_no_mask_leak_and_can_backpropagate(backbone, kind):
    c = small_config()['model']
    c.update(backbone=backbone, quantizer=kind)
    codec = Codec(1, c)
    x = torch.randn(1, 1, 256, 256)
    mask = torch.ones(1, 1, 256, 256, dtype=torch.bool)
    mask[:, :, 64:128, 64:128] = False
    codec.eval()
    a = codec(x, mask)
    changed = x.clone()
    changed[~mask] = 1e6
    b = codec(changed, mask)
    torch.testing.assert_close(a['reconstruction'], b['reconstruction'], atol=0, rtol=0)
    assert a['reconstruction'].shape == x.shape
    codec.train()
    a = codec(x, mask)
    (a['reconstruction'].square().mean()+a['bottleneck_loss']+1e-3*a['kl']).backward()
    gradients = [p.grad for p in codec.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    if kind in ('vq', 'fsq'):
        codec.eval()
        out = codec(x, mask)
        torch.testing.assert_close(codec.decode_indices(out['indices']), out['reconstruction'], atol=2e-5, rtol=0)


def test_token_file_bitwidth_mask_and_checksum():
    ids = np.arange(64).reshape(8,8) % 10
    mask = np.random.default_rng(1).random((1, 256, 256)) > .1
    blob = pack(ids, 10, mask, {'modality':'lst'})
    restored, recovered_mask, header = unpack(blob)
    assert header['token_bytes'] == 32
    np.testing.assert_array_equal(restored, ids)
    np.testing.assert_array_equal(recovered_mask, mask)
    corrupt = bytearray(blob); corrupt[-1] ^= 1
    with pytest.raises(ValueError, match='Corrupt'):
        unpack(corrupt)


def test_physical_metrics_use_sample_equal_weight_and_empty_regions():
    source = {'lst': SimpleNamespace(stats={'std':[100.], 'mean':[2000.]})}
    model = SimpleNamespace(codecs={'lst': SimpleNamespace(config={'quantizer':'ae'})})
    metrics = EvaluationMetrics(source, model)
    target = torch.zeros(2, 1, 2, 2)
    pred = torch.ones_like(target); pred[1] *= 3
    mask = torch.ones_like(target, dtype=torch.bool); mask[0] = False; mask[0, 0, 0, 0] = True
    data = {'target':target,'reference_mask':mask,'corruption_mask':torch.zeros(2,2,1,2,2,dtype=torch.bool),
            'input_mask':mask[:,None].expand(-1,2,-1,-1,-1), 'actual_coverage':torch.zeros(2,2)}
    metrics.add('lst',[{'reconstruction':pred},{'reconstruction':pred}],data,['a','b'],['tile_a','tile_b'])
    result = metrics.summary(50)
    assert result['regions']['lst/all']['rmse'] == pytest.approx(np.sqrt(5))
    assert result['regions']['lst/all']['mae'] == 2
    assert 'lst/corrupt' not in result['regions']
    assert result['regions']['lst/all']['rmse_tile_ci95'] is not None


def test_pair_alignment_bank_split_and_persistent_worker_epochs(paired_root):
    c = small_config(); c['data']['root'] = str(paired_root)
    bank_root = paired_root/'banks'
    for split in ('train', 'val'):
        build_bank(paired_root, bank_root, split, max_records=3, max_shapes=6)
    c['noise'].update(kind='bank_missing', bank_root=str(bank_root))
    ds = PairedDataset(c, 'train')
    assert len(ds) == 2 and ds[0]['cross_view_eligible']
    for m in ds.modalities:
        assert not np.shares_memory(ds[0][m]['image'], ds[0][m]['target'])
    bank = ds.corruptor.bank
    with pytest.raises(ValueError, match='split'):
        MaskBank(bank_root, 'train', bank.info['source_index_sha256'], 'wrong-val-sample-hash')
    sampler = EpochSampler(len(ds), 17)
    loader = DataLoader(ds, batch_size=2, sampler=sampler, num_workers=2, persistent_workers=True)
    a = next(iter(loader))
    positions = list(iter(EpochSampler(len(ds), 17)))
    expected = [ds[p] for p in positions]
    assert a['sample_id'] == [s['sample_id'] for s in expected]
    np.testing.assert_array_equal(a['hls']['image'].numpy(), np.stack([s['hls']['image'] for s in expected]))
    sampler.epoch = 1
    b = next(iter(loader))
    assert not torch.equal(a['hls']['image'], b['hls']['image'])
    del loader
    ds.close()


def test_optimizer_resume_reproduces_uninterrupted_run(paired_root, tmp_path):
    c = small_config(); c['data'].update(root=str(paired_root),modalities=['lst'])
    c['model']['quantizer'] = 'vae'
    c['loss'].update(stability=0,cross_view=0)
    uninterrupted = tmp_path/'continuous_run'
    interrupted = tmp_path/'resume_run'
    train(c, uninterrupted)
    partial = train(c, interrupted, stop_after=2)
    assert partial['status'] == 'stopped_at_checkpoint'
    resumed = train(c, interrupted, resume=interrupted/'last.pt')
    assert resumed['status'] == 'complete'
    a, b = read_checkpoint(uninterrupted/'last.pt'), read_checkpoint(interrupted/'last.pt')
    for name in a['model']:
        torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
    assert (a['epoch'],a['consumed']) == (b['epoch'],b['consumed'])
    bad = copy.deepcopy(c);bad['loss']['gradient'] = .2
    with pytest.raises(ValueError, match='config differs'):
        read_checkpoint(interrupted/'last.pt', bad)
