"""Mechanism, pretrained identity and disposable real-training smoke checks."""
import copy
import hashlib
import itertools
import time

import torch

from .config import write_json
from .runtime import source_hashes, protocol_digest, setup, amp, move_batch, train_update, memory, utc
from .stage2_models import drifting_energy, DriftingVectorQuantizer


def energy_checks(reconstruction_gradient='codes_only'):
    h = torch.tensor([[.1, .2], [-.2, .4], [.3, -.1]], requires_grad=True)
    c = torch.tensor([[-.4, -.3], [.5, .2]], requires_grad=True)
    tau = .2
    actual, _ = drifting_energy(h, c, tau, False, False)
    expected = h.new_zeros(())
    particles = list(h.unbind()) + list(c.unbind())
    charges = [1 / len(h)] * len(h) + [-1 / len(c)] * len(c)
    for i in range(len(particles)):
        for j in range(i):
            r = torch.linalg.vector_norm(particles[i] - particles[j]) + 1e-8
            expected = expected + torch.exp(-r / tau) * tau * (r + tau) * charges[i] * charges[j]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)
    a = torch.autograd.grad(actual, (h, c), retain_graph=True)
    b = torch.autograd.grad(expected, (h, c))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=2e-5, atol=1e-8)
    r = torch.tensor(.3, requires_grad=True)
    potential = torch.exp(-r / tau) * tau * (r + tau)
    derivative = torch.autograd.grad(potential, r)[0]
    assert derivative < 0  # minimizing like-charge energy increases distance
    q = DriftingVectorQuantizer(dict(latent_dim=4, codebook_size=16,
                                    drift_tau=.2, drift_beta=.25, drift_tokens=8,
                                    drift_reconstruction_gradient=reconstruction_gradient))
    x = torch.randn(2, 4, 4, 4, requires_grad=True)
    rng = torch.get_rng_state()
    out = q(x)
    torch.set_rng_state(rng)
    again = q(x)
    assert torch.equal(out['loss'], again['loss'])
    assert torch.equal(out['quantized'], q.decode_indices(out['indices']))
    gx, gc = torch.autograd.grad(out['loss'], (x, q.embedding.weight), retain_graph=True)
    assert gx.isfinite().all() and gx.abs().sum() > 0
    assert gc.isfinite().all() and (gc.abs().sum(1) > 0).all()
    rec_gx, rec_gc = torch.autograd.grad(out['quantized'].square().mean(), (x, q.embedding.weight), allow_unused=True)
    if reconstruction_gradient == 'codes_only':
        assert rec_gx is None
    else:
        assert rec_gx is not None and rec_gx.isfinite().all() and rec_gx.abs().sum() > 0
    assert rec_gc is not None and rec_gc.abs().sum() > 0
    return dict(pair_formula_and_gradients_match=True, attraction_repulsion_signs=True,
                stochastic_token_sample_rng_replay=True, encoder_energy_gradient=True,
                all_codewords_receive_energy_gradients=True,
                reconstruction_gradient=reconstruction_gradient)


def run(config, path):
    from torch.utils.data import DataLoader
    from .models import build_model
    from .data import LSTDataset, EpochSampler, update_batches
    from .objective import model_inputs
    from .representations import Representation
    report = dict(started_utc=utc(), protocol_config_sha256=protocol_digest(config),
                  source_hashes=source_hashes(), gates={'stage2_mechanism': False}, status='running')
    dataset = None
    try:
        report['energy_checks'] = energy_checks(config['model'].get('drift_reconstruction_gradient', 'codes_only'))
        device = setup(config)
        model = build_model(config).to(device)
        rep = Representation(config['representation'])
        dataset = LSTDataset(config, 'train')
        sampler = list(itertools.islice(EpochSampler(len(dataset), config['train']['sample_order_seed'], 0), 256))
        loader = DataLoader(dataset, sampler=sampler, batch_size=config['train']['micro_batch'],
                            num_workers=0, drop_last=False,
                            generator=torch.Generator().manual_seed(20260925))
        batch = move_batch(next(iter(loader)), device)
        inputs = model_inputs(batch, rep)
        model.eval()
        with torch.no_grad():
            # Check the same BF16 batch used by the formal codec evaluator.
            with amp(config, device):
                out = model(**inputs)
                replay = model.decode_indices(out['indices'], batch['mean_celsius'])
            assert torch.equal(out['latent'], model.quantizer.decode_indices(out['indices']))
            assert torch.equal(replay, out['encoded_prediction'])
            report['same_batch_bf16_exact_codec'] = True
            if config['model']['kind'] == 'vqgan_pretrained':
                small = {k: v[:1] if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                x = small['encoded_input']
                adapted = model.input_adapter(x, small['reference_valid'], small['observed_valid'], small['mean_celsius'])
                assert torch.equal(adapted, x)
                rgb = x.repeat(1, 3, 1, 1) if model.channels == 1 else x
                latent = model.quant_conv(model.encoder(rgb))
                native_codes = model.quantizer(latent)
                reference = model.decoder(model.post_quant_conv(native_codes['quantized'])).float()
                if model.channels == 1:
                    reference = reference.mean(1, keepdim=True)
                adapted_out = model(**small)
                torch.testing.assert_close(reference, adapted_out['encoded_prediction'], rtol=1e-6, atol=1e-6)
                report['pretrained_identity'] = dict(passed=True, tensors_loaded=model.loaded_tensor_count,
                    max_output_difference=float((reference-adapted_out['encoded_prediction']).abs().max()),
                    reference='direct unmodified official encoder and decoder modules',
                    input_adapter_identity=True, all_parameters_trainable=all(p.requires_grad for p in model.parameters()))
            del out, replay
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'], betas=(.9,.95),
                                       weight_decay=config['train']['weight_decay'])
        model.train()
        report['updates'] = []
        for index, group in enumerate(update_batches(iter(loader), config['train']['effective_batch'] // config['train']['micro_batch'])):
            start = time.perf_counter()
            actual, losses, profiles = train_update(model, group, optimizer, rep, config, device)
            norms = {}
            for prefix in ('encoder', 'decoder', 'quantizer', 'input_adapter', 'decoder_mean'):
                gradients = [p.grad for n,p in model.named_parameters() if n.startswith(prefix+'.') and p.grad is not None]
                assert gradients and all(g.isfinite().all() for g in gradients)
                norms[prefix] = float(sum(g.float().square().sum() for g in gradients).sqrt())
            if index >= 1:
                assert all(v > 0 for v in norms.values()), norms
            report['updates'].append(dict(update=index+1, records=actual, losses=losses,
                                           gradient_norms_after_clip=norms, seconds=time.perf_counter()-start))
        assert len(report['updates']) == 4 and sum(x['records'] for x in report['updates']) == 256
        report['real_train_smoke'] = dict(records=256, updates=4, weights_discarded=True,
            excluded_from_formal_budget=True, data_fingerprint=dataset.fingerprint,
            sampler_sha256=hashlib.sha256(repr(sampler).encode()).hexdigest(), validation_or_test_used=False)
        report['memory'] = memory(device)
        report['status'] = 'passed'
        report['gates']['stage2_mechanism'] = True
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        if dataset is not None:
            dataset.close()
        report['finished_utc'] = utc()
        if report['source_hashes'] != source_hashes():
            report['status'] = 'failed'
            report['gates']['stage2_mechanism'] = False
        write_json(path, report)
