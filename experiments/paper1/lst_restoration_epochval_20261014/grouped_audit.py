"""Mechanism, pretrained identity and disposable real-training smoke checks."""
import copy
import hashlib
import itertools
import time

import torch

from .config import write_json
from .runtime import source_hashes, protocol_digest, setup, amp, move_batch, train_update, memory, utc
from .grouped_adapter import mechanism_checks



def run(config, path):
    from torch.utils.data import DataLoader
    from .models import build_model
    from .data import LSTDataset, EpochSampler, update_batches
    from .objective import model_inputs
    from .representations import Representation
    report = dict(started_utc=utc(), protocol_config_sha256=protocol_digest(config),
                  source_hashes=source_hashes(), gates={'grouped_mechanism': False}, status='running')
    dataset = None
    try:
        report['grouped_mechanism_checks'] = mechanism_checks()
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
        report['gates']['grouped_mechanism'] = True
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        if dataset is not None:
            dataset.close()
        report['finished_utc'] = utc()
        if report['source_hashes'] != source_hashes():
            report['status'] = 'failed'
            report['gates']['grouped_mechanism'] = False
        write_json(path, report)
