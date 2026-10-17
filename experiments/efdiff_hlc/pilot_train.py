"""Exact-epoch single-GPU EFDiff training on the fixed HLC valid-month membership."""
import copy
import hashlib
import json
import math
import random
import sys
import time
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pilot_data import HLCPairs, COUNTS, PROTOCOL, HERE, UPSTREAM, check_stats, EpochSampler
from model.factory import define_G, init_weights


class MaskedLoss(torch.nn.Module):
    """Equal record weight; invalid target pixels contribute zero."""
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.mask = None

    def forward(self, prediction, target):
        error = (prediction.float()-target.float()).abs()
        if self.kind == 'l2':
            error = error.square()
        valid = self.mask.float()
        count = valid.flatten(1).sum(1)
        if (count == 0).any():
            raise ValueError('Empty target mask; never silently drop a training record')
        return ((error * valid).flatten(1).sum(1) / count).mean()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def config(variant):
    assert variant in ('x0','epsilon')
    return yaml.safe_load((HERE/'configs'/f'pilot_{variant}.yaml').read_text())


def build_model(variant):
    opt = config(variant)
    # Same orthogonal initializer, performed on GPU to avoid lengthy CPU QR.
    opt['phase'] = 'test'
    net = define_G(opt).cuda()
    opt['phase'] = 'train'
    init_weights(net.denoise_fn, init_type='orthogonal')
    net.set_new_noise_schedule(opt['model']['beta_schedule']['train'], 'cuda')
    net.loss_func = MaskedLoss(net.loss_type)
    assert not any(p.requires_grad for p in net.prithvi_encoder.parameters())
    return net, opt


def move(batch):
    return {k: v.cuda(non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}


def apply_update(net, optimizer, records):
    params = [p for p in net.denoise_fn.parameters() if p.grad is not None]
    for p in params:
        p.grad.div_(records)
    grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad)


@torch.no_grad()
def evaluate(net, ds, stats, micro, workers, limit=None, indices=None, seed=20260915, capture_path=None):
    indices = range(len(ds)) if indices is None else indices
    if limit:
        # Fixed spread across the independent validation split; smoke only.
        indices = np.linspace(0, len(ds)-1, limit, dtype=int).tolist()
    loader = DataLoader(ds, sampler=indices, batch_size=micro, num_workers=workers)
    net.eval()
    squared = absolute = baseline = pixels = records = 0
    captures = {key:[] for key in ('target_c','condition_c','mask','prediction_c','index','record_id','month')}
    capture_ids = set(int(ds.members[i]) for i in np.linspace(0,len(ds)-1,16,dtype=int)) if capture_path else set()
    # Restore training RNG afterwards; evaluation frequency cannot change train noise.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for raw in loader:
            b = move(raw)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                residual = net.super_resolution(b, verbose=False,
                    ddim_steps=50 if net.loss_type == 'l1' else None)
            prediction = b['condition_c'] + residual.float() * (3*stats['residual']['std']) + stats['residual']['mean']
            mask = b['mask']
            for i, source_index in enumerate(raw['index'].tolist()):
                if source_index in capture_ids:
                    for key in captures:
                        value = prediction[i] if key=='prediction_c' else raw[key][i]
                        captures[key].append(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
            error = (prediction - b['target_c'])[mask].double()
            squared += error.square().sum().item()
            absolute += error.abs().sum().item()
            baseline += ((b['condition_c']-b['target_c'])[mask].double().square().sum().item())
            pixels += error.numel()
            records += len(raw['index'])
    if capture_path:
        assert len(captures['index']) == 16
        np.savez_compressed(capture_path, **{k:np.asarray(v) for k,v in captures.items()})
    net.train()
    return {'records': records, 'valid_pixels': pixels, 'rmse_c': math.sqrt(squared/pixels),
            'mae_c': absolute/pixels, 'bicubic_rmse_c': math.sqrt(baseline/pixels),
            'squared_error_sum':squared,'absolute_error_sum':absolute,'baseline_squared_error_sum':baseline}


def run(root, output, stats, variant='x0', smoke=False, commit=lambda: None, stop_after_updates=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    opt = config(variant)
    micro, workers = opt['train']['micro_batch'], opt['train']['workers']
    effective = opt['train']['effective_batch']
    assert effective % micro == 0
    assert opt['data']['crop'] == 224 and opt['data']['scale'] == 32
    assert opt['data']['real_modis_input'] is False and opt['normalization']['clip_sigma'] == 3
    assert opt['train']['clip_grad_norm'] == 1.0 and opt['train']['lr_min'] == 1e-6
    torch.set_num_threads(8)
    random.seed(20260915)
    np.random.seed(20260915)
    torch.manual_seed(20260915)
    torch.cuda.manual_seed_all(20260915)
    torch.backends.cudnn.benchmark = True
    ds = HLCPairs(root, stats=stats, augment=True)
    val = HLCPairs(root, split='val', stats=stats)
    if not smoke:
        check_stats(stats, ds)
    budget = json.loads((Path(root)/'experiments/paper1/training_budget.json').read_text())
    assert budget['epochs'] == opt['train']['epochs'] == 5
    assert len(ds) == opt['data']['train_samples']
    stats_hash = hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()
    sources = [HERE/'data.py', HERE/'pilot_data.py', HERE/'pilot_fit_stats.py', HERE/'pilot_train.py', HERE/'pilot_membership/summary.json', HERE/'pilot_membership/train.npy', HERE/'pilot_membership/val.npy', HERE/'configs'/f'pilot_{variant}.yaml'] + sorted((UPSTREAM/'model').glob('*.py')) + sorted((UPSTREAM/'util').glob('*.py'))
    code_hashes = {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    net, opt = build_model(variant)
    net.train()
    optimizer = torch.optim.Adam(net.denoise_fn.parameters(), lr=opt['train']['optimizer']['lr'])
    ema = copy.deepcopy(net.denoise_fn).eval().requires_grad_(False)
    epochs = opt['smoke']['epochs'] if smoke else opt['train']['epochs']
    per_epoch = opt['smoke']['records'] if smoke else len(ds)  # deliberate 3-record tail for smoke
    total_steps = epochs * math.ceil(per_epoch/effective)
    if not smoke:
        assert total_steps == opt['train']['total_updates']
        assert epochs*per_epoch == opt['train']['record_presentations']
    # Pilot-only shared short warmup and EMA, recorded in YAML.
    warmup = opt['train']['warmup_steps']
    base_lr = opt['train']['optimizer']['lr']
    def lr_at(step):
        if step < warmup:
            return base_lr * max(step, 1)/warmup
        progress = (step-warmup)/max(total_steps-warmup, 1)
        return base_lr * max(.5*(1+math.cos(math.pi*progress)), 1e-6/base_lr)
    provenance = {'protocol': PROTOCOL, 'variant': variant, 'smoke': smoke,
        'epochs': epochs, 'records_per_epoch': per_epoch, 'effective_batch': effective, 'micro_batch': micro,
        'updates': total_steps, 'stats_sha256': stats_hash, 'data_fingerprint': ds.fingerprint,
        'upstream_commit': '773f67375991432d790c1e99076597524dbe0f95',
        'model_config': opt, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'code_sha256': code_hashes,
        'gpu': torch.cuda.get_device_name(), 'normalization': stats,
        'selection': 'PILOT fixed epoch5 EMA; all fixed 512 val months each epoch; no test evaluation'}
    save_json(output/'provenance.json', provenance)
    snapshot = output/'source'
    snapshot.mkdir(exist_ok=True)
    for p in sources:
        dst = snapshot/p.relative_to(HERE.parent)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    (output/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
    checkpoint = output/'latest.pt'
    epoch_start = offset = step = presented = 0
    best = math.inf
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['stats_sha256'] == stats_hash and state['data_fingerprint'] == ds.fingerprint
        assert state['variant'] == variant and state['smoke'] == smoke
        assert state['code_sha256'] == code_hashes, 'Runner changed; start a versioned run instead'
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        ema.load_state_dict(state['ema'])
        epoch_start, offset, step, presented, best = (state[k] for k in ('epoch', 'offset', 'step', 'presented', 'best'))
        torch.set_rng_state(state['cpu_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        del state
    def save(epoch, offset):
        state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'ema': ema.state_dict(),
                 'epoch': epoch, 'offset': offset, 'step': step, 'presented': presented, 'best': best,
                 'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                 'stats_sha256': stats_hash, 'data_fingerprint': ds.fingerprint,
                 'code_sha256': code_hashes,
                 'variant': variant, 'smoke': smoke}
        temporary = output/'latest.tmp'
        torch.save(state, temporary)
        temporary.replace(checkpoint)
        commit()
    started = time.time()
    for epoch in range(epoch_start, epochs):
        sampler = EpochSampler(ds, epoch, start=offset, limit=per_epoch)
        # Separate loader generator prevents worker setup from perturbing diffusion RNG.
        loader = DataLoader(ds, sampler=sampler, batch_size=micro, num_workers=workers,
            pin_memory=True, generator=torch.Generator().manual_seed(20260915+epoch))
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        update_loss = 0.0
        for raw in loader:
            b = move(raw)
            n = len(raw['index'])
            net.loss_func.mask = b['mask']
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = net(b)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            (loss*n).backward()
            pending += n
            update_loss += float(loss.detach())*n
            offset += n
            presented += n
            if pending == effective or offset == per_epoch:
                for group in optimizer.param_groups:
                    group['lr'] = lr_at(step)
                grad = apply_update(net, optimizer, pending)
                step += 1
                with torch.no_grad():
                    decay = opt['train']['ema_decay'] if step >= opt['train']['ema_start_step'] else 0.0
                    for e, p in zip(ema.parameters(), net.denoise_fn.parameters()):
                        e.lerp_(p, 1-decay)
                record = {'epoch': epoch+1, 'epoch_records': offset, 'step': step, 'presented': presented,
                          'update_records': pending, 'loss': update_loss/pending, 'grad_norm': grad,
                          'elapsed_seconds': time.time()-started, 'peak_gpu_gb': torch.cuda.max_memory_allocated()/1e9}
                if step % opt['train']['log_every'] == 0 or smoke or offset == per_epoch:
                    print(json.dumps(record), flush=True)
                    with (output/'metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(record)+'\n')
                    save_json(output/'status.json', dict(record, phase='training'))
                pending = 0
                update_loss = 0.0
                if step % opt['train']['checkpoint_every'] == 0:
                    save(epoch, offset)
                if (stop_after_updates is not None and step >= stop_after_updates) or (not smoke and time.time()-started > opt['train']['segment_seconds']):
                    save(epoch, offset)
                    return {'phase': 'checkpointed', 'updates': step, 'record_presentations': presented}
        assert offset == per_epoch and pending == 0
        # Persist before expensive full validation, so interruption loses no epoch.
        save(epoch, offset)
        live = net.denoise_fn
        net.denoise_fn = ema
        metrics = evaluate(net, val, stats, micro, workers,
            limit=opt['smoke']['validation_records'] if smoke else opt['evaluation']['monitor_samples'],
            capture_path=output/'predictions_epoch5.npz' if not smoke and epoch+1==epochs else None)
        net.denoise_fn = live
        metrics.update(epoch=epoch+1, step=step, split='val', smoke=smoke)
        save_json(output/f'validation_epoch_{epoch+1}.json', metrics)
        if metrics['rmse_c'] < best:
            best = metrics['rmse_c']
            # Encoder weights are in latest.pt; selected denoiser + epoch is enough.
            torch.save({'ema': ema.state_dict(), 'epoch': epoch+1, 'metrics': metrics,
                        'stats': stats, 'variant': variant}, output/'best_ema.pt')
        offset = 0
        save(epoch+1, 0)
    assert step == total_steps and presented == epochs*per_epoch
    result = {'phase': 'smoke_passed' if smoke else 'training_complete', 'updates': step,
              'record_presentations': presented, 'epochs': epochs, 'best_val_rmse_c': best,
              'elapsed_seconds': time.time()-started, 'test_used': False}
    save_json(output/'status.json', result)
    commit()
    return result

