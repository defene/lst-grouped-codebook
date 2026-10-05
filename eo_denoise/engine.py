from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
import copy
import json
import math
import platform
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from eo_data.core import sha256_file, json_write
from eo_data.locking import build_lock as output_lock
from .config import fingerprint
from .data import PairedDataset, EpochSampler
from .losses import objective
from .metrics import EvaluationMetrics
from .models import CrossViewModel
from .noise import stable_seed


def device_for(config):
    requested = config['train']['device']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, config['train']['memory_limit_gib']*2**30/total), device)
    return device


def memory_stats(device):
    if device.type != 'cuda':
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return {name: value/2**30 for name, value in {
        'allocated_gib': torch.cuda.memory_allocated(device),
        'reserved_gib': torch.cuda.memory_reserved(device),
        'peak_allocated_gib': torch.cuda.max_memory_allocated(device),
        'peak_reserved_gib': torch.cuda.max_memory_reserved(device),
        'device_free_gib': free, 'device_total_gib': total}.items()}


def source_hashes():
    folder = Path(__file__).parent
    return {p.name: sha256_file(p) for p in sorted(folder.glob('*.py'))}


def environment(device):
    import importlib.metadata
    return {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
            'timm': importlib.metadata.version('timm') if importlib.util.find_spec('timm') else None,
            'cuda_runtime': torch.version.cuda, 'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'cudnn_benchmark': False, 'deterministic_algorithms': True,
            'bf16_supported': torch.cuda.is_bf16_supported() if device.type == 'cuda' else False}


def seed_everything(seed):
    # Must be set before initializing CUDA handles for deterministic matmuls.
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def move_batch(batch, device, modalities):
    moved = {k: v for k, v in batch.items() if k not in modalities}
    moved['cross_view_eligible'] = batch['cross_view_eligible'].to(device)
    for m in modalities:
        moved[m] = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch[m].items()}
    return moved


def inputs_only(batch, modalities):
    return {m: [{'image': batch[m]['image'][:, i], 'input_mask': batch[m]['input_mask'][:, i]}
                for i in range(batch[m]['image'].shape[1])] for m in modalities}


def amp_context(config, device):
    enabled = config['train']['amp'] and device.type == 'cuda' and torch.cuda.is_bf16_supported()
    return torch.autocast('cuda', dtype=torch.bfloat16) if enabled else nullcontext()


def atomic_checkpoint(path, payload):
    path = Path(path)
    temp = path.with_suffix('.partial')
    torch.save(payload, temp)
    temp.replace(path)


def read_checkpoint(path, config=None, data_fingerprint=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['source_hashes'] != source_hashes():
        raise ValueError('Framework code changed since this checkpoint; use the saved source snapshot')
    if config is not None and checkpoint['config_sha256'] != fingerprint(config):
        raise ValueError('Resume config differs; use the exact saved config')
    if data_fingerprint is not None and checkpoint['data_fingerprint'] != data_fingerprint:
        raise ValueError('Data, normalization, pair index, or mask bank changed since checkpoint')
    return checkpoint


@torch.no_grad()
def evaluate_model(model, config, device, coverage=None, output=None):
    c = copy.deepcopy(config)
    coverage = c['eval']['coverage'] if coverage is None else coverage
    ds = PairedDataset(c, 'val', coverage=coverage)
    loader = DataLoader(ds, batch_size=c['eval']['batch_size'], shuffle=False,
                        num_workers=c['train']['workers'], pin_memory=device.type == 'cuda',
                        generator=torch.Generator().manual_seed(20260910))
    metrics = EvaluationMetrics(ds.sources, model)
    was_training = model.training
    model.eval()
    eligible = total = 0
    first_preview = None
    started = time.perf_counter()
    for batch in loader:
        moved = move_batch(batch, device, model.codecs)
        with amp_context(c, device):
            predictions = model(inputs_only(moved, model.codecs))
        for m in model.codecs:
            metrics.add(m, predictions[m], moved[m], batch['sample_id'], batch['tile_id'])
        eligible += int(batch['cross_view_eligible'].sum())
        total += len(batch['sample_id'])
        if first_preview is None:
            first_preview = {}
            for m in model.codecs:
                for field in ('target', 'reference_mask'):
                    first_preview[f'{m}_{field}'] = moved[m][field][0].cpu().numpy()
                for field in ('image', 'input_mask', 'corruption_mask'):
                    first_preview[f'{m}_{field}'] = moved[m][field][0, 0].cpu().numpy()
                first_preview[f'{m}_prediction'] = predictions[m][0]['reconstruction'][0].float().cpu().numpy()
    result = metrics.summary(c['eval']['bootstrap_replicates'])
    result.update(split='val', independent_test_available=False, coverage=coverage,
                  data_fingerprint=ds.fingerprint, noise_kind=c['noise']['kind'],
                  noise_interpretation='controlled missing-region recovery with known input mask' if c['noise']['kind'].endswith('missing') else 'clean or numeric diagnostic',
                  metadata_eligible_pairs=eligible, evaluated_pairs=total,
                  elapsed_seconds=time.perf_counter()-started,
                  rates={m: codec.rate() for m, codec in model.codecs.items()})
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        json_write(output/'metrics.json', result)
        with (output/'samples.jsonl').open('w', encoding='utf-8') as stream:
            for row in metrics.rows:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        np.savez_compressed(output/'preview.npz', **first_preview)
    ds.close()
    model.train(was_training)
    return result


def learning_rate(c, step):
    warmup = c['train']['warmup']
    if step < warmup:
        return c['train']['lr']*(step+1)/max(1, warmup)
    progress = min(1, (step-warmup)/max(1, c['train']['steps']-warmup))
    return c['train']['lr']*(.1+.9*.5*(1+math.cos(math.pi*progress)))


def train(config, output, resume=None, stop_after=None):
    output = Path(output).resolve()
    if resume is None and output.exists():
        raise FileExistsError(f'Use a new run directory or --resume: {output}')
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        try:
            return _train(config, output, resume, stop_after)
        except BaseException as error:
            failure = {'status':'failed', 'error':repr(error),
                       'last_checkpoint':str(output/'last.pt') if (output/'last.pt').exists() else None,
                       'updated_utc':datetime.now(timezone.utc).isoformat()}
            json_write(output/'failure.json', failure)
            json_write(output/'status.json', failure)
            raise


def _train(c, output, resume, stop_after):
    seed_everything(c['seed'])
    device = device_for(c)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ds = PairedDataset(c, 'train')
    model = CrossViewModel(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['train']['lr'], betas=(.9, .95), weight_decay=c['train']['weight_decay'])
    start_step, epoch, consumed, best = 0, 0, 0, float('inf')
    if resume:
        checkpoint = read_checkpoint(resume, c, ds.fingerprint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step, epoch, consumed, best = checkpoint['step'], checkpoint['epoch'], checkpoint['consumed'], checkpoint['best']
        torch.set_rng_state(checkpoint['rng_cpu'])
        if device.type == 'cuda' and checkpoint['rng_cuda']:
            torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
    else:
        json_write(output/'config.json', c)
        json_write(output/'data_fingerprint.json', ds.fingerprint)
        json_write(output/'environment.json', environment(device))
        np.save(output/'train_record_ids.npy', ds.record_ids, allow_pickle=False)
        json_write(output/'model.json', {'parameters': sum(p.numel() for p in model.parameters()),
            'per_modality': {m: sum(p.numel() for p in codec.parameters()) for m, codec in model.codecs.items()},
            'rates': {m: codec.rate() for m, codec in model.codecs.items()},
            'architecture': c['model'], 'pretrained_weights': False})
        import zipfile
        with zipfile.ZipFile(output/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for p in Path(__file__).parent.glob('*.py'):
                archive.write(p, 'eo_denoise/'+p.name)
            for p in Path(__file__).parent.glob('*LICENSE.txt'):
                archive.write(p, 'eo_denoise/'+p.name)
    sampler = EpochSampler(len(ds), c['seed'], epoch, consumed)
    generator = torch.Generator().manual_seed(stable_seed('workers', c['seed']) % (2**63))
    loader = DataLoader(ds, batch_size=c['train']['batch_size'], sampler=sampler,
                        num_workers=c['train']['workers'], persistent_workers=c['train']['workers'] > 0,
                        pin_memory=device.type == 'cuda', generator=generator)
    iterator = iter(loader)
    model.train()
    started = time.perf_counter()
    end_step = min(c['train']['steps'], stop_after) if stop_after is not None else c['train']['steps']
    last_step = start_step
    json_write(output/'status.json', {'status':'running', 'step':last_step,
        'configured_steps':c['train']['steps'], 'memory':memory_stats(device),
        'updated_utc':datetime.now(timezone.utc).isoformat()})
    try:
        for step in range(start_step, end_step):
            batches = []
            for _ in range(c['train']['accumulation']):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch, consumed = epoch+1, 0
                    sampler.epoch, sampler.start = epoch, 0
                    iterator = iter(loader)
                    batch = next(iterator)
                consumed += len(batch['sample_id'])
                batches.append(batch)
            total_samples = sum(len(b['sample_id']) for b in batches)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(c, step)
            optimizer.zero_grad(set_to_none=True)
            logs = {}
            for batch in batches:
                moved = move_batch(batch, device, model.codecs)
                with amp_context(c, device):
                    predictions = model(inputs_only(moved, model.codecs))
                loss, terms = objective(predictions, moved, c)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite objective')
                weight = len(batch['sample_id'])/total_samples
                (loss*weight).backward()
                for key, value in dict(terms, total=loss).items():
                    logs[key] = logs.get(key, 0)+float(value.detach())*weight
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['train']['clip'], error_if_nonfinite=True)
            optimizer.step()
            last_step = step+1
            stop_requested = (output/'STOP').exists()
            if stop_requested:
                end_step = last_step
            logs.update(step=last_step, epoch=epoch, consumed=consumed,
                        lr=optimizer.param_groups[0]['lr'], gradient_norm=float(grad_norm),
                        batch_samples=total_samples, elapsed_seconds=time.perf_counter()-started,
                        sample_ids=[s for batch in batches for s in batch['sample_id']])
            logs['cross_eligible_samples'] = sum(int(b['cross_view_eligible'].sum()) for b in batches)
            logs['coverage'] = {m: float(torch.cat([b[m]['actual_coverage'] for b in batches]).mean()) for m in model.codecs}
            logs['memory'] = memory_stats(device)
            with (output/'train.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(logs, allow_nan=False)+'\n')
            if last_step % c['train']['log_every'] == 0 or last_step == end_step:
                print(json.dumps({k: logs[k] for k in ('step', 'total', 'reconstruction', 'stability', 'cross_view', 'gradient_norm')}) , flush=True)
                json_write(output/'status.json', {'status':'running', 'step':last_step,
                    'configured_steps':c['train']['steps'], 'epoch':epoch, 'consumed':consumed,
                    'elapsed_seconds':logs['elapsed_seconds'], 'memory':logs['memory'],
                    'updated_utc':datetime.now(timezone.utc).isoformat()})
            improved = False
            if not stop_requested and (last_step % c['train']['eval_every'] == 0 or last_step == end_step):
                score = evaluate_model(model, c, device, output=output/f'val_step_{last_step:07d}')
                if score['selection_score'] < best:
                    best, improved = score['selection_score'], True
            if last_step % c['train']['save_every'] == 0 or last_step == end_step or improved:
                state = {'format': 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'step': last_step, 'epoch': epoch, 'consumed': consumed, 'best': best,
                         'config': c, 'config_sha256': fingerprint(c), 'data_fingerprint': ds.fingerprint,
                         'source_hashes': source_hashes(), 'rng_cpu': torch.get_rng_state(),
                         'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
                atomic_checkpoint(output/'last.pt', state)
                if improved:
                    atomic_checkpoint(output/'best.pt', state)
            if stop_requested:
                break
        result = {'status': 'complete' if last_step >= c['train']['steps'] else 'stopped_at_checkpoint',
                  'step': last_step, 'configured_steps': c['train']['steps'], 'best_val_score': best if math.isfinite(best) else None,
                  'last_checkpoint': str(output/'last.pt'), 'device': str(device),
                  'memory':memory_stats(device), 'epoch':epoch, 'consumed':consumed,
                  'purpose': 'framework_smoke' if c['name'].startswith('smoke_') else 'configured_experiment',
                  'finished_utc': datetime.now(timezone.utc).isoformat()}
        json_write(output/'status.json', result)
        return result
    finally:
        ds.close()
        # DataLoader owns worker lifetime; dropping it also ends persistent workers.
        del iterator, loader


def evaluate_checkpoint(checkpoint_path, output, coverage=None, limit=None, workers=None, noise_kind=None):
    checkpoint = read_checkpoint(checkpoint_path)
    config = copy.deepcopy(checkpoint['config'])
    if limit is not None:
        config['data']['val_limit'] = limit
    if workers is not None:
        config['train']['workers'] = workers
    seed_everything(config['seed'])
    device = device_for(config)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = CrossViewModel(config).to(device)
    model.load_state_dict(checkpoint['model'])
    # Validate training identities before evaluating a potentially different val subset.
    original_data = PairedDataset(checkpoint['config'], 'train')
    if original_data.fingerprint != checkpoint['data_fingerprint']:
        raise ValueError('Training data identity differs from checkpoint')
    original_data.close()
    if noise_kind is not None:
        if noise_kind not in ('none', 'bank_missing', 'procedural_missing', 'gaussian', 'stripes'):
            raise ValueError('Unknown evaluation noise kind')
        config['noise']['kind'] = noise_kind
    result = evaluate_model(model, config, device, coverage, output)
    result.update(checkpoint_sha256=sha256_file(checkpoint_path), step=checkpoint['step'],
                  training_noise_kind=checkpoint['config']['noise']['kind'], evaluation_noise_override=noise_kind)
    json_write(Path(output)/'metrics.json', result)
    return result
