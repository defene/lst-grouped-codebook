"""Five exact epochs, resumable at optimizer boundaries, no test-based selection."""
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
import time
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from eo_data.core import sha256_file
from .io import json_write
from eo_data.locking import build_lock
from eo_denoise import engine
from eo_denoise.config import fingerprint
from eo_denoise.data import EpochSampler
from .metrics import EvaluationMetrics
from . import runtime
from experiments.paper1.lst_v2_5epoch.train import source_hashes as frozen_hashes
from .data import DatasetV2, epoch_updates

HERE = Path(__file__).resolve().parent


def source_hashes():
    return {**frozen_hashes(), **{'lst_grouped_vq/'+n: sha256_file(HERE/n)
            for n in ('data.py', 'train.py', 'run.py', 'runtime.py', 'quantizer.py', 'metrics.py', 'io.py')}}


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_log(path, row):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')


@torch.inference_mode()
def evaluate(model, c, device, split, coverage, output, checkpoint_hash=None):
    output.mkdir(parents=True, exist_ok=True)
    ds = DatasetV2(c, split, coverage)
    loader = DataLoader(ds, batch_size=c['eval']['batch_size'], num_workers=c['train']['workers'],
                        shuffle=False, pin_memory=True, generator=torch.Generator().manual_seed(20260915))
    metrics = EvaluationMetrics(ds.sources, model)
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    total = 0
    digest = hashlib.sha256()
    observed = []
    try:
        for batch in loader:
            observed.extend(batch['record_id'].tolist())
            digest.update(batch['record_id'].numpy().astype('<i8').tobytes())
            for field in ('image', 'input_mask', 'reference_mask', 'corruption_mask'):
                digest.update(batch['lst'][field].numpy().tobytes())
            moved = engine.move_batch(batch, device, model.codecs)
            with engine.amp_context(c, device):
                pred = model(engine.inputs_only(moved, model.codecs))
            metrics.add('lst', pred['lst'], moved['lst'], batch['sample_id'], batch['tile_id'])
            total += len(batch['sample_id'])
            if total % 256 == 0 or total == len(ds):
                json_write(output/'progress.json', {'status': 'running', 'processed': total, 'total': len(ds), 'utc': utc(),
                    'memory': engine.memory_stats(device)})
                if (output.parent/'STOP').exists() or (Path(c['experiment']['output_root'])/'STOP').exists():
                    raise RuntimeError('STOP requested during evaluation')
        assert observed == ds.record_ids.tolist()
        all_rows = [r for r in metrics.rows if r['region'] == 'all']
        assert len(all_rows) == total*c['eval']['realizations']
        mean_mse = float(np.mean([r['bias_physical']**2 for r in all_rows]))
        mse = float(np.mean([r['mse_physical'] for r in all_rows]))
        result = metrics.summary(c['eval']['bootstrap_replicates'])
        result.pop('selection_score', None)
        result.pop('selection_definition', None)
        result.update(split=split, coverage=coverage, evaluated_pairs=total, realizations=c['eval']['realizations'],
            mean_rmse=mean_mse**.5, spatial_rmse=max(0, mse-mean_mse)**.5,
            data_fingerprint=ds.fingerprint, input_digest=digest.hexdigest(), config_sha256=fingerprint(c),
            checkpoint_sha256=checkpoint_hash, rates={'lst': model.codecs['lst'].rate()},
            elapsed_seconds=time.perf_counter()-started, completed_utc=utc(), memory=engine.memory_stats(device),
            independent_validation=split=='val', historical_test_monitoring=split=='test')
        with (output/'samples.jsonl').open('w', encoding='utf-8') as f:
            for row in metrics.rows:
                f.write(json.dumps(row, allow_nan=False)+'\n')
        result['samples_sha256'] = sha256_file(output/'samples.jsonl')
        np.savez_compressed(output/'token_counts.npz', counts=metrics.counts, within=metrics.within, scales=metrics.scales)
        result['token_counts_sha256'] = sha256_file(output/'token_counts.npz')
        np.save(output/'record_ids.npy', ds.record_ids, allow_pickle=False)
        json_write(output/'metrics.json', result)
        return result
    finally:
        ds.close()
        model.train(was_training)


def read_checkpoint(path, c, ds):
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['source_hashes'] == source_hashes(), 'Frozen source changed'
    assert state['config_sha256'] == fingerprint(c), 'Config changed'
    assert state['data_fingerprint'] == ds.fingerprint, 'Data fingerprint changed'
    return state


def train(c, output, stop_after=None):
    output.mkdir(parents=True, exist_ok=True)
    with build_lock(output):
        try:
            return _train(c, output, stop_after)
        except BaseException as e:
            json_write(output/'status.json', {'status': 'failed', 'error': repr(e), 'utc': utc(), 'pid': os.getpid()})
            raise


def _train(c, output, stop_after):
    engine.seed_everything(c['seed'])
    torch.set_num_threads(4)
    device = engine.device_for(c)
    ds = DatasetV2(c, 'train')
    effective = c['train']['batch_size']*c['train']['accumulation']
    updates_per_epoch = math.ceil(len(ds)/effective)
    epochs = c['experiment']['epochs']
    assert c['train']['steps'] == epochs*updates_per_epoch
    model = runtime.ExperimentModel(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['train']['lr'], betas=(.9,.95), weight_decay=c['train']['weight_decay'])
    epoch, consumed, step, validated_epoch, records_presented = 0, 0, 0, 0, 0
    if (output/'last.pt').exists():
        state = read_checkpoint(output/'last.pt', c, ds)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch, consumed, step = state['epoch'], state['consumed'], state['step']
        validated_epoch, records_presented = state['validated_epoch'], state['records_presented']
        torch.set_rng_state(state['rng_cpu'])
        torch.cuda.set_rng_state_all(state['rng_cuda'])
        del state
        # Preserve any updates logged after the latest committed checkpoint as abandoned work.
        if (output/'train.jsonl').exists():
            lines = (output/'train.jsonl').read_text(encoding='utf-8').splitlines()
            committed = [s for s in lines if json.loads(s)['step'] <= step]
            if len(committed) != len(lines):
                (output/f'train_before_resume_{time.time_ns()}.jsonl').write_text('\n'.join(lines)+'\n', encoding='utf-8')
                (output/'train.jsonl').write_text('\n'.join(committed)+'\n', encoding='utf-8')
    else:
        assert not (output/'train.jsonl').exists(), 'Training log without checkpoint; use a fresh directory'
        json_write(output/'config.json', c)
        json_write(output/'data_fingerprint.json', ds.fingerprint)
        np.save(output/'train_record_ids.npy', ds.record_ids, allow_pickle=False)
        json_write(output/'environment.json', engine.environment(device))
        json_write(output/'model.json', {'parameters': sum(p.numel() for p in model.parameters()), 'pretrained_weights': False,
            'rates': {'lst': model.codecs['lst'].rate()}, 'source_hashes': source_hashes()})
        with zipfile.ZipFile(output/'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for directory in (HERE, HERE.parent/'lst_v2_5epoch', HERE.parent/'lst_residual', Path(engine.__file__).parent, HERE.parents[2]/'eo_data'):
                for p in directory.glob('*.py'):
                    z.write(p, str(p.relative_to(HERE.parents[2])))
    assert records_presented == epoch*len(ds)+consumed
    started = time.perf_counter()

    def save(path=None):
        engine.atomic_checkpoint(path or output/'last.pt', {'format': 2, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'epoch': epoch, 'consumed': consumed, 'step': step, 'validated_epoch': validated_epoch,
            'records_presented': records_presented, 'config': c, 'config_sha256': fingerprint(c),
            'data_fingerprint': ds.fingerprint, 'source_hashes': source_hashes(),
            'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()})

    def status(value='running', stage='train'):
        r = {'status': value, 'stage': stage, 'epoch_completed': epoch, 'consumed_in_epoch': consumed,
            'step': step, 'configured_steps': c['train']['steps'], 'records_presented': records_presented,
            'validated_epoch': validated_epoch, 'pid': os.getpid(), 'utc': utc(),
            'memory': engine.memory_stats(device), 'session_elapsed_seconds': time.perf_counter()-started}
        json_write(output/'status.json', r)
        return r

    def validate_epoch():
        nonlocal validated_epoch
        status(stage='full_validation')
        epoch_path = output/f'epoch_{epoch:02d}.pt'
        if not epoch_path.exists():
            save(epoch_path)
        evaluate(model, c, device, 'val', .1, output/f'val_epoch_{epoch:02d}', sha256_file(epoch_path))
        validated_epoch = epoch
        save()

    model.train()
    try:
        if epoch > validated_epoch:
            validate_epoch()
        while epoch < epochs:
            sampler = EpochSampler(len(ds), c['seed'], epoch, consumed)
            loader = DataLoader(ds, batch_size=c['train']['batch_size'], sampler=sampler, drop_last=False,
                num_workers=c['train']['workers'], pin_memory=True,
                generator=torch.Generator().manual_seed(c['seed']+epoch))
            iterator = iter(loader)
            for batches in epoch_updates(iterator, c['train']['accumulation']):
                total = sum(len(b['sample_id']) for b in batches)
                assert total == min(effective, len(ds)-consumed)
                optimizer.zero_grad(set_to_none=True)
                for g in optimizer.param_groups:
                    g['lr'] = engine.learning_rate(c, step)
                logs = {}
                for batch in batches:
                    moved = engine.move_batch(batch, device, model.codecs)
                    with engine.amp_context(c, device):
                        pred = model(engine.inputs_only(moved, model.codecs))
                    loss, terms = runtime.objective(pred, moved, c)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite objective')
                    weight = len(batch['sample_id'])/total
                    (loss*weight).backward()
                    for k, v in dict(terms, total=loss).items():
                        logs[k] = logs.get(k, 0)+float(v.detach())*weight
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), c['train']['clip'], error_if_nonfinite=True)
                optimizer.step()
                consumed += total
                records_presented += total
                step += 1
                logs.update(step=step, epoch_index=epoch, consumed=consumed, batch_samples=total,
                    records_presented=records_presented, gradient_norm=float(grad), lr=optimizer.param_groups[0]['lr'],
                    sample_ids=[s for b in batches for s in b['sample_id']], utc=utc(),
                    elapsed_seconds=time.perf_counter()-started, memory=engine.memory_stats(device))
                write_log(output/'train.jsonl', logs)
                ended = consumed == len(ds)
                if ended:
                    epoch, consumed = epoch+1, 0
                    assert step == epoch*updates_per_epoch and records_presented == epoch*len(ds)
                stopped = (output/'STOP').exists() or (stop_after is not None and step >= stop_after)
                low_memory = logs['memory']['device_free_gib'] < 2
                if step % c['train']['save_every'] == 0 or ended or stopped or low_memory:
                    save()
                if step % c['train']['log_every'] == 0 or ended or stopped:
                    status()
                    print(json.dumps({k: logs[k] for k in ('step', 'epoch_index', 'batch_samples', 'total', 'records_presented')}), flush=True)
                if stopped or low_memory:
                    return status('stopped_at_checkpoint')
                if ended:
                    break
            del iterator, loader, batches, moved, pred, loss, terms
            gc.collect()
            assert consumed == 0
            validate_epoch()
        assert step == c['train']['steps'] and records_presented == epochs*len(ds)
        return status('complete')
    finally:
        ds.close()


def final_evaluate(c, output):
    ds = DatasetV2(c, 'train')
    state = read_checkpoint(output/'last.pt', c, ds)
    assert state['epoch'] == c['experiment']['epochs'] and state['consumed'] == 0
    assert state['records_presented'] == len(ds)*c['experiment']['epochs']
    ds.close()
    engine.seed_everything(c['seed'])
    torch.set_num_threads(4)
    device = engine.device_for(c)
    model = runtime.ExperimentModel(c).to(device).eval()
    model.load_state_dict(state['model'])
    del state
    digest = sha256_file(output/'last.pt')
    for split in ('val', 'test'):
        for condition, coverage in [('clean', 0), ('noisy10', .1)]:
            dest = output/f'final_{split}_{condition}'
            if (dest/'metrics.json').exists():
                old = json.loads((dest/'metrics.json').read_text(encoding='utf-8'))
                assert old['checkpoint_sha256'] == digest and old['config_sha256'] == fingerprint(c)
                continue
            evaluate(model, c, device, split, coverage, dest, digest)
