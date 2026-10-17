"""Full validation at epoch boundaries, with restartable checkpoint selection.

Only Python's standard library is imported during static configuration checks.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import shutil


def default_protocol():
    return {
        'split': 'val', 'every_epochs': 1, 'views': 2,
        'profiles': [{'kind': 'clean', 'seed': 20260925},
                     {'kind': 'pixel10', 'coverage': .1, 'seed': 20260925}],
        'selection_profile': 'pixel10', 'selection_metric': 'raw.pixel_weighted_rmse',
        'mode': 'min', 'tie_break': 'earliest_epoch', 'retain_all_epochs': True,
        'rng_seed': 20260925,
    }


def protocol_errors(config):
    errors = []
    if config.get('epoch_validation') != default_protocol():
        errors.append('Epoch validation requires the registered full-val clean/pixel10 protocol')
    if config['corruption']['train_profiles'] != [default_protocol()['profiles'][1]]:
        errors.append('This version selects for pixel10 training only; register other tasks separately')
    if config['evaluation']['stage'] != 'selection' or config['evaluation']['splits'] != ['val']:
        errors.append('This selection run must not evaluate test')
    if config['evaluation']['batch_size'] < 1:
        errors.append('Positive validation batch size required')
    return errors


@contextmanager
def isolated_inference(model, device, seed):
    """Restore every module mode and Python/NumPy/Torch RNG, even on failure."""
    import numpy as np
    import torch
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            model.eval()
            with torch.inference_mode():
                yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        for module, mode in modes:
            module.training = mode


def measure_validation(model, config, device, stop_root):
    import torch
    from torch.utils.data import DataLoader
    from .config import COUNTS
    from .data import LSTDataset
    from .objective import model_inputs, readout
    from .representations import Representation
    from .runtime import amp, check_stop, move_batch
    from .tracking import log
    representation = Representation(config['representation'])
    options = config['epoch_validation']
    cells = []
    from .usage import Usage
    from .config import write_json
    from .runtime import utc
    with isolated_inference(model, device, options['rng_seed']):
        for profile in options['profiles']:
            usage=Usage(config)
            squared, absolute, count, image_mse, rows = 0., 0., 0, 0., 0
            fingerprint = None
            for view in range(options['views']):
                dataset = LSTDataset(config, 'val', profile, view)
                try:
                    if len(dataset) != COUNTS['val']:
                        raise ValueError('Epoch validation must cover the entire validation split')
                    if fingerprint is not None and fingerprint != dataset.fingerprint:
                        raise ValueError('Validation data identity changed between views')
                    fingerprint = dataset.fingerprint
                    seen = []
                    loader = DataLoader(dataset, batch_size=config['evaluation']['batch_size'],
                        shuffle=False, drop_last=False, num_workers=config['train']['workers'],
                        pin_memory=device.type == 'cuda',
                        generator=torch.Generator().manual_seed(options['rng_seed']))
                    for cpu_batch in loader:
                        check_stop(stop_root)
                        seen.extend(cpu_batch['record_id'].tolist())
                        batch = move_batch(cpu_batch, device)
                        with amp(config, device):
                            predicted = model(**model_inputs(batch, representation))
                        if predicted.get('indices') is not None:usage.add(predicted['indices'])
                        raw = readout(predicted, batch, representation, config)
                        valid = batch['reference_valid'].bool()
                        error = raw.double() - batch['temperature'].double()
                        if not torch.isfinite(error[valid]).all():
                            raise FloatingPointError('Non-finite validation error; no checkpoint selected')
                        safe_error = torch.where(valid, error, 0.)
                        counts = valid.flatten(1).sum(1)
                        if not (counts > 0).all():
                            raise ValueError('Validation image has no valid pixels')
                        sums = safe_error.square().flatten(1).sum(1)
                        squared += sums.sum().item()
                        absolute += safe_error.abs().sum().item()
                        count += counts.sum().item()
                        # Same equal-image normalized reconstruction term as training.
                        image_mse += (sums / counts / config['task']['residual_std_celsius'] ** 2).sum().item()
                        rows += len(counts)
                        if rows%256 < len(counts):
                            write_json(Path(stop_root)/'validation_progress.json',dict(profile=profile['kind'],
                                record_views_completed=rows,total_record_views=COUNTS['val']*options['views'],utc=utc()))
                    if seen != dataset.record_ids.tolist() or len(set(seen)) != len(seen):
                        raise ValueError('Incomplete, duplicated or reordered validation membership')
                finally:
                    dataset.close()
                log({'val/progress_profile': profile['kind'], 'val/progress_rows': rows})
            if rows != COUNTS['val'] * options['views'] or not count:
                raise ValueError('Incomplete validation views')
            cells.append({'profile': profile, 'split': 'val', 'views': options['views'],
                'unique_records': COUNTS['val'], 'row_count': rows, 'valid_pixels': count,
                'data_fingerprint': fingerprint,
                'raw': {'pixel_weighted_rmse': math.sqrt(squared / count),
                        'pixel_weighted_mae': absolute / count},
                'loss_pixel': image_mse / rows,
                'loss_definition': 'equal-image mean((raw_C-reference_C)^2 / train_residual_std_C^2); valid pixels only',
                'code_usage':usage.result() if usage.counts.sum() else None})
    return cells


def atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def reconcile_selection(config, output, through_epoch):
    """Rebuild best.pt from committed receipts; no missing epoch may be skipped."""
    from .config import COUNTS, file_digest, write_json
    from .runtime import protocol_digest, source_hashes
    output = Path(output)
    history = []
    fingerprint = None
    for epoch in range(1, through_epoch + 1):
        receipt = output / 'epoch_validation' / f'epoch_{epoch:02d}.json'
        if not receipt.exists():
            raise RuntimeError(f'Missing completed validation for epoch {epoch}; cannot advance')
        row = json.loads(receipt.read_text(encoding='utf-8'))
        checkpoint = output / 'epochs' / f'epoch_{epoch:02d}.pt'
        if (row['epoch'] != epoch or row['protocol_config_sha256'] != protocol_digest(config)
                or row['source_hashes'] != source_hashes()
                or row['checkpoint_sha256'] != file_digest(checkpoint)):
            raise ValueError('Epoch receipt/checkpoint provenance mismatch')
        cells = row['cells']
        if ([c['profile'] for c in cells] != config['epoch_validation']['profiles']
                or any(c['split'] != 'val' or c['unique_records'] != COUNTS['val']
                       or c['views'] != 2 or c['row_count'] != 2 * COUNTS['val'] for c in cells)):
            raise ValueError('Incomplete validation receipt')
        current = [c['data_fingerprint'] for c in cells]
        if fingerprint is not None and fingerprint != current:
            raise ValueError('Validation data changed across epochs')
        fingerprint = current
        score = cells[1]['raw']['pixel_weighted_rmse']
        if not math.isfinite(score) or score < 0:
            raise ValueError('Invalid selection metric')
        history.append({'epoch': epoch, 'score': score, 'checkpoint': str(checkpoint.relative_to(output)),
                        'checkpoint_sha256': row['checkpoint_sha256'],
                        'validation_receipt_sha256': file_digest(receipt)})
    best = min(history, key=lambda row: (row['score'], row['epoch']))
    best_path = output / 'best.pt'
    if not best_path.exists() or file_digest(best_path) != best['checkpoint_sha256']:
        atomic_copy(output / best['checkpoint'], best_path)
    selection = {'protocol': config['epoch_validation'], 'through_epoch': through_epoch,
                 'best': best, 'history': history, 'protocol_config_sha256': protocol_digest(config),
                 'source_hashes': source_hashes(), 'test_used': False}
    write_json(output / 'selection.json', selection)
    return selection


def finish_epoch(model, config, device, output, epoch, step):
    from .config import file_digest, write_json
    from .runtime import check_stop, protocol_digest, source_hashes, utc
    from . import tracking
    output = Path(output)
    check_stop(output)
    checkpoint = output / 'epochs' / f'epoch_{epoch:02d}.pt'
    last_hash = file_digest(output / 'last.pt')
    if checkpoint.exists():
        if file_digest(checkpoint) != last_hash:
            raise ValueError('Refusing to overwrite an immutable epoch checkpoint')
    else:
        atomic_copy(output / 'last.pt', checkpoint)
    receipt = output / 'epoch_validation' / f'epoch_{epoch:02d}.json'
    if not receipt.exists():
        tracking.phase('epoch_validation')
        write_json(output/'status.json',{'status':'epoch_validation','epoch':epoch,'step':step,'utc':utc()})
        cells = measure_validation(model, config, device, output)
        write_json(receipt, {'epoch': epoch, 'step': step, 'utc': utc(), 'cells': cells,
            'checkpoint_sha256': last_hash, 'protocol_config_sha256': protocol_digest(config),
            'source_hashes': source_hashes()})
    selected = reconcile_selection(config, output, epoch)
    cells = json.loads(receipt.read_text(encoding='utf-8'))['cells']
    values = {'val/epoch': epoch, 'val/train_step': step,
              'val/best_epoch': selected['best']['epoch'], 'val/best_rmse': selected['best']['score']}
    for cell in cells:
        prefix = 'val/' + cell['profile']['kind']
        values[prefix + '/loss_pixel'] = cell['loss_pixel']
        values[prefix + '/rmse_celsius'] = cell['raw']['pixel_weighted_rmse']
        values[prefix + '/mae_celsius'] = cell['raw']['pixel_weighted_mae']
    tracking.log(values)
    tracking.phase('training')
    print(json.dumps({'event':'epoch_validated','epoch':epoch,'step':step,
        'rmse_c':cells[1]['raw']['pixel_weighted_rmse'],'best_epoch':selected['best']['epoch']}),flush=True)
    return selected
