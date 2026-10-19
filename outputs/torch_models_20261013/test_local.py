"""Local inference for all 39 frozen LST models, with their own source snapshots.

Defaults to CPU. Saved-gallery mode explicitly uses the original oracle mean.
Custom input NPZ must supply observation, reference_valid, observed_valid and
mean_celsius; no clean reference is implicitly substituted for a missing mean.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='sd_C_affine')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--check-load', action='store_true')
    parser.add_argument('--all', action='store_true', help='With --check-load, verify all 39 in separate processes')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--condition', default='pixel10', help='Saved gallery condition prefix or exact condition ID')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--input', type=Path, help='Custom Celsius observation NPZ; requires explicit scalar mean_celsius')
    parser.add_argument('--output', type=Path, help='New directory for predictions and comparison image')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'MODEL_MANIFEST.json').read_text(encoding='utf-8'))
    registry = {m['name']: m for m in manifest['models']}
    if args.list:
        for name, m in registry.items():
            print(f"{name:28s} {m['weights_bytes'] / 1e6:.1f} MB  epoch={m['final_epoch']}")
        return
    if args.all:
        if not args.check_load or args.device != 'cpu':
            parser.error('--all is supported only with --check-load --device cpu')
        from concurrent.futures import ThreadPoolExecutor
        def check_one(name):
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--model', name,
                                     '--check-load', '--threads', str(args.threads)],
                                    capture_output=True, text=True, encoding='utf-8', errors='replace')
            logs = ROOT / 'checks'
            logs.mkdir(exist_ok=True)
            (logs / (name + '.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
            if result.returncode:
                raise RuntimeError(f'Loading failed for {name}; see checks/{name}.log')
            return name
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i, name in enumerate(pool.map(check_one, registry), 1):
                print(f'PASS {i}/{len(registry)} {name}', flush=True)
        return
    if args.model not in registry:
        parser.error('Unknown model; use --list')
    item = registry[args.model]
    source = ROOT / 'sources' / item['source_stage']
    sys.path.insert(0, str(source))
    # Bind namespace parents so another checkout cannot supply a different model.
    for package, relative in [('experiments', 'experiments'), ('experiments.paper1', 'experiments/paper1')]:
        module = types.ModuleType(package)
        module.__path__ = [str(source / relative)]
        sys.modules[package] = module
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    import numpy as np
    import torch
    from experiments.paper1.lst_restoration.models import build_model
    from experiments.paper1.lst_restoration.representations import Representation
    from experiments.paper1.lst_restoration.objective import model_inputs, readout
    if not Path(sys.modules[build_model.__module__].__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError('Model import resolved outside its frozen source snapshot')
    torch.set_num_threads(args.threads)
    config = json.loads((ROOT / item['config']).read_text(encoding='utf-8'))
    original_config = copy.deepcopy(config)
    config['execution']['enabled'] = False
    config['tracking']['enabled'] = False
    if item.get('pretrained_local'):
        config['model']['pretrained_path'] = str(ROOT / item['pretrained_local'])
    weight = ROOT / item['weights']
    with weight.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    if sha != item['weights_sha256']:
        raise RuntimeError('Model weights SHA256 mismatch')
    state = torch.load(weight, map_location='cpu', weights_only=True, mmap=True)
    if state['original_checkpoint_sha256'] != item['original_checkpoint_sha256']:
        raise RuntimeError('Original checkpoint provenance mismatch')
    model = build_model(config)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    model.requires_grad_(False)
    del state
    receipt = {'model': args.model, 'strict_loading': 'passed', 'source_stage': item['source_stage'],
               'weights_sha256': sha, 'original_checkpoint_sha256': item['original_checkpoint_sha256'],
               'epoch': item['final_epoch'], 'inference_run': False}
    if args.check_load:
        checks = ROOT / 'checks'
        checks.mkdir(exist_ok=True)
        (checks / (args.model + '.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps(receipt))
        return
    if args.output is None:
        parser.error('Inference requires --output pointing to a new directory')
    destination = args.output.resolve()
    if destination.exists():
        parser.error('Output directory exists; choose a new path to preserve earlier results')
    reference = stored = None
    sample_id = None
    if args.input:
        with np.load(args.input, allow_pickle=False) as z:
            observation = np.asarray(z['observation'], dtype=np.float32).reshape(1, 1, 256, 256)
            reference_valid = np.asarray(z['reference_valid'], dtype=bool).reshape(1, 1, 256, 256)
            observed_valid = np.asarray(z['observed_valid'], dtype=bool).reshape(1, 1, 256, 256)
            mean = np.asarray(z['mean_celsius'], dtype=np.float32).reshape(1)
            if 'temperature' in z:
                reference = np.asarray(z['temperature'], dtype=np.float32).reshape(1, 1, 256, 256)
        input_source = str(args.input.resolve())
        mean_note = 'explicit user-supplied mean; original training used oracle reference mean'
    else:
        results = ROOT.parent / 'torch_experiments_20261013'
        index = json.loads((results / 'results_index.json').read_text(encoding='utf-8'))
        run = next(r for r in index['runs'] if r['model'] == args.model)
        matches = [c for c in run['cells'] if c['condition'] == args.condition or c['condition'].startswith(args.condition + '_')]
        if len(matches) != 1:
            parser.error('Condition is missing or ambiguous; pass the exact condition ID from results_index.json')
        gallery = (results / matches[0]['source']).with_name('gallery.npz')
        with np.load(gallery, allow_pickle=False) as z:
            i = args.sample_index
            if not 0 <= i < len(z['sample_id']):
                parser.error('Sample index outside saved gallery')
            sample_id = str(z['sample_id'][i])
            observation = z['observation'][i:i+1].astype(np.float32)
            reference_valid = z['reference_valid'][i:i+1].astype(bool)
            observed_valid = z['observed_valid'][i:i+1].astype(bool)
            reference = z['temperature'][i:i+1].astype(np.float32)
            stored = z['prediction_raw'][i:i+1].astype(np.float32)
        mean = np.array([reference[reference_valid].mean(dtype=np.float64)], dtype=np.float32)
        input_source = str(gallery)
        mean_note = 'oracle reference mean, explicitly reproduced from saved validation reference'
    if not reference_valid.any() or (observed_valid & ~reference_valid).any():
        raise ValueError('Invalid support masks')
    if not np.isfinite(observation[observed_valid]).all() or not np.isfinite(mean).all():
        raise ValueError('Nonfinite observed values or mean')
    if reference is not None and not np.isfinite(reference[reference_valid]).all():
        raise ValueError('Nonfinite reference values')
    if config['task']['field'] != 'residual':
        raise ValueError('This entry point is for the registered residual models')
    field = np.where(observed_valid, observation - mean.reshape(1, 1, 1, 1), 0).astype(np.float32)
    device = torch.device(args.device)
    model = model.to(device)
    batch = {k: torch.from_numpy(v).to(device) for k, v in {
        'input_field': field, 'reference_valid': reference_valid, 'observed_valid': observed_valid,
        'mean_celsius': mean, 'center_celsius': mean,
        'patch_missing': np.zeros((1, 1, 256, 256), dtype=bool)}.items()}
    representation = Representation(config['representation'])
    with torch.inference_mode():
        prediction = model(**model_inputs(batch, representation))
        raw = readout(prediction, batch, representation, config).cpu().numpy()
        projected = readout(prediction, batch, representation, config, constrained=True).cpu().numpy()
    if not np.isfinite(raw[reference_valid]).all():
        raise RuntimeError('Nonfinite prediction on valid support')
    receipt.update(inference_run=True, device=str(device), precision='float32; no autocast',
                   input_source=input_source, sample_id=sample_id, mean_celsius=float(mean[0]), mean_note=mean_note,
                   note='Single-example local test; not a rerun of full validation. Server BF16 and local FP32 can differ.')
    if reference is not None:
        receipt['sample_raw_rmse_C'] = float(np.sqrt(np.mean((raw[reference_valid].astype(np.float64) - reference[reference_valid]) ** 2)))
    if stored is not None:
        receipt['difference_from_saved_prediction_max_abs_C'] = float(np.max(np.abs(raw[reference_valid] - stored[reference_valid])))
    destination.mkdir(parents=True)
    arrays = dict(observation=observation, reference_valid=reference_valid, observed_valid=observed_valid,
                  mean_celsius=mean, prediction_raw=raw, prediction_projected=projected)
    if reference is not None:
        arrays['temperature'] = reference
    np.savez_compressed(destination / 'prediction.npz', **arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mask = reference_valid[0, 0]
    display_reference = reference[0, 0] if reference is not None else observation[0, 0]
    display_valid = mask if reference is not None else observed_valid[0, 0]
    if display_valid.any():
        lo, hi = float(display_reference[display_valid].min()), float(display_reference[display_valid].max())
    else:
        lo, hi = float(mean[0]) - 1, float(mean[0]) + 1
    hi = max(hi, lo + .01)
    fields = [('Observation (C)', np.where(observed_valid[0, 0], observation[0, 0], np.nan), 'viridis', lo, hi),
              ('Raw prediction (C)', np.where(mask, raw[0, 0], np.nan), 'viridis', lo, hi)]
    if reference is not None:
        error = raw[0, 0] - reference[0, 0]
        limit = max(.01, float(np.max(np.abs(error[mask]))))
        fields.insert(0, ('Reference (C)', np.where(mask, reference[0, 0], np.nan), 'viridis', lo, hi))
        fields.append(('Error (C)', np.where(mask, error, np.nan), 'coolwarm', -limit, limit))
    fig, axes = plt.subplots(1, len(fields), figsize=(4 * len(fields), 4), layout='constrained')
    for ax, (title, field, cmap, lower, upper) in zip(axes, fields):
        im = ax.imshow(field, cmap=cmap, vmin=lower, vmax=upper)
        ax.set_title(title); ax.set_axis_off(); fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(f'{args.model} | {device} | FP32 single-example test')
    fig.savefig(destination / 'prediction.png', dpi=140)
    plt.close(fig)
    (destination / 'result.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    (destination / 'config.original.json').write_text(json.dumps(original_config, indent=2), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
