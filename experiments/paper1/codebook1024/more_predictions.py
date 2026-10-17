"""Read-only, tile-diverse A02/A03 VQ-VAE predictions, using final evaluation view 0."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import gc
import json
import math
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eo_data.core import sha256_file
from eo_denoise.data import PairedDataset
from eo_denoise.engine import read_checkpoint, seed_everything, device_for, amp_context, source_hashes, memory_stats
from eo_denoise.models import CrossViewModel
from eo_denoise.noise import stable_seed

DEST = ROOT / 'runs/paper1/codebook1024/full/report/more_predictions'
ROOTS = {4096: ROOT / 'runs/paper1/var_vs_prithvi_ae_full',
         1024: ROOT / 'runs/paper1/codebook1024/full'}
SEED = 20260913


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def gradient_energy(x, mask):
    values = []
    for diff, support in [(np.diff(x, axis=-1), mask[:, 1:] & mask[:, :-1]),
                           (np.diff(x, axis=-2), mask[1:] & mask[:-1])]:
        if support.any():
            values.append(float(np.mean(diff[:, support].astype(np.float64)**2)))
    return math.sqrt(float(np.mean(values))) if values else None


def crop_from_reference(x, mask):
    choices = []
    for y in range(0, 193, 16):
        for xx in range(0, 193, 16):
            support = mask[y:y+64, xx:xx+64]
            choices.append((float(support.mean()), gradient_energy(x[:, y:y+64, xx:xx+64], support) or 0, y, xx))
    eligible = [r for r in choices if r[0] >= .9]
    best = max(eligible, key=lambda r: r[1]) if eligible else max(choices)
    return dict(y=best[2], x=best[3], height=64, width=64, valid_fraction=best[0],
                reference_gradient_rms=best[1], rule='maximum reference-only gradient RMS among 64x64 windows at stride16 with >=90% valid pixels; fallback maximum validity')


def region_metrics(pred, target, mask):
    if not mask.any():
        return None
    e = (pred-target)[:, mask].astype(np.float64)
    gp, gt = gradient_energy(pred, mask), gradient_energy(target, mask)
    return dict(pixels=int(mask.sum()), rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.abs(e).mean()),
                bias=float(e.mean()), reference_gradient_rms=gt, prediction_gradient_rms=gp,
                gradient_amplitude_ratio=gp/gt if gp is not None and gt is not None and gt > 1e-12 else None,
                gradient_rmse=gradient_energy(pred-target, mask))


@torch.inference_mode()
def main():
    DEST.mkdir(exist_ok=True)
    if (DEST / 'results.json').exists():
        raise FileExistsError('Gallery predictions already exist; use the saved arrays instead of rerunning inference.')
    started = time.perf_counter()
    torch.set_num_threads(4)
    source = ROOTS[1024] / 'full_hls_conv_clean_seed17/eval_clean/samples.jsonl'
    tiles = {}
    with source.open(encoding='utf-8') as f:
        for r in map(json.loads, f):
            if r['view'] == 0 and r['region'] == 'all' and r['sample_id'] != 'prithvi:val:230660':
                tiles.setdefault(r['tile_id'], []).append(r['sample_id'])
    selected_tiles = sorted(tiles, key=lambda t: stable_seed('gallery_tile', SEED, t))[:16]
    samples = [dict(tile_id=t, sample_id=min(tiles[t], key=lambda s: stable_seed('gallery_record', SEED, s)))
               for t in selected_tiles]
    for i, s in enumerate(samples):
        s.update(number=i+1, record_id=int(s['sample_id'].rsplit(':', 1)[1]))
    assert len({s['record_id'] for s in samples}) == len({s['tile_id'] for s in samples}) == 16
    manifest = dict(status='preparing', created_utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
                    selection='16 validation tiles ranked by seeded hash, then one seeded record per tile, excluding the previous first preview; no model-output selection',
                    sample_count=16, full_validation_count=11757, view=0, samples=samples,
                    source_samples_sha256=sha256_file(source), frozen_sources=source_hashes(),
                    inference_only=True, checkpoint_steps=10813)
    write(DEST / 'selection.json', manifest)
    write(DEST / 'status.json', dict(status='preparing', completed_models=0))
    data = {}
    for m in ('hls', 'lst'):
        run = ROOTS[1024] / f'full_{m}_conv_clean_seed17'
        c = read(run / 'config.json')
        c['data']['val_limit'] = 11757
        arrays, metadata = {}, []
        for condition, coverage in [('clean', 0), ('noisy10', .1)]:
            ds = PairedDataset(c, 'val', coverage=coverage)
            assert ds.fingerprint == read(run / f'eval_{condition}/metrics.json')['data_fingerprint']
            position = {int(rid): i for i, rid in enumerate(ds.record_ids)}
            entries = [ds[position[s['record_id']]] for s in samples]
            assert [e['sample_id'] for e in entries] == [s['sample_id'] for s in samples]
            assert [e['tile_id'] for e in entries] == selected_tiles
            target = np.stack([e[m]['target'] for e in entries])
            reference = np.stack([e[m]['reference_mask'] for e in entries])
            if condition == 'clean':
                arrays.update(target=target, reference_mask=reference)
            else:
                assert np.array_equal(arrays['target'], target) and np.array_equal(arrays['reference_mask'], reference)
            for key in ('image', 'input_mask', 'corruption_mask'):
                arrays[condition+'_'+key] = np.stack([e[m][key][0] for e in entries])
            arrays[condition+'_actual_coverage'] = np.array([e[m]['actual_coverage'][0] for e in entries])
            stats = ds.sources[m].stats
            mean = np.array(stats['mean'], np.float32)[:, None, None]
            std = np.array(stats['std'], np.float32)[:, None, None]
            arrays['mean'], arrays['std'] = mean, std
            if condition == 'clean':
                for i, s in enumerate(samples):
                    meta = ds.sources[m].get_metadata(int(ds.positions[m][position[s['record_id']]]))
                    physical = (target[i]*std+mean) / (100 if m == 'lst' else 1)
                    support = reference[i, 0].astype(bool)
                    metadata.append(dict(**s, datetime=meta['datetime'], valid_fraction=float(support.mean()),
                        reference_gradient_rms=gradient_energy(physical, support),
                        crop=crop_from_reference(physical, support),
                        display_limits=np.quantile(physical[0, support], [.01, .99]).tolist() if m == 'lst' else None))
            ds.close()
        arrays['record_ids'] = np.array([s['record_id'] for s in samples])
        np.savez_compressed(DEST / f'{m}_inputs.npz', **arrays)
        write(DEST / f'{m}_samples.json', metadata)
        data[m] = arrays
    manifest.update(status='selected_and_inputs_verified', input_hashes={m:sha256_file(DEST/f'{m}_inputs.npz') for m in data})
    write(DEST / 'selection.json', manifest)
    print('Selected 16 new pairs from 16 tiles; inputs and frozen fingerprints verified.', flush=True)
    results = dict(status='running', samples=samples, rows=[], models={}, source_hashes=source_hashes())
    for size, root in ROOTS.items():
        assert read(root / 'manifest.json')['source_hashes'] == source_hashes()
        for m in ('hls', 'lst'):
            arrays = data[m]
            std, mean = arrays['std'], arrays['mean']
            scale = 100 if m == 'lst' else 1
            for training in ('clean', 'noisy'):
                name = f'full_{m}_conv_{training}_seed17'
                run = root / name
                checkpoint_sha = sha256_file(run / 'last.pt')
                checkpoint = read_checkpoint(run / 'last.pt')
                config = checkpoint['config']
                assert config == read(run / 'config.json') and config['model']['codebook_size'] == size
                seed_everything(config['seed'])
                device = device_for(config)
                model = CrossViewModel(config).to(device).eval()
                model.load_state_dict(checkpoint['model'], strict=True)
                del checkpoint
                key = f'{size}_{m}_{training}'
                saved, agreement = {}, []
                for condition in ('clean', 'noisy10'):
                    fp = read(run / f'eval_{condition}/metrics.json')
                    assert fp['checkpoint_sha256'] == checkpoint_sha and fp['step'] == 10813
                    all_outputs = []
                    for start in range(0, 16, 8):
                        image = torch.from_numpy(arrays[condition+'_image'][start:start+8]).to(device)
                        mask = torch.from_numpy(arrays[condition+'_input_mask'][start:start+8]).to(device)
                        with amp_context(config, device):
                            pred = model.codecs[m](image, mask)['reconstruction']
                        all_outputs.append(pred.float().cpu().numpy())
                        del image, mask, pred
                    predictions = np.concatenate(all_outputs)
                    assert predictions.shape == arrays['target'].shape and np.isfinite(predictions).all()
                    saved[condition+'_prediction'] = predictions
                    old_rows = {}
                    with (run / f'eval_{condition}/samples.jsonl').open(encoding='utf-8') as f:
                        for row in map(json.loads, f):
                            if row['view'] == 0 and row['region'] == 'all' and row['sample_id'] in {s['sample_id'] for s in samples}:
                                old_rows[row['sample_id']] = row
                    assert len(old_rows) == 16
                    for i, sample in enumerate(samples):
                        physical_pred = (predictions[i]*std+mean)/scale
                        physical_target = (arrays['target'][i]*std+mean)/scale
                        reference = arrays['reference_mask'][i, 0].astype(bool)
                        corrupt = arrays[condition+'_corruption_mask'][i, 0].astype(bool) & reference
                        regions = {n:region_metrics(physical_pred, physical_target, mask) for n, mask in
                                   [('all', reference), ('corrupt', corrupt), ('retained', reference & ~corrupt)]}
                        # Use normalized differences as in the frozen evaluator to avoid denormalization cancellation.
                        error = ((predictions[i]-arrays['target'][i])*std/scale)[:, reference]
                        rmse = float(np.sqrt(np.mean(error.astype(np.float64)**2)))
                        old_rmse = math.sqrt(old_rows[sample['sample_id']]['mse_physical'])
                        relative = abs(rmse-old_rmse)/max(old_rmse, 1e-12)
                        assert math.isclose(rmse, old_rmse, rel_tol=5e-3, abs_tol=1e-5), (key, sample['sample_id'], condition, rmse, old_rmse)
                        agreement.append(relative)
                        results['rows'].append(dict(model=key, codebook_size=size, modality=m, training=training,
                            condition=condition, **sample, coverage=float(arrays[condition+'_actual_coverage'][i]),
                            regions=regions, final_evaluation_rmse=old_rmse, recomputed_normalized_difference_rmse=rmse))
                np.savez_compressed(DEST / f'{key}_predictions.npz', **saved)
                results['models'][key] = dict(checkpoint=str((run/'last.pt').relative_to(ROOT)),
                    checkpoint_sha256=checkpoint_sha, predictions_sha256=sha256_file(DEST/f'{key}_predictions.npz'),
                    saved_full_evaluation_max_relative_rmse_difference=max(agreement), memory=memory_stats(device))
                print(json.dumps(dict(model=key, complete=True, max_relative_rmse_difference=max(agreement), memory=memory_stats(device))), flush=True)
                del model, config, predictions, all_outputs, saved
                gc.collect()
                torch.cuda.empty_cache()
                write(DEST/'status.json', dict(status='running', completed_models=len(results['models']), last_completed=key))
    results.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
                   gradient_ratio_note='RMS of prediction spatial differences divided by reference RMS on valid adjacent pixels; low values indicate less spatial variation, not proof of cause or true detail fidelity')
    write(DEST / 'results.json', results)
    write(DEST / 'status.json', dict(status='complete', completed_models=8, predictions=256))
    print('INFERENCE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
