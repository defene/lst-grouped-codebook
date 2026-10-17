"""Frozen-checkpoint audit: between/within-image variation and token shuffling.

No training, no changes to model/data sources. Uses the existing 256-record
validation diagnostic subset and final evaluation noise realization zero.
"""
from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eo_data.core import sha256_file
from eo_denoise.data import PairedDataset
from eo_denoise.engine import read_checkpoint, seed_everything, device_for, amp_context, source_hashes, memory_stats
from eo_denoise.models import CrossViewModel
from eo_denoise.noise import stable_seed
from experiments.paper1.codebook1024.spatial_token_stats import reconstruction_parts, variance_summary, token_summary
from experiments.paper1.codebook1024.more_predictions import read, write, ROOTS

DEST = ROOT / 'runs/paper1/codebook1024/full/report/spatial_token_audit'
SEED = 20260914


def combine(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def describe_parts(p):
    target = variance_summary(p['target_mean'], p['target_within'])
    prediction = variance_summary(p['pred_mean'], p['pred_within'])
    em, es = float(p['mean_mse'].mean()), float(p['centered_mse'].mean())
    return dict(target=target, prediction=prediction, mean_mse=em, centered_mse=es, mse=em+es,
                rmse=(em+es)**.5, spatial_r2=1-es/target['within'],
                image_mean_r2=1-em/target['between'],
                spatial_variance_ratio=prediction['within']/target['within'])


def paired_change(native, changed):
    # Exactly one probe per distinct tile, so this resamples tiles as well as images.
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(len(native['mse']), size=(2000, len(native['mse'])))
    result = {}
    for k in ('mse', 'mean_mse', 'centered_mse'):
        delta = changed[k]-native[k]
        result[k] = dict(native=float(native[k].mean()), changed=float(changed[k].mean()),
                         paired_delta=float(delta.mean()),
                         delta_ci95=np.quantile(delta[sampled].mean(1), [.025, .975]).tolist())
    return result


@torch.inference_mode()
def main():
    DEST.mkdir(exist_ok=True)
    if (DEST/'results.json').exists():
        raise FileExistsError('Audit already saved; inspect existing results instead of rerunning.')
    start_time = time.perf_counter()
    torch.set_num_threads(4)
    frozen = source_hashes()
    output = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                  subset_records=256, view=0, permutation_replicates=32, bootstrap_replicates=2000,
                  seed=SEED, source_hashes=frozen, script_sha256=sha256_file(__file__),
                  inference_only=True, models={},
                  support='reference valid pixels; latent/token cells require >=75% reference-valid support',
                  weighting='equal records, then equal channels and equal valid positions within each record',
                  shuffle='within-record, within-scale valid-position permutation; invalid positions unchanged; exact histograms preserved')
    baseline_ids = None
    selected = None
    for size in (1024, 4096):
        root = ROOTS[size]
        assert read(root/'manifest.json')['source_hashes'] == frozen
        for m in ('hls', 'lst'):
            for training in ('clean', 'noisy'):
                run = root/f'full_{m}_conv_{training}_seed17'
                ckpt = read_checkpoint(run/'last.pt')
                c = ckpt['config']
                assert c == read(run/'config.json')
                seed_everything(c['seed'])
                device = device_for(c)
                model = CrossViewModel(c).to(device).eval()
                model.load_state_dict(ckpt['model'], strict=True)
                del ckpt
                codec = model.codecs[m]
                q = codec.quantizer
                scales = codec.scales
                key = f'{size}_{m}_{training}'
                record = dict(checkpoint_sha256=sha256_file(run/'last.pt'), evaluations={})
                for condition, coverage in [('clean', 0), ('noisy10', .1)]:
                    ds = PairedDataset(c, 'val', coverage=coverage)
                    assert len(ds) == 256
                    assert ds.fingerprint == read(run/'val_step_0010813/metrics.json')['data_fingerprint']
                    if baseline_ids is None:
                        baseline_ids = ds.record_ids.copy()
                        tiles = [ds.sources[m].get_metadata(int(p))['tile'] for p in ds.positions[m]]
                        tile_names = sorted(set(tiles), key=lambda t: stable_seed('spatial_tile', SEED, t))[:32]
                        selected = sorted(min([i for i, t in enumerate(tiles) if t == tile],
                                              key=lambda i: stable_seed('spatial_record', SEED, int(baseline_ids[i])))
                                          for tile in tile_names)
                        assert len(selected) == len({tiles[i] for i in selected}) == 32
                        output.update(record_ids=baseline_ids.tolist(), tile_ids=tiles,
                                      probe_positions=selected, illustration_positions=selected[:4],
                                      probe_selection='32 tiles ranked by seeded hash, then one hashed record per tile; independent of predictions')
                        write(DEST/'selection.json', {k: output[k] for k in ('record_ids', 'tile_ids', 'probe_positions', 'illustration_positions', 'probe_selection', 'seed')})
                    else:
                        assert np.array_equal(baseline_ids, ds.record_ids)
                    std = np.asarray(ds.sources[m].stats['std'], dtype=np.float64)[None, :, None, None]
                    mean = np.asarray(ds.sources[m].stats['mean'], dtype=np.float64)[None, :, None, None]
                    unit = 100 if m == 'lst' else 1
                    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True,
                                        generator=torch.Generator().manual_seed(SEED))
                    parts, normalized, latent = [], [], [[] for _ in scales]
                    all_ids, all_support = [], []
                    probes = {k: [] for k in ('native', 'shuffle_all', 'shuffle_finest')}
                    pictures = {k: [] for k in ('target', 'input', 'mask', 'native', 'shuffle_all', 'shuffle_finest')}
                    observed, offset = [], 0
                    for batch in loader:
                        image = batch[m]['image'][:, 0].to(device)
                        mask = batch[m]['input_mask'][:, 0].to(device)
                        reference = batch[m]['reference_mask'].to(device)
                        support = reference[:, 0].cpu().numpy().astype(bool)
                        target = batch[m]['target'].numpy()
                        physical_target = (target.astype(np.float64)*std+mean)/unit
                        x = torch.cat((torch.where(mask.bool(), image, 0), mask.to(image.dtype)), 1)
                        with amp_context(c, device):
                            z = codec.network.quant_conv(codec.network.encoder(x))
                        z = z.float()
                        ids = q.f_to_idxBl_or_fhat(z, False)
                        embeddings = [q.embedding(a).transpose(1, 2).reshape(len(a), c['model']['latent_dim'], s, s)
                                      for a, s in zip(ids, scales)]
                        fhats = q.embed_to_fhat(embeddings, all_to_max_scale=True, last_one=False)
                        with amp_context(c, device):
                            pred = codec.network.decoder(codec.network.post_quant_conv(fhats[-1])).float()
                        pred_np = pred.cpu().numpy()
                        physical_pred = (pred_np.astype(np.float64)*std+mean)/unit
                        parts.append(reconstruction_parts(physical_pred, physical_target, support))
                        normalized.append(reconstruction_parts(pred_np, target, support))
                        supports = [(F.adaptive_avg_pool2d(reference.float(), (s, s))[:, 0] >= .75).flatten(1)
                                    for s in scales]
                        support16 = supports[-1].cpu().numpy()
                        z_np = z.cpu().numpy()
                        for j, fhat in enumerate(fhats):
                            latent[j].append(reconstruction_parts(fhat.cpu().numpy(), z_np, support16))
                        all_ids.append(torch.cat(ids, 1).cpu().numpy().astype(np.int16))
                        all_support.append(torch.cat(supports, 1).cpu().numpy())
                        local_probe = [i-offset for i in selected if offset <= i < offset+len(image)]
                        local_picture = [i-offset for i in selected[:4] if offset <= i < offset+len(image)]
                        if local_probe:
                            probes['native'].append({k: v[local_probe] for k, v in parts[-1].items()})
                            for mode in ('shuffle_all', 'shuffle_finest'):
                                changed_ids = [a[local_probe].clone() for a in ids]
                                for j, (a, s) in enumerate(zip(changed_ids, scales)):
                                    if s == 1 or (mode == 'shuffle_finest' and s != 16):
                                        continue
                                    for b, i in enumerate(local_probe):
                                        valid_positions = torch.nonzero(supports[j][i], as_tuple=False).flatten()
                                        rng = np.random.default_rng(stable_seed('spatial_permutation', SEED, int(baseline_ids[offset+i]), s))
                                        permutation = torch.as_tensor(rng.permutation(len(valid_positions)), device=device)
                                        a[b, valid_positions] = ids[j][i, valid_positions[permutation]]
                                    assert torch.equal(a.sort(1).values, ids[j][local_probe].sort(1).values)
                                changed_z = codec._decode_tokens(changed_ids)
                                with amp_context(c, device):
                                    changed_pred = codec.network.decoder(codec.network.post_quant_conv(changed_z)).float().cpu().numpy()
                                probes[mode].append(reconstruction_parts((changed_pred.astype(np.float64)*std+mean)/unit,
                                                                        physical_target[local_probe], support[local_probe]))
                                for i in local_picture:
                                    pictures[mode].append(changed_pred[local_probe.index(i)])
                        for i in local_picture:
                            for name, values in [('target', target), ('input', image.float().cpu().numpy()),
                                                 ('mask', support), ('native', pred_np)]:
                                pictures[name].append(values[i])
                        observed.extend(batch['record_id'].tolist())
                        offset += len(image)
                    assert observed == baseline_ids.tolist()
                    ds.close()
                    del loader
                    p, pn = combine(parts), combine(normalized)
                    ids_np, support_np = np.concatenate(all_ids), np.concatenate(all_support)
                    token_stats, raw_counts, at = [], [], 0
                    for s in scales:
                        codes, valid = ids_np[:, at:at+s*s].reshape(-1, s, s), support_np[:, at:at+s*s].reshape(-1, s, s)
                        token_stats.append(token_summary(codes, valid, size, permutations=32 if s in (8, 16) else 0, seed=SEED))
                        raw_counts.append(np.bincount(codes.ravel(), minlength=size))
                        at += s*s
                    if condition == 'clean':
                        old_counts = np.load(root/f'report/codebook_audit/{run.name}_{condition}_counts.npz')
                        assert np.array_equal(old_counts['record_ids'], baseline_ids)
                        assert np.array_equal(np.stack(raw_counts)*2, old_counts['counts'])
                    old = {}
                    with (run/f'eval_{condition}/samples.jsonl').open(encoding='utf-8') as f:
                        for row in map(json.loads, f):
                            if row['view'] == 0 and row['region'] == 'all':
                                old[int(row['sample_id'].rsplit(':', 1)[1])] = row['mse_physical']
                    old_mse = np.array([old[int(r)] for r in baseline_ids])
                    agreement = float(np.max(np.abs(np.sqrt(p['mse'])-np.sqrt(old_mse))/np.maximum(np.sqrt(old_mse), 1e-12)))
                    assert agreement < .005, (key, condition, agreement)
                    probe_arrays = {mode: combine(ps) for mode, ps in probes.items()}
                    probe_stats = {mode: paired_change(probe_arrays['native'], probe_arrays[mode])
                                   for mode in ('shuffle_all', 'shuffle_finest')}
                    archive = dict(ids=ids_np, support=support_np, scales=np.asarray(scales), record_ids=baseline_ids,
                                   probe_positions=np.asarray(selected), illustration_positions=np.asarray(selected[:4]), mean=mean, std=std)
                    archive.update({f'physical_{k}': v for k, v in p.items()})
                    archive.update({f'normalized_{k}': v for k, v in pn.items()})
                    for mode, pa in probe_arrays.items():
                        archive.update({f'probe_{mode}_{k}': v for k, v in pa.items()})
                    for name, arrays in pictures.items():
                        archive[f'illustration_{name}'] = np.stack(arrays)
                    latent_stats = []
                    for j, ps in enumerate(latent):
                        combined = combine(ps)
                        latent_stats.append(dict(scale=int(scales[j]), **describe_parts(combined)))
                        archive.update({f'latent_{scales[j]}_{k}': v for k, v in combined.items()})
                    path = DEST/f'{key}_{condition}.npz'
                    np.savez_compressed(path, **archive)
                    record['evaluations'][condition] = dict(physical=describe_parts(p), normalized=describe_parts(pn),
                        token_scales=token_stats, latent_scales=latent_stats, spatial_shuffle_probes=probe_stats,
                        saved_full_evaluation_max_relative_rmse_difference=agreement,
                        exact_clean_counts_match_previous_audit=(condition == 'clean'),
                        arrays_file=path.name, arrays_sha256=sha256_file(path))
                    print(json.dumps(dict(model=key, condition=condition, rmse=record['evaluations'][condition]['physical']['rmse'],
                                          spatial_r2=record['evaluations'][condition]['physical']['spatial_r2'],
                                          max_relative_rmse_difference=agreement)), flush=True)
                record['memory'] = memory_stats(device)
                output['models'][key] = record
                write(DEST/'results.partial.json', output)
                del model, codec, q, ds, c, z, fhats, embeddings, pred, x, image, mask, reference
                gc.collect()
                torch.cuda.empty_cache()
    assert frozen == source_hashes()
    output.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.perf_counter()-start_time)
    write(DEST/'results.json', output)
    print('SPATIAL TOKEN AUDIT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
