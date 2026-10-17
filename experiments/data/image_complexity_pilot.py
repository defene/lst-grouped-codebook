"""Reproducible image complexity pilot; reads frozen data without modifying it."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eo_data import open_dataset

OUT = ROOT / 'data_inspection/image_complexity_pilot_20260912'
SEED = 20260912
N = 256
BANDS = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

def gradient(x, mask):
    """Mean absolute adjacent difference over valid horizontal and vertical edges."""
    edges = []
    for axis in (0, 1):
        a, b = (mask[:-1], mask[1:]) if axis == 0 else (mask[:, :-1], mask[:, 1:])
        edges.append(np.abs(np.diff(x, axis=axis))[a & b])
    values = np.concatenate(edges)
    return (float(values.mean()) if len(values) else float('nan')), int(len(values))

def coarse(x, mask, scale):
    h, w = x.shape
    # Only fully valid blocks qualify, so masked fill cannot create gradients.
    valid = mask.reshape(h//scale, scale, w//scale, scale).all(axis=(1, 3))
    y = x.reshape(h//scale, scale, w//scale, scale).mean(axis=(1, 3))
    return y, valid

def measure_band(x, mask, width):
    v = x[mask].astype(np.float64)
    q25, q75 = np.quantile(v, [.25, .75])
    # Fixed-width bins anchored at zero, unbounded range; no per-image fitting.
    _, count = np.unique(np.floor(v / width).astype(np.int64), return_counts=True)
    p = count / count.sum()
    result = {'iqr': float(q75-q25), 'std': float(v.std()),
              'entropy_bits': float(-(p*np.log2(p)).sum())}
    for scale in (1, 4, 16):
        a, m = (x, mask) if scale == 1 else coarse(x, mask, scale)
        result[f'gradient_s{scale}'], result[f'edges_s{scale}'] = gradient(a, m)
    return result

def self_check():
    mask = np.ones((32, 32), bool)
    c = measure_band(np.zeros((32, 32)), mask, 1)
    assert c['iqr'] == c['entropy_bits'] == c['gradient_s1'] == 0
    ramp = np.tile(np.arange(32.), (32, 1))
    assert gradient(ramp, mask)[0] == .5
    mask[:, 16:] = False
    ramp[:, 16:] = 1e9
    assert gradient(ramp, mask)[0] < 1
    checker = np.indices((32,32)).sum(0) % 2
    r = measure_band(checker.astype(float), np.ones_like(mask), 1)
    assert r['gradient_s1'] == 1 and r['gradient_s4'] == 0

def physical(ds, index):
    sample = ds[int(index)]
    x = ds.denormalize(sample['image']).astype(np.float64)
    if ds.modality == 'lst_hr':
        # Stored original integers recovered after z-score roundtrip.
        x = np.rint(x) / 100
    else:
        x = np.rint(x * 10000) / 10000
    return sample, x, sample['reference_mask'][0]

def clean_json(obj):
    if isinstance(obj, dict): return {k: clean_json(v) for k,v in obj.items()}
    if isinstance(obj, list): return [clean_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj): return None
    return obj

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    self_check()
    rows, bands, groups, fingerprints, representatives = [], [], {}, {}, []
    for di, dataset in enumerate(('prithvi', 'lstsr_tb')):
        for mi, modality in enumerate(('lst', 'hls')):
            name = dataset + '/' + modality
            ds = open_dataset(dataset, modality, 'train')
            fingerprints[name] = ds.fingerprint
            rng = np.random.default_rng(SEED + 100*di + mi)
            ids = np.sort(rng.choice(len(ds), N, replace=False))
            group = []
            for step, index in enumerate(ids):
                sample, x, mask = physical(ds, index)
                per_band = [measure_band(a, mask, 1 if modality == 'lst' else .01) for a in x]
                row = {'group': name, 'index': int(index), 'sample_id': sample['sample_id'],
                       'tile_id': sample['tile_id'], 'month': sample['month'],
                       'valid_fraction': float(mask.mean())}
                for k in per_band[0]:
                    vals = np.array([b[k] for b in per_band])
                    row[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
                rows.append(row); group.append(row)
                for ci, values in enumerate(per_band):
                    bands.append(dict(group=name, index=int(index), sample_id=sample['sample_id'],
                                      band='LST' if modality == 'lst' else BANDS[ci], **values))
                if (step+1) % 64 == 0:
                    print(name, step+1, '/', N, flush=True)
            groups[name] = {}
            for k in ('iqr','std','entropy_bits','gradient_s1','gradient_s4','gradient_s16','valid_fraction'):
                vals = np.array([r[k] for r in group]); vals = vals[np.isfinite(vals)]
                groups[name][k] = dict(zip(('p10','p50','p90'), map(float,np.quantile(vals,[.1,.5,.9]))))
                groups[name][k]['n'] = len(vals)
            groups[name]['unique_tiles'] = len(set(r['tile_id'] for r in group))
            # Gallery is explicitly gradient-ranked, not a generic complexity score.
            candidates = [r for r in group if r['valid_fraction'] >= .9]
            candidates.sort(key=lambda r:r['gradient_s1'])
            for percentile in (.1,.5,.9):
                r = candidates[round((len(candidates)-1)*percentile)]
                _, x, mask = physical(ds, r['index'])
                filename = name.replace('/','_') + f'_p{int(percentile*100)}.npz'
                np.savez_compressed(OUT/filename, image=x.astype(np.float32), mask=mask)
                representatives.append(dict(r, rank_percentile=int(percentile*100),
                                            selection_population=len(candidates), array=filename))
            ds.close()
    for filename, data in [('samples.csv', rows), ('bands.csv', bands)]:
        with (OUT/filename).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    summary = {'seed': SEED, 'n_per_group': N, 'split': 'train',
               'sampling': 'Uniform without replacement within each frozen sample index; LSTSR LST is patch-month weighted.',
               'hls_aggregation': 'Arithmetic mean of six per-band metrics; bands.csv preserves each band.',
               'entropy_bin_width': {'lst_degrees_c': 1, 'hls_reflectance': .01},
               'gradient': 'Mean absolute horizontal/vertical adjacent difference, valid endpoints only. Units per pixel on each scale, not divided by scale.',
               'coarse_grids': 'Nonoverlapping block means at scales 4 and 16; only fully valid blocks.',
               'gallery': 'Nearest P10/P50/P90 gradient among sampled images with >=90% valid pixels; not universal complexity classes.',
               'groups': groups, 'fingerprints': fingerprints, 'representatives': representatives,
               'sanity_checks': 'Passed constant, ramp, masked boundary, and checkerboard checks.'}
    (OUT/'summary.json').write_text(json.dumps(clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    render(summary)
    print(json.dumps(groups, indent=2), flush=True)

def render(summary):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    for modality in ('lst', 'hls'):
        reps = [r for r in summary['representatives'] if r['group'].endswith('/'+modality)]
        fig, axes = plt.subplots(2, 3, figsize=(11.6, 9.4), layout='constrained')
        fig.get_layout_engine().set(h_pad=.12, w_pad=.06, hspace=.08)
        for ax, r in zip(axes.flat, reps):
            with np.load(OUT/r['array']) as z: x, mask = z['image'], z['mask']
            if modality == 'lst':
                med = np.median(x[0][mask])
                display = np.ma.array(x[0]-med, mask=~mask)
                im = ax.imshow(display, cmap='RdBu_r', vmin=-8, vmax=8, interpolation='nearest')
            else:
                display = np.clip(x[[2,1,0]].transpose(1,2,0)/.4,0,1)**(1/2.2)
                display[~mask] = [.65,.65,.65]
                ax.imshow(display, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            dataset = 'Prithvi' if r['group'].startswith('prithvi') else 'LSTSR-TB'
            ax.set_title(f"{dataset} | gradient P{r['rank_percentile']}\nIQR {r['iqr']:.3f}  |  gradient {r['gradient_s1']:.4f}", fontsize=11)
            ax.set_xlabel(f"{r['tile_id']} | index {r['index']} | valid {r['valid_fraction']:.1%}",fontsize=9)
        if modality == 'lst':
            fig.colorbar(im, ax=list(axes.flat), shrink=.82, label='Temperature minus image median (degrees C); fixed -8 to +8')
            title = 'LST spatial detail: low, median and high gradient examples'
        else:
            title = 'HLS spatial detail: six-band mean gradient P10 / P50 / P90'
        fig.suptitle(title + '\n256 training images per dataset; gallery uses >=90% valid pixels',fontsize=12)
        fig.savefig(OUT/f'{modality}_examples.png',dpi=155)
        plt.close(fig)

if __name__ == '__main__': main()
