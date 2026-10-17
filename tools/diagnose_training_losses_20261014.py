"""Offline diagnostics from frozen step logs and completed validation cells."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/torch_models_20261013'
RESULTS = ROOT / 'outputs/torch_experiments_20261013'
OUT = ROOT / 'outputs/loss_diagnostics_20261014'


def weighted(values, weights):
    return float(np.average(values, weights=weights))


def describe(values, weights):
    return {'mean': weighted(values, weights), 'median': float(np.median(values)),
            'p05': float(np.quantile(values, .05)), 'p95': float(np.quantile(values, .95)),
            'max': float(np.max(values)), 'min': float(np.min(values))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'curves').mkdir(exist_ok=True)
    manifest = json.loads((BUNDLE / 'MODEL_MANIFEST.json').read_text(encoding='utf-8'))
    index = json.loads((RESULTS / 'results_index.json').read_text(encoding='utf-8'))
    runs = {r['model']: r for r in index['runs']}
    summaries = []
    for m in manifest['models']:
        c = json.loads((BUNDLE / m['config']).read_text(encoding='utf-8'))
        directory = RESULTS / 'snapshot/output' / m['stage'] / 'runs' / m['name'] / 'training'
        by_step = {}
        for path in sorted(directory.glob('train_*.jsonl')):
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    if line.strip():
                        r = json.loads(line)
                        if r['step'] not in by_step or r['utc'] > by_step[r['step']]['utc']:
                            by_step[r['step']] = r
        assert sorted(by_step) == list(range(1, 17111))
        events = [by_step[s] for s in sorted(by_step)]
        keys = sorted(set().union(*(r['loss'].keys() for r in events)))
        assert all(set(r['loss']) == set(keys) for r in events)
        arrays = {k: np.array([r['loss'][k] for r in events], dtype=np.float64) for k in keys}
        assert all(np.isfinite(a).all() for a in arrays.values())
        counts = np.array([r['actual_batch'] for r in events])
        arrays['total'] = sum(v * arrays[k] for k,v in c['loss'].items() if isinstance(v, (int,float)))
        arrays['weighted_gradient'] = arrays['gradient'] * c['loss']['gradient']
        arrays['weighted_kl'] = arrays['kl'] * c['loss']['kl']
        lr = np.array([r['lr'] for r in events])
        epoch = np.array([r['records'] / c['data']['counts']['train'] for r in events])
        epochs = [{k: describe(a[i*3422:(i+1)*3422], counts[i*3422:(i+1)*3422])
                   for k,a in arrays.items()} for i in range(5)]
        blocks = []
        for start in range(0, len(events), 100):
            stop = min(start+100, len(events))
            blocks.append({'start': start+1, 'end': stop, 'epoch': float(epoch[stop-1]),
                           **{k: weighted(a[start:stop], counts[start:stop]) for k,a in arrays.items()}})
        train_stats = {}
        for k,a in arrays.items():
            tail = a[-1000:]
            train_stats[k] = {'first100': describe(a[:100], counts[:100]),
                             'last1000': describe(tail, counts[-1000:]),
                             'last100': describe(a[-100:], counts[-100:]),
                             'largest_step': int(np.argmax(a))+1}
        validation = {}
        for condition in ('clean', 'pixel10'):
            cell = next(r for r in runs[m['name']]['cells'] if r['condition'].startswith(condition+'_'))
            metrics = json.loads((RESULTS / cell['source']).read_text(encoding='utf-8'))
            validation[condition] = {k: metrics[k] for k in ('raw','projected','observation_mean_fill_baseline')}
            validation[condition]['source'] = cell['source']
        best_block = min((b for b in blocks if b['end']-b['start'] == 99), key=lambda b:b['pixel'])
        correlation = float(np.corrcoef(arrays['pixel'][-3422:], arrays['total'][-3422:])[0,1])
        summary = {'model': m['name'], 'kind': m['kind'], 'representation': m['representation'],
                   'loss_weights': c['loss'], 'residual_std_C': c['task']['residual_std_celsius'],
                   'epochs': epochs, 'windows': train_stats, 'blocks100': blocks,
                   'best_pixel_block100': best_block,
                   'last1000_vs_best_block_pixel_ratio': train_stats['pixel']['last1000']['mean']/best_block['pixel'],
                   'last_epoch_pixel_total_correlation': correlation,
                   'validation': validation,
                   'step_log_files': [str(p.relative_to(ROOT)).replace('\\','/') for p in directory.glob('train_*.jsonl')],
                   'rows': len(events), 'presentations': int(counts.sum()),
                   'lr_start': float(lr[0]), 'lr_end': float(lr[-1])}
        summaries.append(summary)
        np.savez_compressed(OUT / 'curves' / (m['name']+'.npz'), **arrays, actual_batch=counts, epoch=epoch, lr=lr)
        print(m['name'], 'pixel_epoch=',[round(e['pixel']['mean'],6) for e in epochs],
              'bneck_epoch=',[round(e['bottleneck']['mean'],6) for e in epochs], flush=True)
    report = {'scope':'39 original runs; 667290 per-update logs; no new training or inference',
              'aggregation':'sample-count-weighted means; 3422 updates/epoch, 31-record tail; 100-update nonoverlapping plots',
              'notes':['Training physical error is equal-image normalized MSE, not globally pixel-weighted validation RMSE.',
                       'All intermediate validation scores are unavailable; no overfitting claim can be proved from train loss alone.',
                       'Quantizer activity/perplexity in logs is per microbatch, not whole-dataset codebook usage.',
                       'Representation loss weight is read from original configuration; zero-weight diagnostics are excluded.'],
              'models': summaries}
    (OUT/'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
