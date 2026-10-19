"""Sample-equal physical metrics, explicit regions, and tile-cluster intervals."""
from collections import defaultdict
import math
import numpy as np
import torch
import torch.nn.functional as F


class EvaluationMetrics:
    def __init__(self, sources, model):
        self.rows, self.stability = [], []
        self.stats = {m: ds.stats for m, ds in sources.items()}
        self.histograms = {m: np.zeros(codec.quantizer.size, np.int64) for m, codec in model.codecs.items()
                           if codec.config['quantizer'] in ('vq', 'fsq','var_vq')}

    def add(self, modality, outputs, data, sample_ids, tiles):
        target = data['target'].float()
        stats = self.stats[modality]
        std = torch.tensor(stats['std'], device=target.device)[:, None, None]
        mean = torch.tensor(stats['mean'], device=target.device)[:, None, None]
        physical_scale = std/100 if modality == 'lst' else std
        reference = data['reference_mask'].bool()
        for realization, output in enumerate(outputs):
            pred = output['reconstruction'].float()
            error = pred-target
            physical_error = error*physical_scale
            regions = {'all': reference,
                       'corrupt': reference & data['corruption_mask'][:, realization].bool(),
                       'retained': reference & ~data['corruption_mask'][:, realization].bool()}
            for region, masks in regions.items():
                for i, mask in enumerate(masks[:, 0]):
                    if not mask.any():
                        continue
                    e, ep = error[i, :, mask], physical_error[i, :, mask]
                    row = {'modality': modality, 'sample_id': sample_ids[i], 'tile_id': tiles[i],
                           'region': region, 'view': realization, 'pixels': int(mask.sum()),
                           'mse_normalized': float(e.square().mean()),
                           'mse_physical': float(ep.square().mean()),
                           'mae_physical': float(ep.abs().mean()), 'bias_physical': float(ep.mean()),
                           'band_mse_physical': ep.square().mean(-1).cpu().tolist(),
                           'band_mae_physical': ep.abs().mean(-1).cpu().tolist(),
                           'actual_coverage': float(data['actual_coverage'][i, realization]),
                           'gradient_mse_physical': None}
                    dx = physical_error[i, :, :, 1:]-physical_error[i, :, :, :-1]
                    dy = physical_error[i, :, 1:, :]-physical_error[i, :, :-1, :]
                    mx, my = mask[:, 1:] & mask[:, :-1], mask[1:, :] & mask[:-1, :]
                    differences = [v[:, m].square().mean() for v, m in ((dx, mx), (dy, my)) if m.any()]
                    if differences:
                        row['gradient_mse_physical'] = float(torch.stack(differences).mean())
                    if modality == 'hls':
                        p = (pred[i]*std+mean)[:, mask]
                        t = (target[i]*std+mean)[:, mask]
                        denom = p.norm(dim=0)*t.norm(dim=0)
                        good = denom > 1e-8
                        row['sam_degrees'] = float(torch.acos(((p*t).sum(0)[good]/denom[good]).clamp(-1, 1)).mean()*180/math.pi) if good.any() else None
                        pd, td = p[3]+p[2], t[3]+t[2]
                        good = (pd.abs() > 1e-6) & (td.abs() > 1e-6)
                        row['ndvi_mae'] = float(((p[3]-p[2])[good]/pd[good]-(t[3]-t[2])[good]/td[good]).abs().mean()) if good.any() else None
                    self.rows.append(row)
            if modality in self.histograms:
                ids = output['indices'].detach().cpu().numpy()
                self.histograms[modality] += np.bincount(ids.ravel(), minlength=len(self.histograms[modality]))
        if modality in self.histograms and len(outputs) >= 2:
            a, b = outputs[0]['indices'], outputs[1]['indices']
            supports = {'reference': reference,
                        'common_visible': data['input_mask'][:, 0].bool() & data['input_mask'][:, 1].bool()}
            for region, mask in supports.items():
                if 'indices_by_scale' in outputs[0]:
                    grids=outputs[0]['indices_by_scale']
                    support=torch.cat([(F.adaptive_avg_pool2d(mask.float(),g.shape[-2:])[:,0]>=.75).flatten(1)
                                       for g in grids],1)[:,None]
                else:
                    support = F.adaptive_avg_pool2d(mask.float(), a.shape[-2:])[:, 0] >= .75
                for i in range(len(a)):
                    if support[i].any():
                        self.stability.append({'modality': modality, 'region': region, 'sample_id': sample_ids[i],
                            'tile_id': tiles[i], 'flip_rate': float((a[i][support[i]] != b[i][support[i]]).float().mean())})

    def summary(self, bootstrap=500, seed=20260910):
        groups = defaultdict(lambda: defaultdict(list))
        for row in self.rows:
            groups[(row['modality'], row['region'])][row['sample_id']].append(row)
        result = {'weighting': 'equal_samples; average available views within each sample',
                  'regions': {}, 'tokens': {}, 'stability': {},
                  'uncertainty': 'tile-cluster percentile bootstrap over sample-equal MSE, then square root'}
        rng = np.random.default_rng(seed)
        for (modality, region), samples in groups.items():
            rows = []
            fields = ['mse_normalized', 'mse_physical', 'mae_physical', 'bias_physical',
                      'gradient_mse_physical', 'sam_degrees', 'ndvi_mae', 'actual_coverage']
            for sample, entries in samples.items():
                row = {'sample_id': sample, 'tile_id': entries[0]['tile_id']}
                for field in fields:
                    values = [v[field] for v in entries if v.get(field) is not None]
                    row[field] = float(np.mean(values)) if values else None
                row['band_mse_physical'] = np.mean([v['band_mse_physical'] for v in entries], axis=0)
                row['band_mae_physical'] = np.mean([v['band_mae_physical'] for v in entries], axis=0)
                rows.append(row)
            entry = {'samples': len(rows), 'tiles': len({r['tile_id'] for r in rows}),
                     'unit': 'degrees_Celsius' if modality == 'lst' else 'reflectance',
                     'rmse_normalized': float(np.sqrt(np.mean([r['mse_normalized'] for r in rows]))),
                     'rmse': float(np.sqrt(np.mean([r['mse_physical'] for r in rows]))),
                     'mae': float(np.mean([r['mae_physical'] for r in rows])),
                     'bias': float(np.mean([r['bias_physical'] for r in rows])),
                     'band_rmse': np.sqrt(np.mean([r['band_mse_physical'] for r in rows], axis=0)).tolist(),
                     'band_mae': np.mean([r['band_mae_physical'] for r in rows], axis=0).tolist(),
                     'actual_coverage': float(np.mean([r['actual_coverage'] for r in rows]))}
            for field in ('gradient_mse_physical', 'sam_degrees', 'ndvi_mae'):
                values = [r[field] for r in rows if r[field] is not None]
                entry[field] = float(np.mean(values)) if values else None
                entry[field+'_samples'] = len(values)
            entry['gradient_rmse'] = math.sqrt(entry.pop('gradient_mse_physical')) if entry['gradient_mse_physical'] is not None else None
            by_tile = defaultdict(list)
            for r in rows:
                by_tile[r['tile_id']].append(r['mse_physical'])
            if len(by_tile) >= 2 and bootstrap:
                sums = np.array([sum(v) for v in by_tile.values()])
                counts = np.array([len(v) for v in by_tile.values()])
                values = []
                for _ in range(bootstrap):
                    draw = rng.integers(len(sums), size=len(sums))
                    values.append(np.sqrt(sums[draw].sum()/counts[draw].sum()))
                entry['rmse_tile_ci95'] = np.quantile(values, [.025, .975]).tolist()
            else:
                entry['rmse_tile_ci95'] = None
            result['regions'][f'{modality}/{region}'] = entry
        for m, counts in self.histograms.items():
            p = counts[counts > 0]/max(1, counts.sum())
            entropy = float(-(p*np.log2(p)).sum())
            result['tokens'][m] = {'usage_fraction': float(np.mean(counts > 0)), 'used_codes': int((counts > 0).sum()),
                'vocabulary': len(counts), 'empirical_entropy_bits': entropy, 'perplexity': 2**entropy,
                'count_scope': 'all transmitted grid positions across evaluation views',
                'entropy_is_not_an_actual_compressed_file_rate': True}
        for m in self.stats:
            for region in ('reference', 'common_visible'):
                values = [r['flip_rate'] for r in self.stability if r['modality'] == m and r['region'] == region]
                if values:
                    result['stability'][f'{m}/{region}'] = {'flip_rate': float(np.mean(values)), 'samples': len(values)}
        scores = []
        for m in self.stats:
            hidden, retained = result['regions'].get(f'{m}/corrupt'), result['regions'].get(f'{m}/retained')
            if hidden and retained:
                scores.append(.5*(hidden['rmse_normalized']+retained['rmse_normalized']))
            else:
                scores.append(result['regions'][f'{m}/all']['rmse_normalized'])
        result['selection_score'] = float(np.mean(scores))
        result['selection_definition'] = 'equal modalities; .5 corrupt nRMSE + .5 retained nRMSE; clean falls back to all-reference nRMSE'
        return result
