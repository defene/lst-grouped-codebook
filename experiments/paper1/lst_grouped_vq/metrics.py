"""Same pixel metrics, plus explicit per-group/per-scale code usage and within-image diversity."""
import numpy as np
import torch
import torch.nn.functional as F
from eo_denoise.metrics import EvaluationMetrics as OriginalMetrics


def distribution(counts):
    positive = counts[counts > 0]
    p = positive/max(1, positive.sum())
    entropy = float(-(p*np.log2(p)).sum())
    return dict(used_codes=len(positive), vocabulary=len(counts), count=int(counts.sum()),
                entropy_bits=entropy, perplexity=2**entropy,
                top1_fraction=float(counts.max()/max(1, counts.sum())),
                top10_fraction=float(np.sort(counts)[-10:].sum()/max(1, counts.sum())))


class EvaluationMetrics(OriginalMetrics):
    def __init__(self, sources, model):
        super().__init__(sources, model)
        self.histograms = {}  # Never merge code IDs from different group-specific dictionaries.
        c = model.codecs['lst'].config
        self.groups, self.codes, self.scales = c['groups'], c['codes_per_group'], c['var_scales']
        self.counts = np.zeros((self.groups, len(self.scales), self.codes), np.int64)
        self.within = np.zeros((self.groups, len(self.scales), 4), np.float64)
        self.stable = {key:np.zeros((self.groups, 2), np.int64) for key in ('reference','common_visible')}

    def add(self, modality, outputs, data, sample_ids, tiles):
        super().add(modality, outputs, data, sample_ids, tiles)
        for out in outputs:
            for si, grid in enumerate(out['indices_by_scale']):
                ids = grid.detach().cpu().numpy()
                if ids.ndim == 3:
                    ids = ids[:, None]
                for g in range(self.groups):
                    self.counts[g, si] += np.bincount(ids[:, g].ravel(), minlength=self.codes)
                    for sample in ids[:, g]:
                        hist = np.bincount(sample.ravel(), minlength=self.codes)
                        d = distribution(hist)
                        if sample.shape[0] > 1:
                            adjacent = .5*((sample[:,1:]!=sample[:,:-1]).mean()+(sample[1:]!=sample[:-1]).mean())
                        else:
                            adjacent = 0.
                        self.within[g, si] += [d['used_codes'], d['entropy_bits'], adjacent, 1]
        if len(outputs) == 2:
            supports = {'reference':data['reference_mask'].bool(),
                        'common_visible':data['input_mask'][:,0].bool() & data['input_mask'][:,1].bool()}
            for si, pn in enumerate(self.scales):
                a, b = (v['indices_by_scale'][si] for v in outputs)
                if a.ndim == 3:
                    a, b = a[:,None], b[:,None]
                for region, mask in supports.items():
                    support = (F.adaptive_avg_pool2d(mask.float(),(pn,pn))[:,0] >= .75)
                    for g in range(self.groups):
                        self.stable[region][g] += [int(((a[:,g]!=b[:,g]) & support).sum()), int(support.sum())]

    def summary(self, bootstrap=500, seed=20260910):
        result = super().summary(bootstrap, seed)
        groups = []
        for g in range(self.groups):
            per_scale = []
            for si, pn in enumerate(self.scales):
                v = self.within[g,si]
                per_scale.append(dict(scale=pn, **distribution(self.counts[g,si]),
                    within_image_mean_unique_codes=float(v[0]/max(1,v[3])),
                    within_image_mean_entropy_bits=float(v[1]/max(1,v[3])),
                    within_image_adjacent_change_fraction=float(v[2]/max(1,v[3])) if pn>1 else None,
                    image_view_count=int(v[3])))
            groups.append(dict(group=g, **distribution(self.counts[g].sum(0)), scales=per_scale))
        result['tokens'] = {'lst':dict(groups=self.groups, codes_per_group=self.codes, per_group=groups,
            count_scope='all transmitted positions, including positions outside reference support; distinct dictionaries never pooled',
            within_image_scope='separate group and scale; spatial variation is not evidence of accurate detail',
            entropy_is_not_an_actual_compressed_file_rate=True)}
        result['stability'] = {region:[dict(group=g, flip_rate=float(v[0]/v[1]) if v[1] else None,
                                           compared_positions=int(v[1])) for g,v in enumerate(values)]
                               for region, values in self.stable.items()}
        return result
