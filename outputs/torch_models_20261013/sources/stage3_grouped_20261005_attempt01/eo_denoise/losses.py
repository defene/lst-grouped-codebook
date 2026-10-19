import torch
import torch.nn.functional as F
from .ops import pool_grid


def masked_mean(values, mask):
    mask = mask.to(values.dtype).expand_as(values)
    counts = mask.flatten(1).sum(1)
    sums = (values*mask).flatten(1).sum(1)
    valid = counts > 0
    per_sample = sums/counts.clamp_min(1)
    return (per_sample*valid).sum()/valid.sum().clamp_min(1)


def gradient_loss(pred, target, mask):
    dx = (pred[..., 1:]-pred[..., :-1])-(target[..., 1:]-target[..., :-1])
    dy = (pred[..., 1:, :]-pred[..., :-1, :])-(target[..., 1:, :]-target[..., :-1, :])
    return .5*(masked_mean(dx.square(), mask[..., 1:] & mask[..., :-1])+
               masked_mean(dy.square(), mask[..., 1:, :] & mask[..., :-1, :]))


def token_support(mask, size, threshold):
    return pool_grid(mask.float(), size) >= threshold


def stability_loss(a, b, common_input, threshold):
    support = token_support(common_input, a['quantized'].shape[-2:], threshold)
    if a['log_probs'] is not None:
        p, q = a['log_probs'].float(), b['log_probs'].float()
        middle = torch.logaddexp(p, q)-torch.log(torch.tensor(2., device=p.device))
        js = .5*(p.exp()*(p-middle)+q.exp()*(q-middle)).sum(1, keepdim=True)
        return masked_mean(js, support)
    return masked_mean((a['quantized'].float()-b['quantized'].float()).square(), support)


def structure_loss(a, b, reference_a, reference_b, eligible, grid, threshold):
    common = reference_a & reference_b
    valid_cells = token_support(common, (grid, grid), threshold).flatten(1)
    def relations(output):
        feature = output['structure'].float()
        support = pool_grid(common.float(), feature.shape[-2:])
        numerator = pool_grid(feature*support, (grid, grid))
        denominator = pool_grid(support, (grid, grid)).clamp_min(1e-6)
        vectors = F.normalize((numerator/denominator).flatten(2), dim=1, eps=1e-6)
        return vectors.transpose(1, 2)@vectors
    relation_mask = valid_cells[:, :, None] & valid_cells[:, None, :]
    eye = torch.eye(grid*grid, device=common.device, dtype=torch.bool)
    relation_mask &= ~eye[None]
    relation_mask &= eligible[:, None, None].bool()
    return masked_mean((relations(a)-relations(b)).square(), relation_mask)


def objective(outputs, batch, config):
    c = config['loss']
    sample = next(iter(outputs.values()))[0]['reconstruction']
    terms = {k: sample.float().new_zeros(()) for k in ('reconstruction', 'gradient', 'bottleneck', 'kl', 'stability', 'cross_view')}
    for modality, views in outputs.items():
        target = batch[modality]['target'].float()
        reference = batch[modality]['reference_mask'].bool()
        for view in views:
            pred = view['reconstruction'].float()
            terms['reconstruction'] += masked_mean((pred-target).square(), reference)/len(views)/len(outputs)
            terms['gradient'] += gradient_loss(pred, target, reference)/len(views)/len(outputs)
            terms['bottleneck'] += view['bottleneck_loss']/len(views)/len(outputs)
            terms['kl'] += view['kl']/len(views)/len(outputs)
        if len(views) >= 2:
            common = batch[modality]['input_mask'][:, 0].bool() & batch[modality]['input_mask'][:, 1].bool()
            terms['stability'] += stability_loss(views[0], views[1], common, c['min_token_valid'])/len(outputs)
    if len(outputs) == 2:
        for v in range(len(outputs['hls'])):
            terms['cross_view'] += structure_loss(outputs['hls'][v], outputs['lst'][v],
                batch['hls']['reference_mask'].bool(), batch['lst']['reference_mask'].bool(),
                batch['cross_view_eligible'], c['relation_grid'], c['min_token_valid'])/len(outputs['hls'])
    total = terms['reconstruction']+c['gradient']*terms['gradient']+terms['bottleneck']+c['kl']*terms['kl']
    total = total+c['stability']*terms['stability']+c['cross_view']*terms['cross_view']
    return total, terms
