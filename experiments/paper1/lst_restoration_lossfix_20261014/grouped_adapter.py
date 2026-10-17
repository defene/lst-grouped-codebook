"""Pinned g8k256 quantizer adapted to the eight-representation LST protocol."""
import copy
import math
import numpy as np
import torch
from torch import nn

from .models import ConvolutionalRestorationCodec
from .vendor_grouped_quantizer import GroupedQuantizer
from eo_denoise.var_quant import VectorQuantizer2


def learning_rate(config, step):
    warmup = config['train'].get('warmup_updates', 0)
    if step < warmup:
        return config['train']['lr'] * (step + 1) / max(1, warmup)
    progress = min(1., (step - warmup) / max(1, config['train']['updates'] - warmup))
    return config['train']['lr'] * .5 * (1 + math.cos(math.pi * progress))


class GroupedProductQuantizer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scales = list(c['scales'])
        self.groups, self.codebook_size = c['groups'], c['codebook_size']
        native = VectorQuantizer2(vocab_size=1024, Cvae=c['latent_dim'], using_znorm=False,
            beta=c['commitment'], v_patch_nums=tuple(self.scales), quant_resi=c['quant_resi'],
            share_quant_resi=c['share_quant_resi'])
        self.grouped = GroupedQuantizer(native, self.groups, self.codebook_size)

    def forward(self, latent):
        with torch.autocast(device_type=latent.device.type, enabled=False):
            quantized, loss, indices = self.grouped(latent.float())
            ids = torch.cat(indices, 2)
            diagnostics = {'quantization_mse': (latent.float() - quantized).square().mean().detach()}
            for g in range(self.groups):
                counts = torch.bincount(ids[:,g].flatten(), minlength=self.codebook_size).float()
                p = counts / counts.sum()
                diagnostics[f'group{g}/active_codes'] = (counts > 0).sum().float()
                diagnostics[f'group{g}/perplexity'] = torch.exp(-(p * p.clamp_min(1e-20).log()).sum())
        return dict(quantized=quantized, indices=ids, loss=loss, diagnostics=diagnostics)

    def decode_indices(self, indices):
        with torch.autocast(device_type=indices.device.type, enabled=False):
            return self.grouped.decode(indices)


class GroupedRestorationCodec(ConvolutionalRestorationCodec):
    def rate(self):
        q = self.quantizer
        positions = sum(s*s for s in q.scales)
        tokens = positions * q.groups
        bits = math.ceil(math.log2(q.codebook_size))
        return dict(latent_grid=[16,16], latent_channels=self.latent_dim,
            indices_per_image=tokens, groups=q.groups, positions_per_group=positions, scales=q.scales,
            codebook_size=q.codebook_size, codebook_size_scope='per_group',
            fixed_bits_per_index=bits, nominal_index_bits=tokens*bits,
            mean_side_information_bits=32, nominal_index_plus_mean_bits=tokens*bits+32,
            embedding_parameters=q.groups*q.codebook_size*(self.latent_dim//q.groups),
            payload_type='grouped_multiscale_fixed_width_nominal', actual_bitstream_measured=False,
            excludes=['mask/support metadata','container headers','shared weights'], equal_rate_claim=False)


def grouped_statistics(indices, scales, size):
    """Count each independent dictionary separately, never pool group IDs."""
    values = np.asarray(indices)
    if values.ndim != 3 or values.shape[2] != sum(s*s for s in scales):
        raise ValueError('Expected record-view x groups x multiscale positions')
    if values.size == 0 or values.min() < 0 or values.max() >= size:
        raise ValueError('Invalid grouped index')
    groups = []
    for g in range(values.shape[1]):
        per_scale, start = [], 0
        overall = np.zeros(size, dtype=np.int64)
        for s in scales:
            block = values[:,g,start:start+s*s]
            counts = np.bincount(block.flatten().astype(np.int64), minlength=size)
            overall += counts
            probability = counts / counts.sum()
            nonzero = probability > 0
            entropy = float(-(probability[nonzero] * np.log2(probability[nonzero])).sum())
            within_unique, within_entropy, neighbor_equal, neighbor_total = 0., 0., 0, 0
            for begin in range(0, len(block), 256):
                part = block[begin:begin+256].astype(np.int64)
                offsets = np.arange(len(part))[:,None] * size
                hist = np.bincount((part+offsets).ravel(), minlength=len(part)*size).reshape(len(part), size)
                p = hist / (s*s)
                within_unique += float((hist>0).sum())
                within_entropy += float(-(p*np.log2(np.maximum(p,1e-30))).sum())
                grid = part.reshape(-1,s,s)
                if s > 1:
                    neighbor_equal += int((grid[:,1:,:]==grid[:,:-1,:]).sum()+(grid[:,:,1:]==grid[:,:,:-1]).sum())
                    neighbor_total += len(grid)*2*s*(s-1)
            per_scale.append(dict(scale=s, token_count=int(counts.sum()), used_codes=int(nonzero.sum()),
                entropy_bits=entropy, perplexity=2**entropy, index_counts=counts.tolist(),
                mean_within_image_unique_codes=within_unique/len(block),
                mean_within_image_entropy_bits=within_entropy/len(block),
                neighbor_equal_fraction=neighbor_equal/neighbor_total if neighbor_total else None))
            start += s*s
        p = overall / overall.sum()
        entropy = float(-(p[p>0]*np.log2(p[p>0])).sum())
        groups.append(dict(group=g, codebook_size=size, token_count=int(overall.sum()),
            used_codes=int((overall>0).sum()), entropy_bits=entropy, perplexity=2**entropy,
            top1_fraction=float(p.max()), top10_fraction=float(np.sort(p)[-10:].sum()),
            index_counts=overall.tolist(), scales=per_scale))
    return dict(layout='record_view,group,concatenated_scales', per_group=groups,
        token_count=int(values.size), pooled_across_groups=False,
        scope='All transmitted positions, including reference-invalid positions; entropy is not actual bitrate.')


def mechanism_checks():
    """Compare deployed adapter to the pinned original core, including gradients."""
    c=dict(scales=[1,2,3,4,5,6,8,10,13,16],groups=8,codebook_size=256,
           latent_dim=32,commitment=.25,quant_resi=.5,share_quant_resi=4)
    torch.manual_seed(17)
    adapter=GroupedProductQuantizer(c)
    reference=copy.deepcopy(adapter.grouped)
    x=torch.randn(2,32,16,16,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    a=adapter(x)
    b,loss,ids=reference(y)
    torch.testing.assert_close(a['quantized'],b,rtol=0,atol=0)
    torch.testing.assert_close(a['loss'],loss,rtol=0,atol=0)
    assert torch.equal(a['indices'],torch.cat(ids,2))
    (a['quantized'].square().mean()+a['loss']).backward()
    (b.square().mean()+loss).backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
    for p,q in zip(adapter.grouped.parameters(),reference.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    adapter.eval()
    with torch.no_grad():
        a=adapter(x)
        assert torch.equal(adapter.decode_indices(a['indices']),a['quantized'])
    # Same numeric label in different groups must not be one dictionary.
    fake=np.zeros((2,8,680),dtype=np.int64)
    fake[1,0,:]=1
    stats=grouped_statistics(fake,c['scales'],256)
    assert stats['per_group'][0]['used_codes']==2
    assert all(g['used_codes']==1 for g in stats['per_group'][1:])
    assert stats['token_count']==2*8*680
    return dict(pinned_quantizer_values_losses_indices_gradients_exact=True,
                group_statistics_separate=True, token_count_and_rate_correct=True,
                evaluation_index_only_latent_exact=True)
