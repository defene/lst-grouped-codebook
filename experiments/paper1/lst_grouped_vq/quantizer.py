"""Product codebooks at each VAR scale; concatenate before the original full-channel Phi."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from eo_denoise.var_ops import interpolate
from eo_denoise.var_codec import VarCodec


class GroupedQuantizer(nn.Module):
    def __init__(self, native, groups, codes):
        super().__init__()
        if groups < 1 or native.Cvae % groups or codes < 2:
            raise ValueError('Groups must divide latent channels; codes >= 2')
        if native.using_znorm:
            raise ValueError('This ablation uses squared Euclidean nearest code')
        self.groups, self.size, self.vocab_size = groups, codes, codes
        self.Cvae, self.beta, self.v_patch_nums = native.Cvae, native.beta, native.v_patch_nums
        self.quant_resi = native.quant_resi
        # Preserve native embedding exactly for the one-group equivalence test.
        self.embeddings = nn.ModuleList([native.embedding] if groups == 1 and codes == native.vocab_size else
                                       [nn.Embedding(codes, native.Cvae//groups) for _ in range(groups)])

    def embed(self, ids, pn):
        return torch.cat([e(ids[:, g].long()).transpose(1, 2).reshape(len(ids), self.Cvae//self.groups, pn, pn)
                          for g, e in enumerate(self.embeddings)], dim=1)

    def contribution(self, ids, si):
        pn, final = self.v_patch_nums[si], self.v_patch_nums[-1]
        h = self.embed(ids, pn)
        if pn != final:
            h = interpolate(h, (final, final), 'bicubic')
        return self.quant_resi[si/(len(self.v_patch_nums)-1)](h.contiguous())

    def forward(self, z):
        z = z.float()
        b, c, h, w = z.shape
        assert c == self.Cvae and h == w == self.v_patch_nums[-1]
        rest, total = z.detach().clone(), torch.zeros_like(z)
        loss, indices = z.new_zeros(()), []
        for si, pn in enumerate(self.v_patch_nums):
            with torch.no_grad():
                low = interpolate(rest, (pn, pn), 'area') if pn != h else rest
                parts = low.chunk(self.groups, dim=1)
                ids = []
                for part, embedding in zip(parts, self.embeddings):
                    flat = part.permute(0, 2, 3, 1).reshape(-1, c//self.groups)
                    distance = flat.square().sum(1, keepdim=True)+embedding.weight.square().sum(1)
                    distance.addmm_(flat, embedding.weight.T, alpha=-2, beta=1)
                    ids.append(distance.argmin(1).reshape(b, pn*pn))
                idx = torch.stack(ids, 1)
            contribution = self.contribution(idx, si)
            total = total+contribution
            rest = rest-contribution.detach()
            # Average over all C channels, NOT a sum of G independently averaged losses.
            if self.training:
                loss = loss+self.beta*F.mse_loss(total.detach(), z)+F.mse_loss(total, z.detach())
            indices.append(idx)
        loss = loss/len(self.v_patch_nums)
        output = z+(total-z).detach() if self.training else total
        return output, loss, indices

    def decode(self, indices):
        if indices.ndim != 3 or indices.shape[1:] != (self.groups, sum(p*p for p in self.v_patch_nums)):
            raise ValueError('Expected B x groups x 680 transmitted indices')
        if indices.dtype not in (torch.int32, torch.int64) or indices.min() < 0 or indices.max() >= self.size:
            raise ValueError('Invalid codebook indices')
        parts = indices.split([p*p for p in self.v_patch_nums], dim=2)
        total = None
        for si, ids in enumerate(parts):
            h = self.contribution(ids, si)
            total = h if total is None else total+h
        return total


class GroupedCodec(VarCodec):
    def __init__(self, channels, config):
        # Same native construction sequence for all arms: identical initial backbone/Phi.
        super().__init__(channels, config)
        self.network.quantize = GroupedQuantizer(self.network.quantize, config['groups'], config['codes_per_group'])

    def forward(self, image, input_mask):
        if image.shape[-2:] != (256, 256):
            raise ValueError('Frozen 256x256 images required')
        x = torch.cat((torch.where(input_mask.bool(), image, 0), input_mask.to(image.dtype)), 1)
        z = self.network.quant_conv(self._block(self.network.encoder, x))
        with torch.autocast(device_type=z.device.type, enabled=False):
            quantized, loss, ids = self.quantizer(z.float())
        pred = self._block(self.network.decoder, self.network.post_quant_conv(quantized))
        return dict(reconstruction=pred, quantized=quantized, bottleneck_loss=loss, kl=loss.new_zeros(()),
                    log_probs=None, indices=torch.cat(ids, 2),
                    indices_by_scale=[a.reshape(len(image), self.quantizer.groups, p, p) for a, p in zip(ids, self.scales)],
                    structure=self.projection(quantized))

    def decode_indices(self, indices):
        return self.network.decoder(self.network.post_quant_conv(self.quantizer.decode(indices)))

    def rate(self):
        g, k = self.quantizer.groups, self.quantizer.size
        positions = sum(p*p for p in self.scales)
        bits = positions*g*math.ceil(math.log2(k))
        return dict(kind='grouped_multiscale_fixed_length_indices', scales=list(self.scales),
                    positions_per_group=positions, groups=g, tokens=positions*g, vocabulary=k,
                    bits_per_image=bits, bpp_spatial=bits/65536, bpp_band_pixel=bits/(65536*self.channels),
                    embedding_parameters=g*k*(self.config['latent_dim']//g), includes_mask_header_or_model=False)
