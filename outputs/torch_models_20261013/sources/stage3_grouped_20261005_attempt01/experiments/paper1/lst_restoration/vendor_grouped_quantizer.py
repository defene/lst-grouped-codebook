"""Product codebooks at each VAR scale; concatenate before the original full-channel Phi."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from eo_denoise.var_ops import interpolate


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
