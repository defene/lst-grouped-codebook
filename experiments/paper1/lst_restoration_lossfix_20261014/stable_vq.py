"""Explicit new scratch VQ recipe; never applied to pretrained VQGAN weights."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .quantizers import VectorQuantizer, nearest_indices, straight_through, _validate_indices


class SphericalVectorQuantizer(VectorQuantizer):
    """Unit-RMS tokens/codes, train-only first-batch initialization and exact codec.

    There is no per-image physical normalization or validation adaptation.
    The initialization flag and cumulative usage are checkpointed buffers.
    No automatic dead-code replacement changes a code's meaning mid-step.
    """
    def __init__(self, dim, size, commitment=.25, chunk_size=1024):
        super().__init__(dim, size, commitment, chunk_size)
        nn.init.normal_(self.embedding.weight)
        self.register_buffer('initialized_from_train', torch.tensor(False))
        self.register_buffer('usage_counts', torch.zeros(size, dtype=torch.long))

    def normalize(self, values):
        return F.normalize(values.float(), dim=-1, eps=1e-6) * math.sqrt(self.dim)

    @torch.no_grad()
    def initialize_from_train(self, flat):
        # Deterministic evenly spaced training tokens; repeats get small noise
        # only when the first microbatch contains fewer tokens than codes.
        count = len(flat)
        take = torch.linspace(0, count-1, self.codebook_size, device=flat.device).long()
        seeds = flat[take].clone()
        if count < self.codebook_size:
            seeds.add_(.01 * torch.randn_like(seeds))
        zero = seeds.norm(dim=-1) < 1e-6
        if zero.any():
            seeds[zero] = self.embedding.weight.detach()[zero]
        self.embedding.weight.copy_(self.normalize(seeds))
        self.initialized_from_train.fill_(True)

    def decode_indices(self, indices):
        indices = _validate_indices(indices, self.codebook_size)
        if indices.ndim != 3:
            raise ValueError('Single-scale indices must be [B,H,W]')
        with torch.autocast(device_type=indices.device.type, enabled=False):
            codes = F.embedding(indices, self.normalize(self.embedding.weight))
            return codes.permute(0, 3, 1, 2).contiguous()

    def forward(self, latent):
        b, c, h, w = latent.shape
        if c != self.dim or not latent.numel():
            raise ValueError('Invalid VQ latent shape')
        with torch.autocast(device_type=latent.device.type, enabled=False):
            raw = latent.float().permute(0, 2, 3, 1).reshape(-1, c)
            flat = self.normalize(raw)
            if self.training and not bool(self.initialized_from_train):
                self.initialize_from_train(flat.detach())
            x = flat.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            indices = nearest_indices(flat, self.normalize(self.embedding.weight), self.chunk_size).reshape(b, h, w)
            codes = self.decode_indices(indices)
            codebook = F.mse_loss(codes, x.detach())
            commitment = F.mse_loss(x, codes.detach())
            counts = torch.bincount(indices.flatten(), minlength=self.codebook_size)
            if self.training:
                self.usage_counts.add_(counts.detach())
            p = counts.float() / counts.sum()
            diagnostics = dict(codebook_loss=codebook, commitment_loss=commitment,
                active_codes=(counts > 0).sum(), cumulative_active_codes=(self.usage_counts > 0).sum(),
                perplexity=torch.exp(-(p * p.clamp_min(1e-20).log()).sum()),
                raw_latent_rms=raw.square().mean().sqrt(), latent_rms=x.square().mean().sqrt(),
                codebook_rms=self.normalize(self.embedding.weight).square().mean().sqrt(),
                initialized_from_train=self.initialized_from_train.float())
        return dict(quantized=straight_through(x, codes), indices=indices,
                    loss=codebook + self.commitment * commitment,
                    diagnostics={k: v.detach() for k,v in diagnostics.items()})
