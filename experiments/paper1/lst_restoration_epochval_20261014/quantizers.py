"""Explicit, rate-controlled quantizer adaptations for the LST restoration study.

These are mechanism implementations, not claims to reproduce full FSQ,
MAGVIT-v2, or BSQ training recipes. All public latent tensors are BCHW;
single-scale indices are BHW and VAR multi-scale indices are B,sum(scale**2).
No model construction, tensor computation, or experiment runs at import time.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def straight_through(value: Tensor, hard: Tensor) -> Tensor:
    """Exact hard forward values with identity gradients to the soft values.

    value + (hard - value).detach() loses small codebook values through
    cancellation; a decoder can amplify that error. Subtract identical soft
    values first so finite inputs produce exactly the index-decodable codes.
    """
    return hard.detach() + (value - value.detach())


def _validate_indices(indices: Tensor, size: int) -> Tensor:
    if indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("Indices must have an integer dtype")
    if indices.numel() and (bool((indices < 0).any()) or bool((indices >= size).any())):
        raise ValueError(f"Index outside [0,{size})")
    return indices.long()


@torch.no_grad()
def nearest_indices(tokens: Tensor, codebook: Tensor, chunk_size: int = 1024) -> Tensor:
    """FP32 exact nearest neighbour; ties follow torch.argmin's first index.

    Distance matrices are at most chunk_size x codebook_size. Neither the
    search nor indices participate in autograd; codebook updates use embedding
    lookup in the differentiable loss after this search.
    """
    if tokens.ndim != 2 or codebook.ndim != 2 or tokens.shape[1] != codebook.shape[1]:
        raise ValueError("Expected matching [N,D] and [K,D] tensors")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        weights = codebook.float()
        norm = weights.square().sum(-1)
        answers = []
        for start in range(0, tokens.shape[0], chunk_size):
            part = tokens[start:start + chunk_size].float()
            distances = part.square().sum(-1, keepdim=True) + norm.unsqueeze(0)
            distances.addmm_(part, weights.T, beta=1.0, alpha=-2.0)
            answers.append(distances.argmin(-1))
        return torch.cat(answers) if answers else torch.empty(0, dtype=torch.long, device=tokens.device)


class VectorQuantizer(nn.Module):
    """Single-scale gradient-updated VQ with an explicit commitment term."""

    def __init__(self, dim: int, size: int, commitment: float = 0.25, chunk_size: int = 1024):
        super().__init__()
        if dim < 1 or size < 2 or commitment < 0:
            raise ValueError("Invalid VQ dimension, size or commitment")
        self.dim, self.codebook_size = dim, size
        self.commitment, self.chunk_size = commitment, chunk_size
        self.embedding = nn.Embedding(size, dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / size, 1.0 / size)

    def decode_indices(self, indices: Tensor) -> Tensor:
        indices = _validate_indices(indices, self.codebook_size)
        if indices.ndim != 3:
            raise ValueError("Single-scale indices must be [B,H,W]")
        with torch.autocast(device_type=indices.device.type, enabled=False):
            return self.embedding(indices).float().permute(0, 3, 1, 2).contiguous()

    def forward(self, latent: Tensor) -> dict[str, Any]:
        b, c, h, w = latent.shape
        if c != self.dim:
            raise ValueError("VQ channel mismatch")
        with torch.autocast(device_type=latent.device.type, enabled=False):
            x = latent.float()
            flat = x.permute(0, 2, 3, 1).reshape(-1, c)
            indices = nearest_indices(flat, self.embedding.weight, self.chunk_size).reshape(b, h, w)
            codes = self.decode_indices(indices).float()
            codebook_loss = F.mse_loss(codes, x.detach())
            commitment_loss = F.mse_loss(x, codes.detach())
            loss = codebook_loss + self.commitment * commitment_loss
            quantized = straight_through(x, codes)
        return {"quantized": quantized, "indices": indices, "loss": loss,
                "diagnostics": {"codebook_loss": codebook_loss.detach(),
                                "commitment_loss": commitment_loss.detach()}}


class FiniteScalarQuantizer(nn.Module):
    """FSQ with mixed-radix indices, including the even-level offset.

    The scalar bound/offset follows the FSQ formulation; [8,8,16] is this
    study's exact-1024-state adaptation, not a claimed paper default.
    """

    def __init__(self, dim: int, levels: list[int], eps: float = 1e-3):
        super().__init__()
        if not levels or any(int(v) != v or v < 2 for v in levels):
            raise ValueError("FSQ levels must be integers >= 2")
        self.dim, self.codebook_size = dim, math.prod(levels)
        self.eps = eps
        self.project_in = nn.Conv2d(dim, len(levels), 1)
        self.project_out = nn.Conv2d(len(levels), dim, 1)
        bases, product = [], 1
        for level in levels:
            bases.append(product)
            product *= level
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.long))
        self.register_buffer("bases", torch.tensor(bases, dtype=torch.long))

    def decode_indices(self, indices: Tensor) -> Tensor:
        indices = _validate_indices(indices, self.codebook_size)
        if indices.ndim != 3:
            raise ValueError("FSQ indices must be [B,H,W]")
        with torch.autocast(device_type=indices.device.type, enabled=False):
            digits = (indices.unsqueeze(1) // self.bases[None, :, None, None]) % self.levels[None, :, None, None]
            half = (self.levels // 2)[None, :, None, None].float()
            codes = (digits.float() - half) / half
            return self.project_out(codes)

    def forward(self, latent: Tensor) -> dict[str, Any]:
        with torch.autocast(device_type=latent.device.type, enabled=False):
            z = self.project_in(latent.float())
            levels = self.levels[None, :, None, None].float()
            half_width = self.levels.div(2, rounding_mode="floor")[None, :, None, None].float()
            half_bound = (levels - 1) * (1 + self.eps) / 2
            offset = torch.where((self.levels % 2 == 0)[None, :, None, None], 0.5, 0.0)
            shift = torch.atanh(offset / half_bound)
            bounded = torch.tanh(z + shift) * half_bound - offset
            rounded = bounded.round()
            ste = straight_through(bounded, rounded)
            digits = (rounded + half_width).long()
            indices = (digits * self.bases[None, :, None, None]).sum(1)
            codes = ste / half_width
            quantized = self.project_out(codes)
        return {"quantized": quantized, "indices": indices,
                "loss": quantized.new_zeros(()), "diagnostics": {}}


class BinaryQuantizer(nn.Module):
    """LFQ or BSQ mechanism at an explicitly limited binary vocabulary.

    LFQ: {-1,+1} sign codes, commitment and confidence/diversity entropy.
    BSQ: L2-normalized input, {-1,+1}/sqrt(bits) codes; entropy, no VQ-style
    commitment. Both use an exact joint categorical entropy adaptation:
    E[H(p(code|token))] - H(E[p(code|token)]), with a positive configured
    multiplier. It is not bitwise entropy mislabeled joint code entropy.
    Soft assignments use squared-Euclidean Gibbs probabilities. Coefficients
    and temperature are experiment adaptations and must be reported.
    """

    def __init__(self, dim: int, bits: int, spherical: bool, commitment: float,
                 entropy_weight: float, temperature: float, entropy_chunk: int = 256):
        super().__init__()
        if not 1 <= bits <= 12:
            raise ValueError("Exact joint entropy implementation supports 1..12 bits")
        if temperature <= 0 or entropy_weight < 0 or commitment < 0 or entropy_chunk < 1:
            raise ValueError("Invalid binary quantizer loss parameters")
        self.dim, self.bits, self.spherical = dim, bits, spherical
        self.codebook_size = 2 ** bits
        self.commitment = 0.0 if spherical else commitment
        self.entropy_weight, self.temperature = entropy_weight, temperature
        self.entropy_chunk = entropy_chunk
        self.project_in = nn.Conv2d(dim, bits, 1)
        self.project_out = nn.Conv2d(bits, dim, 1)
        self.register_buffer("bases", 2 ** torch.arange(bits, dtype=torch.long))
        values = torch.arange(self.codebook_size, dtype=torch.long).unsqueeze(1)
        binary = ((values // self.bases) % 2).float() * 2 - 1
        self.register_buffer("binary_codes", binary / math.sqrt(bits) if spherical else binary)

    def _codes(self, indices: Tensor) -> Tensor:
        return self.binary_codes[_validate_indices(indices, self.codebook_size)].permute(0, 3, 1, 2).contiguous()

    def decode_indices(self, indices: Tensor) -> Tensor:
        if indices.ndim != 3:
            raise ValueError("Binary indices must be [B,H,W]")
        with torch.autocast(device_type=indices.device.type, enabled=False):
            return self.project_out(self._codes(indices))

    def _entropy(self, z: Tensor) -> tuple[Tensor, Tensor]:
        flat = z.permute(0, 2, 3, 1).reshape(-1, self.bits)
        count = flat.shape[0]
        marginal_sum = flat.new_zeros(self.codebook_size)
        conditional_sum = flat.new_zeros(())
        for chunk in flat.split(self.entropy_chunk):
            logits = (2.0 / self.temperature) * (chunk @ self.binary_codes.T)
            log_p = F.log_softmax(logits, dim=-1)
            p = log_p.exp()
            conditional_sum = conditional_sum - (p * log_p).sum()
            marginal_sum = marginal_sum + p.sum(0)
        marginal = marginal_sum / max(count, 1)
        conditional = conditional_sum / max(count, 1)
        entropy = -(marginal * marginal.clamp_min(1e-12).log()).sum()
        return conditional, entropy

    def forward(self, latent: Tensor) -> dict[str, Any]:
        with torch.autocast(device_type=latent.device.type, enabled=False):
            z = self.project_in(latent.float())
            if self.spherical:
                z = F.normalize(z, dim=1, eps=1e-8)
            positive = z >= 0
            indices = (positive.long() * self.bases[None, :, None, None]).sum(1)
            hard = self._codes(indices)
            codes = straight_through(z, hard)
            quantized = self.project_out(codes)
            commitment = F.mse_loss(z, hard.detach()) if self.commitment else z.new_zeros(())
            if self.training and self.entropy_weight:
                conditional, marginal = self._entropy(z)
            else:
                conditional, marginal = z.new_zeros(()), z.new_zeros(())
            loss = self.commitment * commitment + self.entropy_weight * (conditional - marginal)
        return {"quantized": quantized, "indices": indices, "loss": loss,
                "diagnostics": {"conditional_soft_entropy_nats": conditional.detach(),
                                "marginal_soft_entropy_nats": marginal.detach(),
                                "commitment_loss": commitment.detach()}}


class MultiScaleVectorQuantizer(nn.Module):
    """VAR residual multi-scale quantization with explicit returned indices.

    Uses the pinned local VAR residual convolutions/interpolation policy.
    Search is chunked. Per-scale losses follow the reference's accumulated
    f_hat objective; no autoregressive generation model is constructed.
    """

    def __init__(self, dim: int, size: int, commitment: float, scales: list[int],
                 quant_resi: float = 0.5, share_quant_resi: int = 4, chunk_size: int = 1024):
        super().__init__()
        from eo_denoise.var_quant import VectorQuantizer2
        if len(scales) < 2 or scales != sorted(set(scales)) or min(scales) < 1:
            raise ValueError("VAR requires at least two unique ascending positive scales")
        self.core = VectorQuantizer2(vocab_size=size, Cvae=dim, using_znorm=False,
                                    beta=commitment, v_patch_nums=tuple(scales),
                                    quant_resi=quant_resi, share_quant_resi=share_quant_resi)
        self.dim, self.codebook_size, self.scales = dim, size, tuple(scales)
        self.commitment, self.chunk_size = commitment, chunk_size

    def decode_indices(self, indices: Tensor) -> Tensor:
        from eo_denoise.var_ops import interpolate
        indices = _validate_indices(indices, self.codebook_size)
        if indices.ndim != 2 or indices.shape[1] != sum(s * s for s in self.scales):
            raise ValueError("VAR indices have the wrong number of scale tokens")
        with torch.autocast(device_type=indices.device.type, enabled=False):
            b, end = indices.shape[0], self.scales[-1]
            result = self.core.embedding.weight.new_zeros(b, self.dim, end, end, dtype=torch.float32)
            offset = 0
            for i, side in enumerate(self.scales):
                idx = indices[:, offset:offset + side * side].reshape(b, side, side)
                h = self.core.embedding(idx).float().permute(0, 3, 1, 2).contiguous()
                if side != end:
                    h = interpolate(h, size=(end, end), mode="bicubic")
                result = result + self.core.quant_resi[i / (len(self.scales) - 1)](h)
                offset += side * side
            return result

    def forward(self, latent: Tensor) -> dict[str, Any]:
        from eo_denoise.var_ops import interpolate
        b, c, h, w = latent.shape
        if c != self.dim or (h, w) != (self.scales[-1], self.scales[-1]):
            raise ValueError("VAR finest scale must equal the encoder latent grid")
        with torch.autocast(device_type=latent.device.type, enabled=False):
            x = latent.float()
            residual = x.detach().clone()
            accumulated = torch.zeros_like(x)
            loss, indices = x.new_zeros(()), []
            for i, side in enumerate(self.scales):
                coarse = interpolate(residual, size=(side, side), mode="area") if side != h else residual
                idx = nearest_indices(coarse.permute(0, 2, 3, 1).reshape(-1, c),
                                      self.core.embedding.weight, self.chunk_size).reshape(b, side, side)
                indices.append(idx.flatten(1))
                codes = self.core.embedding(idx).permute(0, 3, 1, 2).contiguous()
                if side != h:
                    codes = interpolate(codes, size=(h, w), mode="bicubic")
                codes = self.core.quant_resi[i / (len(self.scales) - 1)](codes)
                accumulated = accumulated + codes
                residual = residual - codes.detach()
                loss = loss + self.commitment * F.mse_loss(x, accumulated.detach()) + F.mse_loss(accumulated, x.detach())
            loss = loss / len(self.scales)
            quantized = x + (accumulated - x).detach()
        return {"quantized": quantized, "indices": torch.cat(indices, dim=1),
                "loss": loss, "diagnostics": {}}


def build_quantizer(model_config: dict[str, Any]) -> nn.Module | None:
    kind = model_config["kind"]
    dim = int(model_config.get("latent_dim", 32))
    size = int(model_config.get("codebook_size", 1024))
    commitment = float(model_config.get("commitment", 0.25))
    if kind == 'grouped_g8k256':
        from .grouped_adapter import GroupedProductQuantizer
        return GroupedProductQuantizer(model_config)
    if kind == 'vq_drifting':
        from .stage2_models import DriftingVectorQuantizer
        return DriftingVectorQuantizer(model_config)
    if kind == "ae":
        return None
    if kind == "vq":
        if model_config.get("vq_update", "gradient") != "gradient":
            raise NotImplementedError("EMA/restart VQ remains a separate pending recipe; no gradient-VQ substitution")
        return VectorQuantizer(dim, size, commitment, int(model_config.get("distance_chunk", 1024)))
    if kind == "fsq":
        levels = list(model_config.get("levels", [8, 8, 16]))
        if math.prod(levels) != size:
            raise ValueError("FSQ codebook_size must equal product(levels)")
        return FiniteScalarQuantizer(dim, levels)
    if kind in ("lfq", "bsq"):
        bits = int(model_config.get("bits_per_token", 10))
        if 2 ** bits != size:
            raise ValueError("Binary quantizer codebook_size must equal 2**bits_per_token")
        return BinaryQuantizer(dim, bits, kind == "bsq", commitment,
                               float(model_config.get("entropy_weight", 0.01)),
                               float(model_config.get("entropy_temperature", 0.1)))
    if kind == "var_multiscale":
        return MultiScaleVectorQuantizer(dim, size, commitment,
                                        list(model_config.get("scales", [1, 2, 3, 4, 5, 6, 8, 10, 13, 16])),
                                        float(model_config.get("quant_resi", 0.5)),
                                        int(model_config.get("share_quant_resi", 4)))
    raise ValueError(f"Not a supported quantizer kind: {kind}")
