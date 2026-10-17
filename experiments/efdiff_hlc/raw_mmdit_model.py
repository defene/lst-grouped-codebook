"""HLC raw-HLS dual-stream pixel MMDiT. No pretrained encoder or VAE.

Input x is [normalized coarse LST, noisy normalized LST residual]. HLS is
unclipped reflectance; HLS_valid is a full-resolution boolean observation mask.
This module never starts training or downloads weights.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from dit_model import Attention, AxialRoPE2D, HeadRMSNorm, fourier, modulate


class FeedForward(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        gate, value = self.up(x).chunk(2, -1)
        return self.down(F.silu(gate) * value)


class JointMetadata(nn.Module):
    """One MLP jointly mixes log-SNR, periodic month and spherical geography."""
    def __init__(self, width):
        super().__init__()
        self.register_buffer('harmonics', torch.arange(1, 4).float())
        self.register_buffer('bands', 2. ** torch.arange(3).float())
        self.mlp = nn.Sequential(nn.Linear(256 + 6 + 21, width), nn.SiLU(),
                                 nn.Linear(width, width))

    def features(self, alpha, month, latlon):
        b = alpha.numel()
        if month.shape != (b,) or latlon.shape != (b, 2):
            raise ValueError('Expected month [B], crop-center latlon [B,2]')
        if not torch.isfinite(alpha).all() or not ((alpha >= 0) & (alpha <= 1)).all():
            raise ValueError('Expected finite VP alpha in [0,1], not a timestep ID')
        if not torch.isfinite(month).all() or not ((month >= 1) & (month <= 12) & (month == month.round())).all():
            raise ValueError('Expected integer calendar month 1..12')
        if not torch.isfinite(latlon).all() or not ((latlon[:, 0].abs() <= 90) & (latlon[:, 1].abs() <= 180)).all():
            raise ValueError('Expected valid latitude/longitude in degrees')
        a2 = alpha.float().reshape(b).square().clamp(1e-6, 1-1e-6)
        time = fourier(torch.log(a2) - torch.log1p(-a2), 256)
        phase = (month.float()[:, None] - 1) * (2 * math.pi / 12) * self.harmonics
        month_features = torch.cat((phase.sin(), phase.cos()), -1)
        lat, lon = torch.deg2rad(latlon.float()).unbind(-1)
        xyz = torch.stack((lat.cos()*lon.cos(), lat.cos()*lon.sin(), lat.sin()), -1)
        phase = xyz[:, :, None] * self.bands * math.pi
        geo = torch.cat((xyz, phase.sin().flatten(1), phase.cos().flatten(1)), -1)
        return torch.cat((time, month_features, geo), -1)

    def forward(self, alpha, month, latlon):
        return self.mlp(self.features(alpha, month, latlon))


class ConditionBlock(nn.Module):
    def __init__(self, width, heads, grid, ffn_hidden):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = Attention(width, heads, rope_grid=grid)
        self.ffn = FeedForward(width, ffn_hidden)

    def forward(self, c):
        c = c + self.attention(self.norm1(c))
        return c + self.ffn(self.norm2(c))


class Stream(nn.Module):
    def __init__(self, width, heads, ffn_hidden, read_only=False):
        super().__init__()
        self.heads, self.dim = heads, width // heads
        self.read_only = read_only
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, (2 if read_only else 6) * width))
        self.qkv = nn.Linear(width, (2 if read_only else 3) * width)
        self.k_norm = HeadRMSNorm(self.dim)
        if not read_only:
            self.q_norm = HeadRMSNorm(self.dim)
            self.out = nn.Linear(width, width)
            self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            self.ffn = FeedForward(width, ffn_hidden)

    def prepare(self, tokens, metadata, rope):
        mods = self.modulation(metadata).chunk(2 if self.read_only else 6, -1)
        value = modulate(self.norm1(tokens), mods[0], mods[1])
        b, n, _ = value.shape
        parts = self.qkv(value).reshape(b, n, 2 if self.read_only else 3, self.heads, self.dim).permute(2, 0, 3, 1, 4)
        if self.read_only:
            k, v = parts.unbind(0)
            return None, rope(self.k_norm(k)), v, mods
        q, k, v = parts.unbind(0)
        return rope(self.q_norm(q)), rope(self.k_norm(k)), v, mods

    def finish(self, tokens, attention, mods):
        value = attention.transpose(1, 2).reshape_as(tokens)
        tokens = tokens + mods[2][:, None] * self.out(value)
        return tokens + mods[5][:, None] * self.ffn(modulate(self.norm2(tokens), mods[3], mods[4]))


class JointBlock(nn.Module):
    def __init__(self, width, heads, grid, ffn_hidden, last=False):
        super().__init__()
        self.last = last
        self.rope = AxialRoPE2D(grid, width // heads)
        self.condition = Stream(width, heads, ffn_hidden, read_only=last)
        self.temperature = Stream(width, heads, ffn_hidden)

    def forward(self, c, x, metadata):
        qc, kc, vc, mc = self.condition.prepare(c, metadata, self.rope)
        qx, kx, vx, mx = self.temperature.prepare(x, metadata, self.rope)
        k, v = torch.cat((kc, kx), -2), torch.cat((vc, vx), -2)
        # No causal or cloud attention mask: each condition token also holds LST.
        if self.last:
            # The final condition output is never consumed. Omit its Q/out/FFN
            # and gates, rather than padding the matched budget with dead weights.
            ax = F.scaled_dot_product_attention(qx, k, v, dropout_p=0.)
        else:
            q = torch.cat((qc, qx), -2)
            ac, ax = F.scaled_dot_product_attention(q, k, v, dropout_p=0.).split((c.shape[1], x.shape[1]), -2)
            c = self.condition.finish(c, ac, mc)
        return c, self.temperature.finish(x, ax, mx)


class RawHLSMMDiT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, hidden_dim=896, heads=14,
                 encoder_depth=4, depth=6, ffn_hidden=2672, gradient_checkpointing=True):
        super().__init__()
        if min(image_size, patch_size, hidden_dim, heads, encoder_depth, depth, ffn_hidden) <= 0:
            raise ValueError('Architecture dimensions must be positive')
        if image_size % patch_size or hidden_dim % heads or (hidden_dim // heads) % 4:
            raise ValueError('Image/patch and width/head dimensions incompatible with 2D RoPE')
        self.image_size, self.patch_size, self.hidden_dim = image_size, patch_size, hidden_dim
        self.grid = image_size // patch_size
        self.gradient_checkpointing = gradient_checkpointing
        # Preserve the previous affine HLS scaling to avoid changing normalization
        # alongside the backbone. These are constants, not pretrained parameters.
        self.register_buffer('hls_mean_dn', torch.tensor([1087,1342,1433,2734,1958,1363.]).reshape(1,6,1,1))
        self.register_buffer('hls_std_dn', torch.tensor([2248,2179,2178,1850,1242,1049.]).reshape(1,6,1,1))
        # All 256 mask pixels of every patch enter this learned projection.
        self.condition_embed = nn.Conv2d(8, hidden_dim, patch_size, stride=patch_size)
        self.temperature_embed = nn.Conv2d(1, hidden_dim, patch_size, stride=patch_size)
        self.encoder = nn.ModuleList([ConditionBlock(hidden_dim, heads, self.grid, ffn_hidden) for _ in range(encoder_depth)])
        self.metadata = JointMetadata(hidden_dim)
        self.blocks = nn.ModuleList([JointBlock(hidden_dim, heads, self.grid, ffn_hidden, last=i == depth-1) for i in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2*hidden_dim))
        self.head = nn.Linear(hidden_dim, patch_size**2)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight.flatten(1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            for stream in (block.condition, block.temperature):
                nn.init.zeros_(stream.modulation[-1].weight)
                nn.init.zeros_(stream.modulation[-1].bias)
        for module in (self.final_modulation[-1], self.head):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def encode_condition(self, coarse, hls, hls_valid):
        b, size = coarse.shape[0], self.image_size
        if coarse.shape != (b,1,size,size) or hls.shape != (b,6,size,size):
            raise ValueError('Expected coarse [B,1,H,W] and six-band HLS [B,6,H,W]')
        if hls_valid.shape != (b,1,size,size) or hls_valid.dtype != torch.bool:
            raise ValueError('Expected full-resolution boolean HLS_valid [B,1,H,W]')
        if not torch.isfinite(coarse).all() or not torch.isfinite(hls.masked_select(hls_valid.expand_as(hls))).all():
            raise ValueError('Nonfinite valid input observations')
        # Replace invalid NaN/Inf before arithmetic; mask remains a separate input.
        safe = torch.where(hls_valid, hls.float(), torch.zeros_like(hls, dtype=torch.float32))
        normalized = (safe * 10000 - self.hls_mean_dn) / self.hls_std_dn
        normalized = torch.where(hls_valid, normalized, normalized.new_tensor(.0001))
        c = self.condition_embed(torch.cat((normalized, coarse.float(), hls_valid.float()), 1)).flatten(2).transpose(1,2)
        for block in self.encoder:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                c = checkpoint(block, c, use_reentrant=False)
            else:
                c = block(c)
        return c

    def forward(self, x, noise_level, context=None, *, hls=None, hls_valid=None, month=None, geo_latlon=None):
        b, size = x.shape[0], self.image_size
        if x.shape != (b,2,size,size):
            raise ValueError('Expected [coarse LST, noisy residual] at configured image size')
        if noise_level.numel() != b or month is None or geo_latlon is None:
            raise ValueError('Supply one alpha, month and crop-center coordinate per example')
        if context is None:
            if hls is None or hls_valid is None:
                raise ValueError('Raw HLS and its full-resolution validity mask are required')
            context = self.encode_condition(x[:,:1], hls, hls_valid)
        elif context.shape != (b, self.grid**2, self.hidden_dim):
            raise ValueError('Invalid cached condition shape')
        metadata = self.metadata(noise_level, month, geo_latlon)
        tokens = self.temperature_embed(x[:,1:]).flatten(2).transpose(1,2)
        # Each call starts from encoder context; do not carry updated C across NFE.
        c = context
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                c, tokens = checkpoint(block, c, tokens, metadata, use_reentrant=False)
            else:
                c, tokens = block(c, tokens, metadata)
        shift, scale = self.final_modulation(metadata).chunk(2, -1)
        patches = self.head(modulate(self.final_norm(tokens), shift, scale))
        p, g = self.patch_size, self.grid
        return patches.reshape(b,g,g,p,p).permute(0,1,3,2,4).reshape(b,1,size,size)


def build(objective='v', device='cpu', architecture=None):
    """Build the existing v-target/sampler with a trainable raw condition encoder."""
    from raw_mmdit_objective import RawMMDiTObjective
    if objective != 'v':
        raise ValueError('This baseline fixes the existing VP v-pred objective')
    return RawMMDiTObjective(RawHLSMMDiT(**(architecture or {}))).to(device)
