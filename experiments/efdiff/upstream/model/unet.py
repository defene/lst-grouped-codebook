"""UNet architecture for diffusion-based LST super-resolution.

Building blocks:
  - PositionalEncoding, FeatureWiseAffine, Swish, Block, ResnetBlock
  - SelfAttention, CrossAttention
  - Upsample, Downsample

Single unified UNet class that supports:
  - Standard mode (no cross-attention): for sr3, resshift
  - Cross-attention mode: for sr3_prithvi, resshift_prithvi
    Activated by passing context_dim to the constructor.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from inspect import isfunction


# ============================================================================
# Utility helpers
# ============================================================================

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# ============================================================================
# Building blocks
# ============================================================================

# PositionalEncoding Source: https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=8, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        # (B, 3*C, H, W) -> (B, 3, n_head, head_dim, H*W)
        qkv = self.qkv(norm).view(batch, 3, n_head, head_dim, height * width)
        query, key, value = qkv.unbind(1)  # each (B, n_head, head_dim, H*W)

        # (B, n_head, H*W, head_dim) — expected by scaled_dot_product_attention
        query = query.transpose(2, 3)
        key = key.transpose(2, 3)
        value = value.transpose(2, 3)

        out = F.scaled_dot_product_attention(query, key, value)  # Flash Attention
        out = out.transpose(2, 3).contiguous().view(batch, channel, height, width)
        out = self.out(out)

        return out + input


# ============================================================================
# Cross-attention (for Prithvi-EO conditioning)
# ============================================================================

class CrossAttention(nn.Module):
    """Cross-attention: Q from UNet spatial features, K/V from Prithvi context.

    Args:
        query_dim: Channel dimension of UNet features (Q source)
        context_dim: Embedding dimension of Prithvi output (K/V source), default 1024
        n_head: Number of attention heads
        norm_groups: Groups for GroupNorm on query
    """

    def __init__(self, query_dim, context_dim=1024, n_head=8, norm_groups=32):
        super().__init__()
        self.n_head = n_head
        self.head_dim = query_dim // n_head
        assert query_dim % n_head == 0, \
            "query_dim ({}) must be divisible by n_head ({})".format(query_dim, n_head)

        self.norm_q = nn.GroupNorm(norm_groups, query_dim)
        self.norm_ctx = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(0.0),
        )

    def forward(self, x, context):
        B, C, H, W = x.shape
        n_head = self.n_head
        head_dim = self.head_dim

        q = self.norm_q(x)
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        q = self.to_q(q)

        ctx = self.norm_ctx(context)
        k = self.to_k(ctx)
        v = self.to_v(ctx)

        q = q.view(B, -1, n_head, head_dim).permute(0, 2, 1, 3)
        k = k.view(B, -1, n_head, head_dim).permute(0, 2, 1, 3)
        v = v.view(B, -1, n_head, head_dim).permute(0, 2, 1, 3)

        if getattr(self, '_store_attn', False):
            # Manual attention to capture weights
            scale = self.head_dim ** -0.5
            attn_weights = torch.matmul(q * scale, k.transpose(-2, -1))
            attn_weights = attn_weights.softmax(dim=-1)
            self._attn_weights = attn_weights.detach().cpu()  # (B, n_head, H*W, N_ctx)
            out = torch.matmul(attn_weights, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v)  # Flash Attention

        out = out.permute(0, 2, 1, 3).contiguous().view(B, H * W, C)
        out = self.to_out(out)
        out = out.permute(0, 2, 1).view(B, C, H, W)

        return out + x


# ============================================================================
# ResNet block with optional self-attention and cross-attention
# ============================================================================

class ResnetBlockWithAttn(nn.Module):
    """ResnetBlock + optional SelfAttention + optional CrossAttention."""

    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32,
                 dropout=0, with_attn=False, with_cross_attn=False,
                 context_dim=1024, cross_attn_heads=8):
        super().__init__()
        self.with_attn = with_attn
        self.with_cross_attn = with_cross_attn

        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim,
            norm_groups=norm_groups, dropout=dropout)

        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

        if with_cross_attn:
            self.cross_attn = CrossAttention(
                query_dim=dim_out,
                context_dim=context_dim,
                n_head=cross_attn_heads,
                norm_groups=norm_groups,
            )

    def forward(self, x, time_emb, context=None):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        if self.with_cross_attn and context is not None:
            x = self.cross_attn(x, context)
        return x


# ============================================================================
# Unified UNet
# ============================================================================

class UNet(nn.Module):
    """Encoder-decoder UNet with optional Prithvi cross-attention conditioning.

    When context_dim is None (default), this is a standard UNet with only
    self-attention. When context_dim is set, cross-attention is injected at
    the mid block and decoder blocks at resolutions in cross_attn_res.
    The encoder never uses cross-attention.

    Args:
        context_dim: If set, enables cross-attention with this embedding dim.
                     Use 1024 for Prithvi-EO-2.0-300M.
        cross_attn_heads: Number of heads for cross-attention layers.
        cross_attn_res: Resolutions at which to inject decoder cross-attention.
                        Defaults to attn_res if not specified.
    """

    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8,),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128,
        context_dim=None,
        cross_attn_heads=8,
        cross_attn_res=None,
    ):
        super().__init__()

        has_cross_attn = context_dim is not None
        if cross_attn_res is None:
            cross_attn_res = attn_res if has_cross_attn else []
        self.context_dim = context_dim

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size

        # Encoder (no cross-attention)
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlockWithAttn(
                    pre_channel, channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout,
                    with_attn=use_attn,
                    with_cross_attn=False,
                ))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        # Mid block (cross-attention if enabled)
        self.mid = nn.ModuleList([
            ResnetBlockWithAttn(
                pre_channel, pre_channel,
                noise_level_emb_dim=noise_level_channel,
                norm_groups=norm_groups, dropout=dropout,
                with_attn=True,
                with_cross_attn=has_cross_attn,
                context_dim=context_dim or 1024,
                cross_attn_heads=cross_attn_heads,
            ),
            ResnetBlockWithAttn(
                pre_channel, pre_channel,
                noise_level_emb_dim=noise_level_channel,
                norm_groups=norm_groups, dropout=dropout,
                with_attn=False,
                with_cross_attn=False,
            ),
        ])

        # Decoder (cross-attention at specified resolutions)
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            use_cross = has_cross_attn and (now_res in cross_attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlockWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout,
                    with_attn=use_attn,
                    with_cross_attn=use_cross,
                    context_dim=context_dim or 1024,
                    cross_attn_heads=cross_attn_heads,
                ))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)
        self.final_conv = Block(pre_channel, default(out_channel, in_channel),
                                groups=norm_groups)

    def forward(self, x, time, context=None):
        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlockWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlockWithAttn):
                x = layer(x, t, context=context)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlockWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t,
                          context=context)
            else:
                x = layer(x)

        return self.final_conv(x)
