"""Pixel-space DiT with time adaLN-Zero and spatial Prithvi cross-attention.

Architecture reference: https://arxiv.org/abs/2212.09748 (not pretrained DiT).
Keeps the existing denoiser interface: (condition+noisy residual, alpha, context).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def fourier(values, width):
    freq = torch.exp(-math.log(10000) * torch.arange(width // 2, device=values.device).float() / (width // 2))
    phase = values.float().unsqueeze(-1) * freq
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


def position_embedding(grid, width):
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
    return torch.cat((fourier(y.flatten(), width // 2), fourier(x.flatten(), width // 2)), dim=-1)[None]


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class HeadRMSNorm(nn.Module):
    """Normalize Q/K per head in fp32, then return to the attention dtype."""
    def __init__(self, head_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight.float()).to(x.dtype)


class AxialRoPE2D(nn.Module):
    """Half the channel pairs rotate by row, half by column (row-major tokens)."""
    def __init__(self, grid, head_dim, theta=10000.):
        super().__init__()
        if head_dim % 4:
            raise ValueError('2D RoPE needs head_dim divisible by four')
        y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
        freq = theta ** (-torch.arange(head_dim // 4).float() / (head_dim // 4))
        angles = torch.cat((y.flatten()[:, None]*freq, x.flatten()[:, None]*freq), dim=-1)
        self.register_buffer('cos', angles.cos()[None, None])
        self.register_buffer('sin', angles.sin()[None, None])

    def forward(self, x):
        if x.shape[-2] != self.cos.shape[-2]:
            raise ValueError('RoPE token count must match the configured spatial grid')
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        even, odd = pairs.unbind(-1)
        cos, sin = self.cos.to(x.dtype), self.sin.to(x.dtype)
        return torch.stack((even*cos-odd*sin, even*sin+odd*cos), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, width, mlp_ratio=4, multiple=64):
        super().__init__()
        # Three projection matrices instead of GELU FFN's two: match matrix budget.
        self.hidden_dim = math.ceil((2 * width * mlp_ratio / 3) / multiple) * multiple
        self.gate_up = nn.Linear(width, 2*self.hidden_dim, bias=False)
        self.down = nn.Linear(self.hidden_dim, width, bias=False)

    def forward(self, x):
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class Attention(nn.Module):
    def __init__(self, width, heads, context_dim=None, qk_norm=True, rope_grid=None):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(context_dim or width, 2 * width)
        self.out = nn.Linear(width, width)
        self.q_norm = HeadRMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = HeadRMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.rope = AxialRoPE2D(rope_grid, self.head_dim) if rope_grid is not None else None

    def forward(self, x, context=None):
        b, n, d = x.shape
        context = x if context is None else context
        q = self.q(x).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(context).reshape(b, -1, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        value = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        return self.out(value.transpose(1, 2).reshape(b, n, d))


class CrossDiTBlock(nn.Module):
    def __init__(self, width, heads, context_dim, mlp_ratio, ffn_type, ffn_multiple, qk_norm, rope_grid):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width, elementwise_affine=False, eps=1e-6) for _ in range(3)])
        self.self_attention = Attention(width, heads, qk_norm=qk_norm, rope_grid=rope_grid)
        self.context_norm = nn.LayerNorm(context_dim, eps=1e-6)
        # Prithvi tokens already carry the frozen encoder's positional information.
        # RoPE is applied only to DiT self-attention; cross-attention uses QK-Norm.
        self.cross_attention = Attention(width, heads, context_dim, qk_norm=qk_norm)
        self.mlp = (SwiGLU(width, mlp_ratio, ffn_multiple) if ffn_type == 'swiglu' else
                    nn.Sequential(nn.Linear(width, int(width * mlp_ratio)), nn.GELU(approximate='tanh'),
                                  nn.Linear(int(width * mlp_ratio), width)))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 9 * width))

    def forward(self, x, time, context):
        s1, c1, g1, s2, c2, g2, s3, c3, g3 = self.modulation(time).chunk(9, dim=-1)
        x = x + g1[:, None] * self.self_attention(modulate(self.norms[0](x), s1, c1))
        x = x + g2[:, None] * self.cross_attention(modulate(self.norms[1](x), s2, c2), self.context_norm(context))
        return x + g3[:, None] * self.mlp(modulate(self.norms[2](x), s3, c3))


class PrithviCrossDiT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, hidden_dim=1024, depth=8,
                 heads=16, context_dim=1024, mlp_ratio=4, gradient_checkpointing=True,
                 ffn_type='swiglu', ffn_multiple=64, qk_norm=True, rope_2d=True):
        super().__init__()
        if min(image_size, patch_size, hidden_dim, depth, heads, context_dim, mlp_ratio, ffn_multiple) <= 0:
            raise ValueError('Architecture dimensions must be positive')
        if image_size % patch_size or hidden_dim % 4 or hidden_dim % heads:
            raise ValueError('Image must divide into patches; width must divide by 4 and heads')
        if ffn_type not in ('swiglu', 'gelu'):
            raise ValueError('ffn_type must be swiglu or gelu')
        self.image_size, self.patch_size = image_size, patch_size
        self.hidden_dim, self.context_dim = hidden_dim, context_dim
        self.grid = image_size // patch_size
        self.gradient_checkpointing = gradient_checkpointing
        self.patch_embed = nn.Conv2d(2, hidden_dim, patch_size, stride=patch_size)
        self.register_buffer('pos_embed', None if rope_2d else position_embedding(self.grid, hidden_dim))
        self.time_mlp = nn.Sequential(nn.Linear(256, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([CrossDiTBlock(hidden_dim, heads, context_dim, mlp_ratio,
                                   ffn_type, ffn_multiple, qk_norm, self.grid if rope_2d else None)
                                   for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.head = nn.Linear(hidden_dim, patch_size ** 2)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embed.weight.flatten(1))
        nn.init.zeros_(self.patch_embed.bias)
        for m in (self.time_mlp[0], self.time_mlp[2]):
            nn.init.normal_(m.weight, std=.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def unpatchify(self, patches):
        b = patches.shape[0]
        p, g = self.patch_size, self.grid
        return patches.reshape(b, g, g, p, p).permute(0, 1, 3, 2, 4).reshape(b, 1, g*p, g*p)

    def forward(self, x, noise_level, context=None):
        b = x.shape[0]
        if x.shape[1:] != (2, self.image_size, self.image_size):
            raise ValueError('Expected concatenated condition/noisy residual at configured image size')
        if context is None or context.ndim != 3 or context.shape[0] != b or context.shape[2] != self.context_dim or not context.shape[1]:
            raise ValueError('Prithvi context must have shape [batch, tokens, context_dim]')
        if noise_level.numel() != b:
            raise ValueError('Expected one alpha coefficient per example')
        # Existing diffusion wrapper passes a=sqrt(alpha_bar), not a timestep ID.
        alpha2 = noise_level.float().reshape(b).square().clamp(1e-6, 1-1e-6)
        log_snr = torch.log(alpha2) - torch.log1p(-alpha2)
        time = self.time_mlp(fourier(log_snr, 256))
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed.to(tokens.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, time, context, use_reentrant=False)
            else:
                tokens = block(tokens, time, context)
        shift, scale = self.final_modulation(time).chunk(2, dim=-1)
        return self.unpatchify(self.head(modulate(self.final_norm(tokens), shift, scale)))


def build(objective, device='cuda', initialize=True, architecture=None):
    """Build flow/epsilon MSE, or the earlier v/score adapter; no VAE."""
    from vscore_diffusion import VScoreDiffusion
    from dit_objectives import DiTObjective
    # Import the upstream encoder lazily; unit tests need neither weights nor downloads.
    from data import UPSTREAM
    from model.prithvi_encoder import PrithviEncoder
    del UPSTREAM, initialize  # Architecture initializes itself; resume loads checkpoint afterwards.
    denoiser = PrithviCrossDiT(**(architecture or {})).to(device)
    encoder = PrithviEncoder(freeze=True).to(device).eval()
    wrapper = DiTObjective if objective in ('flow', 'epsilon') else VScoreDiffusion
    return wrapper(denoiser, encoder, objective).to(device)
