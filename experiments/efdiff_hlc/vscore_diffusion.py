"""Versioned VP v/score experiment; historical diffusion implementations are untouched."""
from pathlib import Path
import sys
import torch
from torch import nn


def coefficients(t, alpha_bar):
    a = alpha_bar[t].sqrt().view(-1, 1, 1, 1)
    s = (1 - alpha_bar[t]).sqrt().view(-1, 1, 1, 1)
    return a, s


def predictions(raw, xt, a, s, objective):
    raw, xt = raw.float(), xt.float()
    if objective == 'v':
        return a * xt - s * raw, s * xt + a * raw
    if objective == 'score':
        eps = -s * raw
        return (xt - s * eps) / a, eps
    raise ValueError(objective)


def objective_loss(raw, x0, eps, a, s, mask, objective):
    if objective == 'v':
        error = raw.float() - (a * eps - s * x0)
    elif objective == 'score':
        # sigma^2 weighted DSM; network itself outputs the raw score.
        error = s * raw.float() + eps
    else:
        raise ValueError(objective)
    valid = mask.bool()
    count = valid.flatten(1).sum(1)
    if (count == 0).any():
        raise ValueError('Empty target mask')
    return (error.square().masked_fill(~valid, 0).flatten(1).sum(1) / count).mean()


def time_grid(steps, total=1000):
    if not 1 <= steps <= total:
        raise ValueError('NFE must be between 1 and total training timesteps')
    grid = torch.linspace(total - 1, 0, steps).round().long().tolist()
    assert len(set(grid)) == steps
    return list(zip(grid, grid[1:] + [-1]))


class VScoreDiffusion(nn.Module):
    def __init__(self, denoiser, encoder, objective):
        super().__init__()
        if objective not in ('v', 'score'):
            raise ValueError(objective)
        self.denoise_fn, self.prithvi_encoder = denoiser, encoder
        self.objective = objective
        # Same endpoints as the historical epsilon model; calculate in fp64.
        betas = torch.linspace(1e-6, .01, 1000, dtype=torch.float64)
        self.register_buffer('alpha_bar', (1 - betas).cumprod(0).float())

    def train(self, mode=True):
        super().train(mode)
        if self.prithvi_encoder is not None:
            self.prithvi_encoder.eval()
        return self

    def encode(self, hls):
        if self.prithvi_encoder is None:
            return None
        with torch.no_grad():
            return self.prithvi_encoder(hls)

    def denoise(self, xt, condition, a, context):
        x = torch.cat([condition, xt], dim=1)
        level = a.reshape(-1, 1).expand(xt.shape[0], 1)
        if context is None:
            return self.denoise_fn(x, level)
        return self.denoise_fn(x, level, context=context)

    def forward(self, batch, t=None, eps=None):
        x0 = batch['HR'].float()
        if t is None:
            t = torch.randint(0, len(self.alpha_bar), (len(x0),), device=x0.device)
        if eps is None:
            eps = torch.randn_like(x0)
        a, s = coefficients(t, self.alpha_bar)
        xt = a * x0 + s * eps
        raw = self.denoise(xt, batch['SR'], a, self.encode(batch.get('HLS')))
        return objective_loss(raw, x0, eps, a, s, batch['mask'], self.objective)

    @torch.no_grad()
    def sample(self, batch, steps, noise):
        context = self.encode(batch.get('HLS'))
        xt = noise.float().clone()
        for t, prev in time_grid(steps, len(self.alpha_bar)):
            at = self.alpha_bar[t].sqrt().reshape(1, 1, 1, 1)
            st = (1 - self.alpha_bar[t]).sqrt().reshape(1, 1, 1, 1)
            raw = self.denoise(xt, batch['SR'], at, context)
            x0, eps = predictions(raw, xt, at, st, self.objective)
            # Shared clipping policy; recompute epsilon to preserve xt=a*x0+s*eps.
            x0 = x0.clamp(-1, 1)
            eps = (xt - at * x0) / st
            ap = self.alpha_bar[prev] if prev >= 0 else xt.new_tensor(1.)
            xt = ap.sqrt() * x0 + (1 - ap).sqrt() * eps
        return xt


def build(objective, device='cuda', initialize=True):
    here = Path(__file__).resolve().parent
    upstream = here.parent / 'efdiff' / 'upstream'
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from model.unet import UNet
    from model.prithvi_encoder import PrithviEncoder
    from model.factory import init_weights
    unet = UNet(in_channel=2, out_channel=1, norm_groups=16, inner_channel=64,
                channel_mults=[1, 2, 4, 8, 16], attn_res=[14], res_blocks=2,
                dropout=0, image_size=224, context_dim=1024,
                cross_attn_heads=8, cross_attn_res=[14]).to(device)
    assert sum(p.numel() for p in unet.parameters()) == 259049729
    if initialize:
        init_weights(unet, init_type='orthogonal')
    encoder = PrithviEncoder(freeze=True).to(device).eval()
    assert not any(p.requires_grad for p in encoder.parameters())
    return VScoreDiffusion(unet, encoder, objective).to(device)
