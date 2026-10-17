"""Pixel DiT with independently encoded calendar/geolocation adaLN conditioning.

This is a new architecture. Old DiT weights/runs and their build() stay frozen.
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from dit_model import PrithviCrossDiT, fourier, modulate


class CalendarGeoCondition(nn.Module):
    def __init__(self, width, condition_dim=128, month_harmonics=3, geo_bands=3):
        super().__init__()
        if min(condition_dim, month_harmonics, geo_bands) <= 0 or month_harmonics > 5:
            raise ValueError('Use positive widths/bands and 1..5 monthly harmonics')
        self.register_buffer('month_frequencies', torch.arange(1, month_harmonics+1).float())
        self.register_buffer('geo_frequencies', 2.**torch.arange(geo_bands).float())
        self.month_encoder = nn.Sequential(nn.Linear(2*month_harmonics, condition_dim), nn.SiLU(),
                                           nn.Linear(condition_dim, condition_dim))
        self.geo_encoder = nn.Sequential(nn.Linear(3+6*geo_bands, condition_dim), nn.SiLU(),
                                         nn.Linear(condition_dim, condition_dim))
        # Disjoint input blocks (time, month, geo); learned mixing is intentional.
        self.fusion = nn.Sequential(nn.Linear(width+2*condition_dim, width), nn.SiLU(), nn.Linear(width, width))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def month_features(self, month):
        month = month.float()
        if month.ndim != 1 or not torch.isfinite(month).all() or not ((month >= 1) & (month <= 12) & (month == month.round())).all():
            raise ValueError('Target calendar month must have shape [B] and integer values 1..12')
        phase = (month[:,None]-1) * (2*math.pi/12) * self.month_frequencies
        return torch.cat((phase.sin(), phase.cos()), dim=-1)

    def geo_features(self, latlon):
        latlon = latlon.float()
        if latlon.ndim != 2 or latlon.shape[1] != 2 or not torch.isfinite(latlon).all():
            raise ValueError('Geolocation must be finite [B,2] latitude/longitude in degrees')
        if not ((latlon[:,0].abs() <= 90) & (latlon[:,1].abs() <= 180)).all():
            raise ValueError('Latitude/longitude outside valid degree range')
        lat, lon = torch.deg2rad(latlon).unbind(-1)
        xyz = torch.stack((lat.cos()*lon.cos(), lat.cos()*lon.sin(), lat.sin()), dim=-1)
        phase = xyz[:,:,None] * self.geo_frequencies * math.pi
        return torch.cat((xyz, phase.sin().flatten(1), phase.cos().flatten(1)), dim=-1)

    def forward(self, diffusion_time, month, latlon):
        if len(month) != len(diffusion_time) or len(latlon) != len(diffusion_time):
            raise ValueError('Conditioning batch sizes disagree')
        month_embed = self.month_encoder(self.month_features(month))
        geo_embed = self.geo_encoder(self.geo_features(latlon))
        return self.fusion(torch.cat((diffusion_time, month_embed, geo_embed), dim=-1))


class TimeGeoDiT(PrithviCrossDiT):
    def __init__(self, *, condition_dim=128, month_harmonics=3, geo_bands=3, **architecture):
        super().__init__(**architecture)
        self.calendar_geo = CalendarGeoCondition(self.hidden_dim, condition_dim, month_harmonics, geo_bands)

    def forward(self, x, noise_level, context=None, *, month=None, geo_latlon=None):
        b = x.shape[0]
        if month is None or geo_latlon is None:
            raise ValueError('TimeGeoDiT requires target month and crop-center geo_latlon')
        if x.shape[1:] != (2, self.image_size, self.image_size):
            raise ValueError('Expected concatenated LST condition/noisy residual')
        if context is None or context.ndim != 3 or context.shape[0] != b or context.shape[2] != self.context_dim or not context.shape[1]:
            raise ValueError('Expected Prithvi context [B,tokens,context_dim]')
        if noise_level.numel() != b:
            raise ValueError('Expected one noise level per example')
        alpha2 = noise_level.float().reshape(b).square().clamp(1e-6, 1-1e-6)
        log_snr = torch.log(alpha2)-torch.log1p(-alpha2)
        diffusion_time = self.time_mlp(fourier(log_snr, 256))
        condition = self.calendar_geo(diffusion_time, month, geo_latlon)
        tokens = self.patch_embed(x).flatten(2).transpose(1,2)
        if self.pos_embed is not None:
            tokens = tokens+self.pos_embed.to(tokens.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, condition, context, use_reentrant=False)
            else:
                tokens = block(tokens, condition, context)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        return self.unpatchify(self.head(modulate(self.final_norm(tokens), shift, scale)))


def build(objective, device='cuda', initialize=True, architecture=None, corrected_hls=False):
    from dit_timegeo_objectives import TimeGeoObjective
    from data import UPSTREAM
    from model.prithvi_encoder import PrithviEncoder
    del UPSTREAM, initialize
    denoiser = TimeGeoDiT(**(architecture or {})).to(device)
    if corrected_hls:
        from prithvi_input_encoder import MaskAwarePrithviEncoder
        encoder = MaskAwarePrithviEncoder(freeze=True).to(device).eval()
    else:
        encoder = PrithviEncoder(freeze=True).to(device).eval()
    return TimeGeoObjective(denoiser, encoder, objective).to(device)
