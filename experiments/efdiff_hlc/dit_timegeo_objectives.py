"""Metadata propagation for both training and every inference step; unchanged L2 targets."""
import torch
from dit_objectives import DiTObjective, masked_mse
from vscore_diffusion import time_grid


class TimeGeoObjective(DiTObjective):
    def __init__(self, denoiser, encoder, objective):
        if objective not in ('flow', 'epsilon', 'v'):
            raise ValueError(objective)
        super().__init__(denoiser, encoder, 'epsilon' if objective == 'v' else objective)
        self.objective = objective

    def training_pair(self, x0, noise, t):
        xt, target, level = super().training_pair(x0, noise, t)
        if self.objective == 'v':
            sigma = (1-self.alpha_bar[t]).sqrt().reshape(-1,1,1,1)
            target = level*noise-sigma*x0
        return xt, target, level

    def encode_batch(self, batch):
        if self.prithvi_encoder is None:
            return None
        with torch.no_grad():
            if getattr(self.prithvi_encoder, 'requires_hls_valid', False):
                return self.prithvi_encoder(batch['HLS'], batch['HLS_valid'])
            return self.prithvi_encoder(batch['HLS'])

    def denoise_conditioned(self, xt, batch, level, context):
        if 'month' not in batch or 'geo_latlon' not in batch:
            raise ValueError('Missing target month or crop-center geo_latlon')
        return self.denoise_fn(torch.cat((batch['SR'], xt), dim=1), level.reshape(-1,1),
                              context=context, month=batch['month'], geo_latlon=batch['geo_latlon'])

    def forward(self, batch, t=None, eps=None):
        x0 = batch['HR'].float()
        if t is None:
            t = (torch.rand(len(x0), device=x0.device) if self.objective == 'flow' else
                 torch.randint(0,1000,(len(x0),),device=x0.device))
        noise = torch.randn_like(x0) if eps is None else eps.float()
        xt, target, level = self.training_pair(x0, noise, t)
        pred = self.denoise_conditioned(xt, batch, level, self.encode_batch(batch))
        return masked_mse(pred, target, batch['mask'])

    @torch.no_grad()
    def sample(self, batch, steps, noise):
        if not isinstance(steps,int) or steps < 1:
            raise ValueError('steps must be a positive integer')
        context = self.encode_batch(batch)
        xt = noise.float().clone()
        if self.objective == 'flow':
            for step in range(steps):
                t = xt.new_full((len(xt),), step/steps)
                velocity = self.denoise_conditioned(xt,batch,self.flow_level(t),context)
                xt = xt+velocity.float()/steps
            return xt.clamp(-1,1)
        for t,prev in time_grid(steps,len(self.alpha_bar)):
            a,s = self.alpha_bar[t].sqrt(), (1-self.alpha_bar[t]).sqrt()
            raw = self.denoise_conditioned(xt,batch,a.expand(len(xt)),context).float()
            x0 = (a*xt-s*raw if self.objective == 'v' else (xt-s*raw)/a).clamp(-1,1)
            predicted_noise = (xt-a*x0)/s
            ap = self.alpha_bar[prev] if prev>=0 else xt.new_tensor(1.)
            xt = ap.sqrt()*x0+(1-ap).sqrt()*predicted_noise
        return xt
