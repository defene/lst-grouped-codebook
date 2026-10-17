"""Independent linear conditional flow matching and VP epsilon prediction, both MSE."""
import torch
from torch import nn
from vscore_diffusion import time_grid


def masked_mse(prediction, target, mask):
    valid = mask.bool()
    count = valid.flatten(1).sum(1)
    if (count == 0).any():
        raise ValueError('Empty reference mask')
    error = (prediction.float() - target.float()).square().masked_fill(~valid, 0)
    return (error.flatten(1).sum(1) / count).mean()


class DiTObjective(nn.Module):
    def __init__(self, denoiser, encoder, objective):
        super().__init__()
        if objective not in ('flow', 'epsilon'):
            raise ValueError(objective)
        self.denoise_fn, self.prithvi_encoder, self.objective = denoiser, encoder, objective
        betas = torch.linspace(1e-6, .01, 1000, dtype=torch.float64)
        self.register_buffer('alpha_bar', (1-betas).cumprod(0).float())

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

    def denoise(self, xt, condition, level, context):
        return self.denoise_fn(torch.cat((condition, xt), dim=1), level.reshape(-1,1), context=context)

    @staticmethod
    def flow_level(t):
        # Keep DiT's log-SNR embedding interface. For the linear path,
        # SNR=t^2/(1-t)^2, so its equivalent VP alpha is t/sqrt(t^2+(1-t)^2).
        return t / torch.sqrt(t.square() + (1-t).square())

    def training_pair(self, x0, noise, t):
        if self.objective == 'flow':
            time = t.float().reshape(-1,1,1,1)
            return (1-time)*noise + time*x0, x0-noise, self.flow_level(time)
        a = self.alpha_bar[t].sqrt().reshape(-1,1,1,1)
        s = (1-self.alpha_bar[t]).sqrt().reshape(-1,1,1,1)
        return a*x0+s*noise, noise, a

    def forward(self, batch, t=None, eps=None):
        x0 = batch['HR'].float()
        if t is None:
            t = (torch.rand(len(x0),device=x0.device) if self.objective == 'flow' else
                 torch.randint(0,1000,(len(x0),),device=x0.device))
        noise = torch.randn_like(x0) if eps is None else eps.float()
        xt, target, level = self.training_pair(x0, noise, t)
        pred = self.denoise(xt, batch['SR'], level, self.encode(batch.get('HLS')))
        return masked_mse(pred, target, batch['mask'])

    @torch.no_grad()
    def sample(self, batch, steps, noise):
        if not isinstance(steps,int) or steps < 1:
            raise ValueError('steps must be a positive integer')
        context = self.encode(batch.get('HLS'))
        xt = noise.float().clone()
        if self.objective == 'flow':
            # N Euler evaluations at 0,1/N,...,(N-1)/N; endpoint is exactly t=1.
            for step in range(steps):
                t = xt.new_full((len(xt),),step/steps)
                velocity = self.denoise(xt,batch['SR'],self.flow_level(t),context)
                xt = xt + velocity.float()/steps
            return xt.clamp(-1,1)
        for t,prev in time_grid(steps,len(self.alpha_bar)):
            a = self.alpha_bar[t].sqrt()
            s = (1-self.alpha_bar[t]).sqrt()
            predicted_noise = self.denoise(xt,batch['SR'],a.expand(len(xt)),context).float()
            x0 = ((xt-s*predicted_noise)/a).clamp(-1,1)
            predicted_noise = (xt-a*x0)/s
            ap = self.alpha_bar[prev] if prev>=0 else xt.new_tensor(1.)
            xt = ap.sqrt()*x0 + (1-ap).sqrt()*predicted_noise
        return xt
