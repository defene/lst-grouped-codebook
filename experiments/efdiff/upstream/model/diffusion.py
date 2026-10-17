"""Diffusion models for LST super-resolution.

Contains two diffusion variants sharing a common base:
  - GaussianDiffusion (SR3): DDPM-style noise prediction, 500-1000 timesteps
  - ResShiftDiffusion: x_0-prediction in residual space, ~15 timesteps

Both support optional Prithvi-EO cross-attention conditioning.
"""
import math
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from functools import partial

from .unet import default


# ============================================================================
# Beta schedule (used by SR3 only)
# ============================================================================

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':
        betas = 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# ============================================================================
# Base diffusion
# ============================================================================

class BaseDiffusion(nn.Module):
    """Common interface for SR3 and ResShift diffusion models."""

    def __init__(self, denoise_fn, image_size, channels, loss_type,
                 conditional, prithvi_encoder=None, concat_hls=False):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.prithvi_encoder = prithvi_encoder
        self.loss_type = loss_type
        self.conditional = conditional
        self.concat_hls = concat_hls
        self._schedule_initialized = False

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(device)
        else:
            raise NotImplementedError(f'Unknown loss type: {self.loss_type}')

    def _check_schedule(self):
        """Guard against using the model before noise schedule is set."""
        if not self._schedule_initialized:
            raise RuntimeError(
                'Noise schedule not initialized. '
                'Call set_new_noise_schedule() before training or inference.')

    def _encode_hls(self, hls):
        """Encode HLS via Prithvi. Returns None if no encoder."""
        if self.prithvi_encoder is None or hls is None:
            return None
        return self.prithvi_encoder(hls)

    def _call_denoise(self, x_input, noise_level, context=None):
        """Call denoise_fn, passing context only if encoder is present."""
        if self.prithvi_encoder is not None and context is not None:
            return self.denoise_fn(x_input, noise_level, context=context)
        return self.denoise_fn(x_input, noise_level)

    def _denoise_conditional(self, x, condition_x, noise_level, context=None):
        """Denoise with optional conditional concatenation (shared by all paths)."""
        if condition_x is not None:
            return self._call_denoise(
                torch.cat([condition_x, x], dim=1), noise_level, context=context)
        return self._call_denoise(x, noise_level, context=context)

    def _build_condition(self, sr, hls):
        """Build condition tensor, concatenating HLS if concat_hls mode."""
        if self.concat_hls and hls is not None:
            return torch.cat([sr, hls], dim=1)
        return sr

    @torch.no_grad()
    def super_resolution(self, x_in, continuous=False, verbose=True,
                         ddim_steps=None):
        if isinstance(x_in, dict):
            condition = self._build_condition(
                x_in['SR'], x_in.get('HLS', None))
            context = self._encode_hls(x_in.get('HLS', None))
        else:
            condition = x_in
            context = None
        if ddim_steps is not None and hasattr(self, 'ddim_sample_loop'):
            return self.ddim_sample_loop(
                condition, ddim_steps, context=context,
                continuous=continuous, verbose=verbose)
        return self.p_sample_loop(condition, context=context,
                                  continuous=continuous, verbose=verbose)

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)


# ============================================================================
# SR3 Gaussian diffusion (noise prediction, 500-1000 steps)
# ============================================================================

class GaussianDiffusion(BaseDiffusion):
    """SR3-style Gaussian diffusion with optional Prithvi-EO conditioning.

    Predicts noise epsilon, then derives x_0 via DDPM posterior.
    """

    def __init__(self, denoise_fn, image_size, channels=3, loss_type='l1',
                 conditional=True, schedule_opt=None, prithvi_encoder=None,
                 concat_hls=False):
        super().__init__(denoise_fn, image_size, channels, loss_type,
                         conditional, prithvi_encoder, concat_hls)

    def set_new_noise_schedule(self, schedule_opt, device):
        for key in ('schedule', 'n_timestep', 'linear_start', 'linear_end'):
            if key not in schedule_opt:
                raise KeyError(f'schedule_opt missing required key: {key}')

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.register_buffer('sqrt_alphas_cumprod_prev',
                             to_torch(np.sqrt(np.append(1., alphas_cumprod))))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self._schedule_initialized = True

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised=True,
                        condition_x=None, context=None):
        batch_size = x.shape[0]
        noise_level = self.sqrt_alphas_cumprod_prev[t + 1].view(
            1, 1).expand(batch_size, 1).to(x.dtype)

        noise_pred = self._denoise_conditional(
            x, condition_x, noise_level, context=context)

        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon = x_recon.clamp(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None, context=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised,
            condition_x=condition_x, context=context)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, context=None, continuous=False, verbose=True):
        self._check_schedule()
        device = self.betas.device
        sample_inter = max(1, self.num_timesteps // 10)

        timesteps = reversed(range(0, self.num_timesteps))
        if verbose:
            timesteps = tqdm(timesteps, desc='sampling loop time step',
                             total=self.num_timesteps)

        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img if continuous else None
            for i in timesteps:
                img = self.p_sample(img, i, context=context)
                if continuous and i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            noise_shape = (x.shape[0], self.channels, x.shape[2], x.shape[3])
            img = torch.randn(noise_shape, device=device)
            ret_img = x if continuous else None
            for i in timesteps:
                img = self.p_sample(
                    img, i, condition_x=x, context=context)
                if continuous and i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)

        return ret_img if continuous else img

    @torch.no_grad()
    def sample(self, batch_size=1, continuous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size), continuous=continuous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        device = x_start.device

        # Per-sample timestep sampling (reduces loss variance)
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=device)
        low = self.sqrt_alphas_cumprod_prev[t - 1]   # (b,)
        high = self.sqrt_alphas_cumprod_prev[t]       # (b,)
        continuous_sqrt_alpha_cumprod = (
            torch.rand(b, device=device) * (high - low) + low
        ).unsqueeze(1)  # (b, 1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
            noise=noise)

        context = self._encode_hls(x_in.get('HLS', None))
        condition = self._build_condition(
            x_in['SR'], x_in.get('HLS', None)) if self.conditional else None

        x_recon = self._denoise_conditional(
            x_noisy, condition, continuous_sqrt_alpha_cumprod, context=context)

        loss = self.loss_func(noise, x_recon)
        return loss

    # ---- DDIM sampling (inference only) ----

    def _make_ddim_timesteps(self, ddim_steps):
        """Create uniformly spaced subsequence of timesteps."""
        c = self.num_timesteps // ddim_steps
        return list(reversed(range(0, self.num_timesteps, c)))

    @torch.no_grad()
    def ddim_sample(self, x, t, t_prev, clip_denoised=True,
                    condition_x=None, context=None):
        batch_size = x.shape[0]
        noise_level = self.sqrt_alphas_cumprod_prev[t + 1].view(
            1, 1).expand(batch_size, 1).to(x.dtype)

        eps_pred = self._denoise_conditional(
            x, condition_x, noise_level, context=context)

        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else \
            torch.tensor(1.0, device=x.device, dtype=self.alphas_cumprod.dtype)

        x0_pred = (x - (1 - alpha_t).sqrt() * eps_pred) / alpha_t.sqrt()
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1., 1.)

        # DDIM deterministic step (eta=0)
        return alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * eps_pred

    @torch.no_grad()
    def ddim_sample_loop(self, x_in, ddim_steps, context=None,
                         continuous=False, verbose=True):
        self._check_schedule()
        device = self.betas.device
        timesteps = self._make_ddim_timesteps(ddim_steps)

        if self.conditional:
            x = x_in
            noise_shape = (x.shape[0], self.channels, x.shape[2], x.shape[3])
            img = torch.randn(noise_shape, device=device)
        else:
            shape = x_in
            x = None
            img = torch.randn(shape, device=device)

        pairs = list(zip(timesteps[:-1], timesteps[1:])) + [(timesteps[-1], -1)]
        if verbose:
            pairs = tqdm(pairs, desc='DDIM sampling')

        for t, t_prev in pairs:
            img = self.ddim_sample(img, t, t_prev, condition_x=x, context=context)

        return img


# ============================================================================
# ResShift diffusion (x_0 prediction, ~15 steps)
# ============================================================================

class ResShiftDiffusion(BaseDiffusion):
    """Few-step residual diffusion with optional Prithvi-EO conditioning.

    Operates in residual space (x_0 = HR - LR). Predicts x_0 directly.
    Forward: x_t = (1 - eta_t)*x_0 + kappa*sqrt(eta_t)*eps
    Supports exponential schedule (official) and polynomial schedule (legacy).
    """

    def __init__(self, denoise_fn, image_size, channels=1, loss_type='l2',
                 conditional=True, schedule_opt=None, kappa=1.0,
                 prithvi_encoder=None, concat_hls=False):
        super().__init__(denoise_fn, image_size, channels, loss_type,
                         conditional, prithvi_encoder, concat_hls)
        self.kappa = kappa
        if schedule_opt is not None:
            self.set_new_noise_schedule(schedule_opt, device='cpu')

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        T = int(schedule_opt['n_timestep'])
        schedule = schedule_opt.get('schedule', 'polynomial')

        if schedule == 'exponential':
            power = float(schedule_opt.get('schedule_power', 0.3))
            etas_end = float(schedule_opt.get('etas_end', 0.99))
            min_noise = float(schedule_opt.get('min_noise_level', 0.04))

            etas_start = min(min_noise / max(self.kappa, 1e-8),
                             min_noise, math.sqrt(0.001))
            increaser = math.exp(
                1.0 / max(T - 1, 1) * math.log(etas_end / etas_start))
            base = np.ones(T, dtype=np.float64) * increaser
            power_ts = np.linspace(0, 1, T, endpoint=True) ** power
            power_ts *= (T - 1)
            sqrt_etas_inner = np.power(base, power_ts) * etas_start
            etas_inner = sqrt_etas_inner ** 2
            eta = np.concatenate([[0.0], etas_inner])
        else:  # polynomial (legacy)
            p = float(schedule_opt.get('schedule_power', 3.0))
            t_arr = np.arange(0, T + 1, dtype=np.float64)
            eta = (t_arr / T) ** p

        self.num_timesteps = T
        self.register_buffer('eta', to_torch(eta))
        self.register_buffer('sqrt_eta', to_torch(np.sqrt(eta)))

        alpha = eta[1:] - eta[:-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            post_var = (self.kappa ** 2) * eta[:-1] * alpha / (eta[1:] + 1e-8)
        post_var = np.maximum(post_var, 1e-20)
        self.register_buffer('posterior_log_variance',
                             to_torch(np.log(post_var)))

        # Input normalization scale: 1 / sqrt(κ²η_t + 1)
        input_scale = 1.0 / np.sqrt(self.kappa ** 2 * eta + 1.0)
        self.register_buffer('input_scale', to_torch(input_scale))

        self._schedule_initialized = True

    def q_sample(self, x_start, t_index, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        eta_t = self.eta[t_index]
        if eta_t.dim() > 0:
            eta_t = eta_t.view(-1, 1, 1, 1)
        return (1.0 - eta_t) * x_start + self.kappa * eta_t.sqrt() * noise

    def p_mean_variance(self, x, t, condition_x=None, context=None):
        batch_size = x.shape[0]
        noise_level = self.sqrt_eta[t].view(1, 1).expand(batch_size, 1).to(x.dtype)

        # Input normalization: scale noisy input by 1/sqrt(κ²η_t + 1)
        scale = self.input_scale[t]
        x_scaled = x * scale

        x0_pred = self._denoise_conditional(
            x_scaled, condition_x, noise_level, context=context)

        x0_pred = x0_pred.clamp(-1.0, 1.0)

        eta_t      = self.eta[t]
        eta_t_prev = self.eta[t - 1]
        alpha_t    = eta_t - eta_t_prev
        denom      = eta_t + 1e-8

        coef_xt = eta_t_prev / denom
        coef_x0 = alpha_t / denom

        model_mean = coef_xt * x + coef_x0 * x0_pred
        posterior_log_var = self.posterior_log_variance[t - 1]
        return model_mean, posterior_log_var

    @torch.no_grad()
    def p_sample(self, x, t, condition_x=None, context=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, condition_x=condition_x, context=context)
        noise = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, context=None, continuous=False, verbose=True):
        self._check_schedule()
        y_lr = x_in
        noise_shape = (y_lr.shape[0], self.channels, y_lr.shape[2], y_lr.shape[3])
        img = self.kappa * torch.randn(noise_shape, device=y_lr.device)
        sample_inter = max(1, self.num_timesteps // 10)
        ret_img = y_lr

        timesteps = list(reversed(range(1, self.num_timesteps + 1)))
        if verbose:
            timesteps = tqdm(timesteps, desc='ResShift sampling',
                             total=self.num_timesteps)

        for t in timesteps:
            img = self.p_sample(img, t, condition_x=y_lr, context=context)
            if continuous and (self.num_timesteps - t) % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)

        if continuous:
            return ret_img
        return img

    # ---- DDIM-style step-skipping (deterministic, fewer steps) ----

    def _ddim_step(self, x, t, t_prev, condition_x=None, context=None):
        """Deterministic step from t to t_prev, skipping intermediate steps."""
        batch_size = x.shape[0]
        noise_level = self.sqrt_eta[t].view(1, 1).expand(batch_size, 1).to(x.dtype)

        scale = self.input_scale[t]
        x_scaled = x * scale

        x0_pred = self._denoise_conditional(
            x_scaled, condition_x, noise_level, context=context)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        eta_t = self.eta[t]
        eta_prev = self.eta[t_prev]
        alpha_t = eta_t - eta_prev
        denom = eta_t + 1e-8

        # Deterministic update (no noise)
        return (eta_prev / denom) * x + (alpha_t / denom) * x0_pred

    @torch.no_grad()
    def ddim_sample_loop(self, x_in, ddim_steps, context=None,
                         continuous=False, verbose=True):
        """ResShift DDIM: skip steps from the original schedule, deterministic."""
        self._check_schedule()
        y_lr = x_in
        noise_shape = (y_lr.shape[0], self.channels, y_lr.shape[2], y_lr.shape[3])
        img = self.kappa * torch.randn(noise_shape, device=y_lr.device)

        # Select subset of timesteps (uniformly spaced from original schedule)
        T = self.num_timesteps
        if ddim_steps >= T:
            timestep_seq = list(range(1, T + 1))
        else:
            indices = np.linspace(0, T, ddim_steps + 1, dtype=np.float64)
            timestep_seq = [int(round(x)) for x in indices[1:]]
            timestep_seq = sorted(set(timestep_seq))

        pairs = list(zip(list(reversed(timestep_seq)), list(reversed(timestep_seq))[1:] + [0]))
        if verbose:
            pairs = tqdm(pairs, desc=f'ResShift DDIM ({len(pairs)} steps)')

        for t, t_prev in pairs:
            img = self._ddim_step(img, t, t_prev, condition_x=y_lr, context=context)

        return img

    @torch.no_grad()
    def sample(self, batch_size=1, continuous=False):
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        device = self.eta.device
        img = self.kappa * torch.randn(shape, device=device)
        for t in reversed(range(1, self.num_timesteps + 1)):
            img = self.p_sample(img, t)
        return img

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        y_cond  = self._build_condition(
            x_in['SR'], x_in.get('HLS', None))
        b       = x_start.shape[0]
        device  = x_start.device

        t = torch.randint(1, self.num_timesteps + 1, (b,), device=device)

        sqrt_eta_low  = self.sqrt_eta[t - 1]
        sqrt_eta_high = self.sqrt_eta[t]
        sqrt_eta_sample = (
            torch.rand(b, device=device) * (sqrt_eta_high - sqrt_eta_low)
            + sqrt_eta_low
        ).view(b, 1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        eta4d = (sqrt_eta_sample ** 2).view(b, 1, 1, 1)
        sqrt_eta4d = sqrt_eta_sample.view(b, 1, 1, 1)
        x_noisy = (1.0 - eta4d) * x_start + self.kappa * sqrt_eta4d * noise

        # Input normalization: scale noisy input by 1/sqrt(κ²η_t + 1)
        input_scale = 1.0 / (self.kappa ** 2 * eta4d + 1.0).sqrt()
        x_noisy_scaled = x_noisy * input_scale

        context = self._encode_hls(x_in.get('HLS', None))
        condition = y_cond if self.conditional else None

        x0_pred = self._denoise_conditional(
            x_noisy_scaled, condition, sqrt_eta_sample, context=context)

        return self.loss_func(x0_pred, x_start)
