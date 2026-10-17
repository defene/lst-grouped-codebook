"""Single-scale pretrained VQGAN and blog-derived drifting VQ for HLP v2.

Deployed into an isolated lst_restoration source snapshot, never the old run.
"""
import math
from pathlib import Path

import torch
from torch import nn

from .models import ConvolutionalRestorationCodec, ObservationAdapter, MeanAffine, _checkpoint_call
from .quantizers import VectorQuantizer, nearest_indices
from .config import file_digest


def drifting_energy(hidden, codes, tau, dimension_scale=True, normalize_potential=True):
    """Liu notebook's charge-normalized unordered pair energy, without self pairs.

    Image adaptation uses RMS distance and U/tau^2. Both are fixed analytic
    scalings, with no batch-estimated statistics or detached adaptive bandwidth.
    """
    if tau <= 0 or len(hidden) < 2 or len(codes) < 2:
        raise ValueError('Positive bandwidth and at least two particles required')
    scale = math.sqrt(hidden.shape[-1]) if dimension_scale else 1.0
    h, c = hidden.float() / scale, codes.float() / scale
    def potential(a, b):
        distance = torch.cdist(a, b, compute_mode='donot_use_mm_for_euclid_dist')
        r = distance + 1e-8
        result = torch.exp(-r / tau) * tau * (r + tau)
        return result / tau ** 2 if normalize_potential else result
    hh, cc, hc = potential(h, h), potential(c, c), potential(h, c)
    pp = (hh.sum() - hh.diagonal().sum()) / (2 * len(h) ** 2)
    nn_energy = (cc.sum() - cc.diagonal().sum()) / (2 * len(c) ** 2)
    pn = -hc.mean()
    return pp + nn_energy + pn, (pp, nn_energy, pn)


class DriftingVectorQuantizer(VectorQuantizer):
    def __init__(self, config):
        super().__init__(config['latent_dim'], config['codebook_size'])
        self.tau = float(config['drift_tau'])
        self.beta = float(config['drift_beta'])
        self.max_tokens = int(config['drift_tokens'])
        # Same range as the author's toy notebook, explicitly different from
        # the baseline's narrow +/-1/K initialization.
        nn.init.uniform_(self.embedding.weight, -.5, .5)

    def forward(self, latent):
        b, c, h, w = latent.shape
        with torch.autocast(device_type=latent.device.type, enabled=False):
            x = latent.float()
            flat = x.permute(0, 2, 3, 1).reshape(-1, c)
            indices = nearest_indices(flat, self.embedding.weight).reshape(b, h, w)
            codes = self.decode_indices(indices)
            # Author's implementation uses decoder(e), no straight-through
            # estimator. Encoder gradients come from energy; reconstruction
            # gradients train selected codes and the decoder directly.
            if self.training:
                sampled = flat[torch.randperm(len(flat), device=flat.device)[:self.max_tokens]]
                energy, components = drifting_energy(sampled, self.embedding.weight, self.tau)
            else:
                energy = x.new_zeros(())
                components = (energy, energy, energy)
            counts = torch.bincount(indices.flatten(), minlength=self.codebook_size).float()
            p = counts / counts.sum()
            diagnostics = dict(zip(('drift_pp', 'drift_nn', 'drift_pn'), components))
            diagnostics.update(drift_energy=energy, quantization_mse=(x - codes).square().mean(),
                               active_codes=(counts > 0).sum().float(),
                               perplexity=torch.exp(-(p * p.clamp_min(1e-20).log()).sum()),
                               latent_rms=x.square().mean().sqrt(),
                               codebook_rms=self.embedding.weight.square().mean().sqrt())
        return {'quantized': codes, 'indices': indices, 'loss': self.beta * energy,
                'diagnostics': {k: v.detach() for k, v in diagnostics.items()}}


class PretrainedVQGAN(ConvolutionalRestorationCodec):
    """Official CompVis ImageNet f16 K1024 AE, full trainable state loaded.

    Fine-tuning uses canonical VQ codebook+0.25 commitment and physical LST
    reconstruction, without adversarial/perceptual objectives. For R only,
    replicate input to RGB and average decoder output back to one channel.
    """
    def __init__(self, config):
        nn.Module.__init__(self)
        from .taming_model import Encoder, Decoder
        self.config = config
        c, task = config['model'], config['task']
        self.kind = c['kind']
        self.channels, self.grid, self.latent_dim = c['channels'], 16, 256
        self.checkpointing = bool(c.get('checkpointing', False))
        if (c['latent_dim'], c['grid'], c['codebook_size']) != (256, 16, 1024):
            raise ValueError('Official f16 K1024 architecture must be preserved')
        root = Path(c['pretrained_path'])
        for name, expected in c['pretrained_sha256'].items():
            if file_digest(root / name) != expected:
                raise ValueError('Pretrained asset changed: ' + name)
        import yaml
        spec = yaml.safe_load((root / 'model.yaml').read_text())['model']['params']
        if spec['embed_dim'] != 256 or spec['n_embed'] != 1024:
            raise ValueError('Wrong checkpoint configuration')
        self.encoder = Encoder(**spec['ddconfig'])
        self.decoder = Decoder(**spec['ddconfig'])
        self.quant_conv = nn.Conv2d(256, 256, 1)
        self.post_quant_conv = nn.Conv2d(256, 256, 1)
        self.quantizer = VectorQuantizer(256, 1024, float(c['commitment']))
        # Only the acquisition script loads the official Lightning container.
        # Runtime reads a tensors-only state extracted with recorded hashes.
        state = torch.load(root / 'autoencoder_state.pt', map_location='cpu', weights_only=True)
        used = set()
        for name in ('encoder', 'decoder', 'quant_conv', 'post_quant_conv'):
            selected = {k[len(name)+1:]: v for k, v in state.items() if k.startswith(name + '.')}
            getattr(self, name).load_state_dict(selected, strict=True)
            used.update(name + '.' + k for k in selected)
        self.quantizer.embedding.load_state_dict({'weight': state['quantize.embedding.weight']}, strict=True)
        used.add('quantize.embedding.weight')
        if used != set(state):
            raise ValueError('Unaccounted autoencoder tensors: ' + str(set(state) - used))
        self.loaded_tensor_count = len(used)
        scale, center = task['mean_scale_celsius'], task['mean_center_celsius']
        self.input_adapter = ObservationAdapter(self.channels, scale, center)
        self.decoder_mean = MeanAffine(512, scale, center)

    def _encode(self, x):
        if self.channels == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.quant_conv(self.encoder(x))

    def _decode(self, latent, mean):
        from functools import partial
        from .taming_model import nonlinearity, ResnetBlock
        d, check = self.decoder, self.training and self.checkpointing
        def call(module, value):
            # Official CompVis residual blocks require the explicit time
            # embedding argument even for an unconditional autoencoder.
            function = partial(module, temb=None) if isinstance(module, ResnetBlock) else module
            return _checkpoint_call(function, value, check)
        h = self.decoder_mean(d.conv_in(self.post_quant_conv(latent)), mean)
        for module in (d.mid.block_1, d.mid.attn_1, d.mid.block_2):
            h = call(module, h)
        for level in reversed(range(d.num_resolutions)):
            for i, block in enumerate(d.up[level].block):
                h = call(block, h)
                if len(d.up[level].attn):
                    h = _checkpoint_call(d.up[level].attn[i], h, check)
            if level != 0:
                h = d.up[level].upsample(h)
        result = d.conv_out(nonlinearity(d.norm_out(h))).float()
        return result.mean(1, keepdim=True) if self.channels == 1 else result

    def model_metadata(self):
        return {**super().model_metadata(), 'official_tensors_strictly_loaded': self.loaded_tensor_count,
                'initialization': 'CompVis ImageNet f16 K1024', 'adversarial_finetuning': False,
                'r_rgb_adapter': 'replicate input / mean output', 'equal_parameter_claim': False}
