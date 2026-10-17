"""A07: pretrained SD KL-VAE, identical physical losses, reversible LST RGB adapters."""
from pathlib import Path
import math
import torch
from torch import nn
from diffusers import AutoencoderKL
from eo_data.core import sha256_file
from eo_denoise import engine
from eo_denoise.models import CrossViewModel
from eo_denoise.losses import masked_mean, gradient_loss
from experiments.paper1.lst_residual.runtime import visible_mean, spatial_mse

HERE = Path(__file__).resolve().parent


class Representation(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.kind = c['representation']['kind']
        self.std_c = c['representation']['std_celsius']
        self.scale_c = c['representation']['residual_std_celsius']
        self.bound_c = 120.0  # Difference of two valid values in source support [-50, 70] C.
        self.warp_norm = math.asinh(self.bound_c / self.scale_c)
        self.register_buffer('vertices', torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                                                      [0,1,1],[0,0,1],[1,0,1],[1,1,1]], dtype=torch.float32))

    def scalar(self, residual):
        r = residual.float() * self.std_c
        if self.kind == 'linear_gray':
            return (r / self.bound_c + 1) / 2
        return (torch.asinh(r / self.scale_c) / self.warp_norm + 1) / 2

    def unscalar(self, u):
        v = 2*u.float()-1
        if self.kind == 'linear_gray':
            return v*self.bound_c/self.std_c
        return torch.sinh(v*self.warp_norm)*self.scale_c/self.std_c

    def encode(self, residual):
        u = self.scalar(residual)
        if self.kind != 'asinh_rgb':
            return (2*u-1).expand(-1,3,-1,-1)
        if bool(((u < -1e-6) | (u > 1+1e-6)).any()):
            raise ValueError('LST residual exceeds declared 120 C support; no silent clipping')
        q = u[:,0].clamp(0,1)*7
        k = q.floor().long().clamp(max=6)
        f = (q-k).unsqueeze(-1)
        rgb = self.vertices[k] + f*(self.vertices[k+1]-self.vertices[k])
        return (2*rgb-1).permute(0,3,1,2)

    def decode(self, encoded):
        rgb = (encoded.float()+1)/2
        if self.kind != 'asinh_rgb':
            u_raw = rgb.mean(1,keepdim=True)
            u = u_raw.clamp(0,1)
            distance = (rgb-u).square().mean(1,keepdim=True)
        else:
            x = rgb.permute(0,2,3,1).unsqueeze(-2)
            a = self.vertices[:-1]
            d = self.vertices[1:]-a
            t = ((x-a)*d).sum(-1).clamp(0,1)
            projected = a+t[...,None]*d
            distances = (x-projected).square().sum(-1)
            best = distances.argmin(-1,keepdim=True)
            u = ((best+t.gather(-1,best))/7).permute(0,3,1,2)
            distance = distances.gather(-1,best).permute(0,3,1,2)/3
        return self.unscalar(u), distance, ((u <= 0) | (u >= 1)).float()


class Float32Output(nn.Module):
    """Keep output precision finer than BF16's roughly 0.47 C gray-code spacing."""
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.conv(x.float())


class SDCodec(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c['model']
        self.representation = Representation(c)
        p = Path(c['pretrained']['directory'])
        for name, digest in c['pretrained']['sha256'].items():
            assert sha256_file(p/name) == digest, f'Pretrained file changed: {name}'
        self.network = AutoencoderKL.from_pretrained(str(p), local_files_only=True,
            use_safetensors=True, low_cpu_mem_usage=False)
        self.network.decoder.conv_out = Float32Output(self.network.decoder.conv_out)
        # Both encoder and decoder are fine-tuned. No diffusion denoiser or scaling factor.

    def forward(self, image, input_mask):
        baseline = visible_mean(image,input_mask).detach()
        residual = torch.where(input_mask.bool(),image.float()-baseline,0)
        encoded = self.representation.encode(residual)
        posterior = self.network.encode(encoded).latent_dist
        z = posterior.sample() if self.training else posterior.mode()
        output = self.network.decode(z).sample
        predicted, off_path, saturated = self.representation.decode(output)
        mean, logvar = posterior.mean.float(), posterior.logvar.float()
        kl = .5*(mean.square()+logvar.exp()-1-logvar).mean()
        return dict(reconstruction=predicted+baseline, temperature_baseline=baseline,
            encoded_prediction=output, quantized=z, indices=None, log_probs=None,
            bottleneck_loss=kl.new_zeros(()), kl=kl, posterior_std=(.5*logvar).exp().mean(),
            off_path_mse=off_path.mean(), saturated_fraction=saturated.mean())

    def rate(self):
        return dict(kind='continuous_latent_plus_visible_mean', latent_shape=[4,32,32],
            latent_elements=4096, side_information_bits=32, bits_per_image=4096*32+32,
            not_a_discrete_codec_rate=True, nominal_float32_storage_only=True)


class ExperimentModel(CrossViewModel):
    def __init__(self,c):
        nn.Module.__init__(self)
        self.codecs = nn.ModuleDict({'lst': SDCodec(c)})


def objective(outputs,batch,c):
    target = batch['lst']['target'].float()
    mask = batch['lst']['reference_mask'].bool()
    rep = Representation(c).to(target.device)
    terms = {}
    for v in outputs['lst']:
        pred = v['reconstruction']
        desired = rep.encode(torch.where(mask,target-v['temperature_baseline'],0))
        # Equal local scalar metric for RGB cube edges and replicated gray values.
        factor = 3/49 if rep.kind == 'asinh_rgb' else 1.
        row = dict(reconstruction=masked_mean((pred-target).square(),mask),
            spatial=spatial_mse(pred,target,mask), gradient=gradient_loss(pred,target,mask),
            representation=factor*masked_mean((v['encoded_prediction'].float()-desired).square(),mask),
            kl=v['kl'], posterior_std=v['posterior_std'], off_path_mse=v['off_path_mse'],
            saturated_fraction=v['saturated_fraction'])
        for k,value in row.items():
            terms[k] = terms.get(k,0)+value/len(outputs['lst'])
    loss = (terms['reconstruction']+c['loss']['spatial_extra']*terms['spatial']+
        c['loss']['gradient']*terms['gradient']+c['loss']['representation']*terms['representation']+
        c['loss']['kl']*terms['kl'])
    return loss,terms


def source_hashes():
    root = HERE.parents[2]
    paths = list(HERE.glob('*.py'))+list((root/'eo_denoise').glob('*.py'))+list((root/'eo_data').glob('*.py'))
    paths += [root/'experiments/paper1/lst_residual/runtime.py']
    return {str(p.relative_to(root)):sha256_file(p) for p in sorted(paths)}
