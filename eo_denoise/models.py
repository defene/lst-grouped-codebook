"""Independent modality codecs; cross-view training never bypasses bottlenecks."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .ops import resize_grid


def norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(norm(channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
                                   norm(channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x+self.block(x)


class Upsample(nn.Module):
    def forward(self, x):
        return x.repeat_interleave(2, -2).repeat_interleave(2, -1)


class ConvBackbone(nn.Module):
    def __init__(self, channels, c):
        super().__init__()
        widths = [c['base_width']*m for m in (1, 1, 2, 2, 4)]
        encoder = [nn.Conv2d(channels+1, widths[0], 3, padding=1)]
        for i, width in enumerate(widths):
            encoder.extend(Residual(width) for _ in range(c['blocks']))
            if i < 4:
                encoder.append(nn.Conv2d(width, widths[i+1], 4, stride=2, padding=1))
        encoder.extend([norm(widths[-1]), nn.SiLU(), nn.Conv2d(widths[-1], c['latent_dim'], 1)])
        decoder = [nn.Conv2d(c['latent_dim'], widths[-1], 3, padding=1)]
        for i in range(4, -1, -1):
            decoder.extend(Residual(widths[i]) for _ in range(c['blocks']))
            if i:
                decoder.extend([Upsample(),
                                nn.Conv2d(widths[i], widths[i-1], 3, padding=1)])
        decoder.extend([norm(widths[0]), nn.SiLU(), nn.Conv2d(widths[0], channels, 3, padding=1)])
        self.encoder, self.decoder = nn.Sequential(*encoder), nn.Sequential(*decoder)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def position_3d(dim, height=16, width=16):
    def one(coords):
        per_axis = dim//3
        omega = 1.0/(10000**(torch.arange(per_axis//2).float()/(per_axis//2)))
        values = coords.reshape(-1, 1)*omega[None]
        return torch.cat([values.sin(), values.cos()], -1)
    t, y, x = torch.meshgrid(torch.arange(1), torch.arange(height), torch.arange(width), indexing='ij')
    return torch.cat([one(t), one(y), one(x)], -1)[None]


class PrithviStyleBackbone(nn.Module):
    """From-scratch T=1 spatiotemporal-patch ViT adaptation, not pretrained Prithvi.

    Uses 3D patch embedding + separable 3D sin/cos positions, full token grids,
    and a transformer pixel decoder. No geo/date conditioning or MAE mask is
    added to the common corruption protocol.
    """
    def __init__(self, channels, c):
        super().__init__()
        dim = c['vit_dim']
        self.channels = channels
        self.patch = nn.Conv3d(channels+1, dim, (1, 16, 16), stride=(1, 16, 16))
        self.register_buffer('position', position_3d(dim), persistent=True)
        def transformer():
            layer = nn.TransformerEncoderLayer(dim, c['vit_heads'], 4*dim, dropout=0,
                                               activation='gelu', batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, c['vit_depth'], nn.LayerNorm(dim), enable_nested_tensor=False)
        self.encoder, self.decoder = transformer(), transformer()
        self.to_latent = nn.Linear(dim, c['latent_dim'])
        self.from_latent = nn.Linear(c['latent_dim'], dim)
        self.to_pixels = nn.Linear(dim, channels*16*16)

    def encode(self, x):
        z = self.patch(x.unsqueeze(2)).flatten(2).transpose(1, 2)
        z = self.encoder(z+self.position.to(z.dtype))
        return self.to_latent(z).transpose(1, 2).reshape(x.shape[0], -1, 16, 16)

    def decode(self, z):
        b = z.shape[0]
        h = self.from_latent(z.flatten(2).transpose(1, 2))
        h = self.decoder(h+self.position.to(h.dtype))
        pixels = self.to_pixels(h).reshape(b, 16, 16, self.channels, 16, 16)
        return pixels.permute(0, 3, 1, 4, 2, 5).reshape(b, self.channels, 256, 256)


class VectorQuantizer(nn.Module):
    def __init__(self, dim, size, beta=.25, temperature=1):
        super().__init__()
        self.embedding = nn.Embedding(size, dim)
        nn.init.uniform_(self.embedding.weight, -1/math.sqrt(dim), 1/math.sqrt(dim))
        self.dim, self.size, self.beta, self.temperature = dim, size, beta, temperature

    def forward(self, z):
        b, d, h, w = z.shape
        # Code assignments/distances always use float32, including under BF16.
        with torch.autocast(device_type=z.device.type, enabled=False):
            vectors = z.float().permute(0, 2, 3, 1).reshape(-1, d)
            codebook = self.embedding.weight.float()
            distance = (vectors.square().sum(-1, keepdim=True) + codebook.square().sum(-1)[None]
                        - 2*vectors@codebook.T).clamp_min(0)/d
            indices = distance.argmin(-1)
            values = F.embedding(indices, codebook).reshape(b, h, w, d).permute(0, 3, 1, 2)
            loss = F.mse_loss(values, z.float().detach()) + self.beta*F.mse_loss(z.float(), values.detach())
            quantized = z.float()+(values-z.float()).detach() if self.training else values
            log_probs = (-distance/self.temperature).log_softmax(-1).reshape(b, h, w, self.size).permute(0, 3, 1, 2)
        return {'quantized': quantized.to(z.dtype), 'indices': indices.reshape(b, h, w),
                'bottleneck_loss': loss, 'kl': z.new_zeros(()), 'log_probs': log_probs}

    def from_indices(self, indices):
        if indices.min() < 0 or indices.max() >= self.size:
            raise ValueError('Token index outside codebook')
        return self.embedding(indices.long()).permute(0, 3, 1, 2)


class FiniteScalarQuantizer(nn.Module):
    """Bounded scalar STE; even-level offset follows the FSQ reference equations."""
    def __init__(self, dim, levels):
        super().__init__()
        self.project_in, self.project_out = nn.Conv2d(dim, len(levels), 1), nn.Conv2d(len(levels), dim, 1)
        self.register_buffer('levels', torch.tensor(levels, dtype=torch.long))
        basis = [1]
        for level in levels[:-1]:
            basis.append(basis[-1]*level)
        self.register_buffer('basis', torch.tensor(basis, dtype=torch.long))
        self.size = math.prod(levels)

    def forward(self, z):
        projected = self.project_in(z).float()
        levels = self.levels[None, :, None, None]
        half_range = (levels-1)*1.001/2
        offset = (levels % 2 == 0).float()*.5
        shift = torch.atanh(offset/half_range)
        bounded = torch.tanh(projected+shift)*half_range-offset
        integers = bounded.round()
        quantized = bounded+(integers-bounded).detach() if self.training else integers
        half_width = (levels//2).float()
        scalar_codes = quantized/half_width
        digits = integers.long()+levels//2
        indices = (digits*self.basis[None, :, None, None]).sum(1)
        return {'quantized': self.project_out(scalar_codes.to(z.dtype)), 'indices': indices,
                'bottleneck_loss': z.new_zeros(()), 'kl': z.new_zeros(()), 'log_probs': None}

    def from_indices(self, indices):
        if indices.min() < 0 or indices.max() >= self.size:
            raise ValueError('FSQ token outside vocabulary')
        digits = (indices[:, None]//self.basis[None, :, None, None]) % self.levels[None, :, None, None]
        half = (self.levels//2)[None, :, None, None]
        return self.project_out((digits-half).float()/half)


class Codec(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        self.config, self.channels = config, channels
        if config['backbone'] == 'prithvi_eo2_spatial':
            from .prithvi_spatial import PrithviSpatialBackbone
            backbone_type = PrithviSpatialBackbone
        else:
            backbone_type = ConvBackbone if config['backbone'] == 'conv' else PrithviStyleBackbone
        self.backbone = backbone_type(channels, config)
        dim = config['latent_dim']
        self.projection = nn.Sequential(nn.Conv2d(dim, 32, 1), nn.SiLU(), nn.Conv2d(32, 16, 1))
        kind = config['quantizer']
        if kind == 'vq':
            self.quantizer = VectorQuantizer(dim, config['codebook_size'], config['commitment'], config['temperature'])
        elif kind == 'fsq':
            self.quantizer = FiniteScalarQuantizer(dim, config['levels'])
        elif kind == 'vae':
            self.moments = nn.Conv2d(dim, 2*dim, 1)

    def forward(self, image, input_mask):
        if image.shape[-2:] != (256, 256):
            raise ValueError('Input must use the frozen 256x256 data protocol')
        x = torch.cat([torch.where(input_mask.bool(), image, 0), input_mask.to(image.dtype)], 1)
        latent = self.backbone.encode(x)
        grid = self.config['grid']
        z = resize_grid(latent, grid)
        kind = self.config['quantizer']
        if kind in ('vq', 'fsq'):
            result = self.quantizer(z)
        else:
            kl = z.new_zeros(())
            if kind == 'vae':
                mean, logvar = self.moments(z).chunk(2, dim=1)
                logvar = logvar.clamp(-20, 10)
                z = mean+torch.exp(.5*logvar)*torch.randn_like(mean) if self.training else mean
                kl = .5*(mean.float().square()+logvar.float().exp()-1-logvar.float()).mean()
            result = {'quantized': z, 'indices': None, 'bottleneck_loss': z.new_zeros(()), 'kl': kl, 'log_probs': None}
        result['reconstruction'] = self.decode_latent(result['quantized'])
        result['structure'] = self.projection(result['quantized'])
        return result

    def decode_latent(self, z):
        return self.backbone.decode(resize_grid(z, 16))

    def decode_indices(self, indices):
        if self.config['quantizer'] not in ('vq', 'fsq'):
            raise ValueError('Continuous AE/VAE do not expose discrete codes')
        return self.decode_latent(self.quantizer.from_indices(indices))

    def rate(self):
        tokens = self.config['grid']**2
        if self.config['quantizer'] in ('vq', 'fsq'):
            vocabulary = self.quantizer.size
            bits = tokens*math.ceil(math.log2(vocabulary))
            return {'kind': 'fixed_length_token_payload', 'tokens': tokens, 'vocabulary': vocabulary,
                    'bits_per_image': bits, 'bpp_spatial': bits/(256*256),
                    'bpp_band_pixel': bits/(256*256*self.channels),
                    'includes_mask_header_or_model': False}
        return {'kind': 'continuous_float32_latent_payload', 'latent_elements': tokens*self.config['latent_dim'],
                'bits_per_image': tokens*self.config['latent_dim']*32,
                'not_a_discrete_codec_rate': True}


class CrossViewModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['model']['backbone']=='prithvi_eo2_continuous':
            from .prithvi_continuous import PrithviContinuousCodec
            codec_class=PrithviContinuousCodec
        elif config['model']['quantizer']=='var_vq':
            from .var_codec import VarCodec
            codec_class=VarCodec
        else:codec_class=Codec
        self.codecs = nn.ModuleDict({m: codec_class(6 if m == 'hls' else 1, config['model']) for m in config['data']['modalities']})

    def forward(self, inputs):
        # Only noisy images and input masks are accepted. Clean target/reference
        # masks and other-modality observations never feed the codec decoder.
        return {m: [codec(view['image'], view['input_mask']) for view in inputs[m]]
                for m, codec in self.codecs.items()}
