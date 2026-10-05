"""Isolated LST residual experiment; the existing eo_denoise sources stay frozen."""
from pathlib import Path
import torch
from torch import nn
from eo_data.core import sha256_file
from eo_denoise import engine
from eo_denoise.models import CrossViewModel as OriginalModel
from eo_denoise.losses import objective as original_objective, masked_mean

HERE = Path(__file__).resolve().parent
_original_hashes = engine.source_hashes


def visible_mean(image, input_mask):
    valid = input_mask.bool()
    count = valid.float().sum((-2, -1), keepdim=True)
    if (count == 0).any():
        raise ValueError('Residual prediction requires at least one visible pixel per image')
    return torch.where(valid, image.float(), 0).sum((-2, -1), keepdim=True)/count


def spatial_mse(pred, target, reference):
    error = pred.float()-target.float()
    mean_error = visible_mean(error, reference)
    return masked_mean((error-mean_error).square(), reference)


class ResidualCodec(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.config = inner.config

    @property
    def quantizer(self):
        return self.inner.quantizer

    def forward(self, image, input_mask):
        baseline = visible_mean(image, input_mask).detach()
        centered = torch.where(input_mask.bool(), image.float()-baseline, 0)
        result = self.inner(centered, input_mask)
        result['predicted_residual'] = result['reconstruction']
        result['temperature_baseline'] = baseline
        result['reconstruction'] = result['reconstruction'].float()+baseline
        return result

    def decode_indices(self, indices, temperature_baseline):
        if temperature_baseline.shape != (len(indices), 1, 1, 1):
            raise ValueError('Decode requires one transmitted temperature baseline per image')
        return self.inner.decode_indices(indices).float()+temperature_baseline.float()

    def rate(self):
        r = dict(self.inner.rate())
        r.update(kind='multiscale_tokens_plus_visible_temperature_mean', token_bits_per_image=r['bits_per_image'],
                 side_information_bits=32, side_information='one float32 visible-input mean in frozen normalized units')
        r['bits_per_image'] += 32
        r['bpp_spatial'] = r['bpp_band_pixel'] = r['bits_per_image']/65536
        return r


class ExperimentModel(OriginalModel):
    def __init__(self, config):
        if config['data']['modalities'] != ['lst']:
            raise ValueError('This experiment is temperature only')
        super().__init__(config)
        if config['residual_experiment']['predict_residual']:
            self.codecs['lst'] = ResidualCodec(self.codecs['lst'])


def objective(outputs, batch, config):
    loss, terms = original_objective(outputs, batch, config)
    spatial = torch.stack([spatial_mse(v['reconstruction'], batch['lst']['target'], batch['lst']['reference_mask'])
                           for v in outputs['lst']]).mean()
    extra = config['residual_experiment']['spatial_extra_weight']*spatial
    terms.update(spatial=spatial, image_mean=terms['reconstruction']-spatial, spatial_extra=extra)
    return loss+extra, terms


def source_hashes():
    return {**_original_hashes(), **{'lst_residual/'+name: sha256_file(HERE/name) for name in ('runtime.py', 'run.py')}}


def install():
    engine.CrossViewModel = ExperimentModel
    engine.objective = objective
    engine.source_hashes = source_hashes
