"""A06: fixed train-only residual scaling, with all image losses in A05 units."""
from pathlib import Path
import math
import torch
from eo_data.core import sha256_file
from eo_denoise.models import CrossViewModel
from experiments.paper1.lst_residual import runtime as base

HERE = Path(__file__).resolve().parent
objective = base.objective


class ScaledResidualCodec(base.ResidualCodec):
    def __init__(self, inner, scale):
        super().__init__(inner)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Residual scale must be finite and positive')
        self.register_buffer('residual_scale', torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, image, input_mask):
        baseline = base.visible_mean(image, input_mask).detach()
        centered = torch.where(input_mask.bool(), (image.float()-baseline)/self.residual_scale, 0)
        result = self.inner(centered, input_mask)
        result['predicted_residual_scaled'] = result['reconstruction']
        result['predicted_residual'] = result['reconstruction'].float()*self.residual_scale
        result['temperature_baseline'] = baseline
        result['reconstruction'] = result['predicted_residual']+baseline
        return result

    def decode_indices(self, indices, temperature_baseline):
        if temperature_baseline.shape != (len(indices), 1, 1, 1):
            raise ValueError('One visible-input mean per image is required')
        flat = indices.reshape(len(indices), -1)
        lengths = [s*s for s in self.inner.scales]
        if flat.shape[1] != sum(lengths) or flat.min() < 0 or flat.max() >= self.quantizer.size:
            raise ValueError('Invalid multiscale token sequence')
        # Match VarCodec.forward: residual token composition is FP32 even under AMP.
        with torch.autocast(device_type=indices.device.type, enabled=False):
            z = self.inner._decode_tokens(torch.split(flat, lengths, dim=1))
        residual = self.inner.network.decoder(self.inner.network.post_quant_conv(z))
        return residual.float()*self.residual_scale+temperature_baseline.float()

    def rate(self):
        result = super().rate()
        result.update(residual_scale=float(self.residual_scale),
                      scale_storage='one shared float32 model buffer; no additional per-image payload')
        return result


class ExperimentModel(CrossViewModel):
    def __init__(self, config):
        assert config['data']['modalities'] == ['lst']
        assert config['residual_experiment']['predict_residual']
        super().__init__(config)
        self.codecs['lst'] = ScaledResidualCodec(self.codecs['lst'],
            config['residual_scaling']['normalized_scale'])


def source_hashes():
    return {**base.source_hashes(), 'lst_residual_scale/runtime.py': sha256_file(HERE/'runtime.py')}
