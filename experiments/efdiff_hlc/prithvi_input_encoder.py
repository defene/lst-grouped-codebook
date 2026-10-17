"""Official Prithvi-300M scaling with explicit post-normalization invalid fill.

HLS validity is NOT the LST target mask and NOT an attention mask. Frozen
Prithvi weights are unchanged. Neutral fill follows the official inference
example; no claim of native cloud masking by the encoder is made.
"""
import torch
from data import UPSTREAM
from model.prithvi_encoder import PrithviEncoder


def normalize_hls(hls, valid, mean, std):
    if hls.ndim != 4 or hls.shape[1] != 6 or valid.shape != (len(hls),1,*hls.shape[-2:]):
        raise ValueError('Expected HLS [B,6,H,W] and HLS_valid [B,1,H,W]')
    if valid.dtype != torch.bool:
        raise ValueError('HLS validity must be boolean')
    normalized = (hls.float()*10000-mean)/std
    if not torch.isfinite(normalized.masked_select(valid.expand_as(normalized))).all():
        raise ValueError('Nonfinite valid HLS values')
    return torch.where(valid, normalized, normalized.new_tensor(.0001))


class MaskAwarePrithviEncoder(PrithviEncoder):
    requires_hls_valid = True

    def forward(self, hls, valid):
        x = normalize_hls(hls,valid,self.pixel_mean,self.pixel_std).unsqueeze(2)
        with torch.no_grad():
            return self._extract_features(x)
