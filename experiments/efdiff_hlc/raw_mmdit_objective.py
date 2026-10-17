"""Reuse the frozen v-target, valid-LST loss and DDIM sampler, without Prithvi."""
from dit_timegeo_objectives import TimeGeoObjective


class RawMMDiTObjective(TimeGeoObjective):
    def __init__(self, denoiser):
        super().__init__(denoiser, None, 'v')

    def encode_batch(self, batch):
        # Must retain autograd in training: unlike Prithvi this encoder is learned.
        # Inherited sample() is no_grad and caches this output once per sampling.
        return self.denoise_fn.encode_condition(batch['SR'], batch['HLS'], batch['HLS_valid'])
