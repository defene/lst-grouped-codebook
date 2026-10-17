"""Versioned HLS input correction; original data.py and center_data.py are frozen."""
import numpy as np
from timegeo_data import HLCPairs as TimeGeoPairs, EpochSampler, check_stats
from timegeo_metadata import sha256

INPUT_PROTOCOL = 'hlc-prithvi-hls-unclipped-valid-neutralfill-20261004-v1'


class HLCPairs(TimeGeoPairs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.hls is None:
            raise ValueError('This protocol requires HLS')
        self.fingerprint['prithvi_input'] = dict(protocol=INPUT_PROTOCOL, data_code_sha256=sha256(__file__),
                valid_values='unclipped reflectance', invalid_values='handled after Prithvi normalization',
                mask='source_valid AND non-fill in all bands AND fmask!=255 AND (fmask&14)==0')

    def __getitem__(self, item):
        result = super().__getitem__(item)
        hs = self.hls[int(self.hls_positions[result['index']])]
        valid = hs['reference_mask'][:,16:240,16:240].astype(bool)
        reflectance = self.hls.denormalize(hs['image'])[:,16:240,16:240]
        result['HLS'] = np.where(valid, reflectance, 0).astype(np.float32)
        result['HLS_valid'] = valid.copy()
        return result
