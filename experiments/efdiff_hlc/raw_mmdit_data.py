"""Versioned raw-HLS data entry; preserves fixed center crop and split members."""
from prithvi_input_data import HLCPairs as ReflectancePairs, EpochSampler, check_stats
from timegeo_metadata import sha256


class HLCPairs(ReflectancePairs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fingerprint.pop('prithvi_input', None)
        self.fingerprint['raw_hls_input'] = dict(
            protocol='hlc-raw-hls-fullmask-mmdit-20261014-v1',
            code_sha256=sha256(__file__),
            reflectance_reader_sha256=sha256(__import__('prithvi_input_data').__file__),
            encoder='trainable patch embedding; no pretrained weights',
            hls='six unclipped reflectance bands; existing affine scaling in model',
            validity='full-resolution HLS_valid supplied separately; not attention mask')
