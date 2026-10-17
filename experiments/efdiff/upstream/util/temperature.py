"""Temperature reconstruction utilities shared by train.py and infer.py."""
import numpy as np
from util.normalization import NormStats

# ---------------------------------------------------------------------------
# Model variant helpers
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    'sr3': 'SR3',
    'sr3_prithvi': 'SR3+Prithvi',
    'resshift': 'ResShift',
    'resshift_prithvi': 'ResShift+Prithvi',
}

# Model types that always need HLS data loaded from disk
_NEEDS_HLS = ('sr3_prithvi', 'resshift_prithvi')


def get_model_info(opt):
    """Return (model_type, needs_hls, model_label) from config."""
    model_type = opt['model'].get('which_model_G', 'sr3')
    # HLS needed for Prithvi/HLS variants, or any model with load_hls in config
    needs_hls = model_type in _NEEDS_HLS or opt['model'].get('load_hls', False)
    model_label = MODEL_LABELS.get(model_type, model_type)
    return model_type, needs_hls, model_label


# ---------------------------------------------------------------------------
# Denormalization: tensor -> Celsius numpy
# ---------------------------------------------------------------------------

def residual_to_celsius(tensor, norm_stats):
    """Convert normalized residual tensor to Celsius numpy array."""
    return norm_stats.denormalize_residual(
        tensor.squeeze().float().cpu().numpy())


def condition_to_celsius(tensor, norm_stats):
    """Convert normalized LST condition tensor to Celsius numpy array."""
    return norm_stats.denormalize_lst(
        tensor.squeeze().float().cpu().numpy())


def reconstruct_temperature(sr_residual_tensor, condition_tensor, norm_stats):
    """Reconstruct absolute temperature (Celsius) from residual + condition."""
    return (condition_to_celsius(condition_tensor, norm_stats)
            + residual_to_celsius(sr_residual_tensor, norm_stats))
