"""Real-time downsampling methods for LST super-resolution (Wald's Protocol)."""
import numpy as np
from scipy import ndimage


def calculate_psf_sigma(downsample_factor, method='standard'):
    """Calculate Gaussian PSF sigma based on downsampling factor.

    Methods:
        - 'standard': sigma = downsample_factor / pi (most common)
        - 'opencv': sigma = 2 * downsample_factor / 6.0
        - 'conservative': sigma = downsample_factor / 2.0
    """
    if method == 'standard':
        return downsample_factor / np.pi
    elif method == 'opencv':
        return 2 * downsample_factor / 6.0
    elif method == 'conservative':
        return downsample_factor / 2.0
    else:
        raise ValueError(f"Unknown method: {method}")


def downsample_area_average(data, factor):
    """Simple area averaging downsampling for (H, W) array."""
    h, w = data.shape
    factor_int = int(np.round(factor))
    h_new = (h // factor_int) * factor_int
    w_new = (w // factor_int) * factor_int
    reshaped = data[:h_new, :w_new].reshape(
        h_new // factor_int, factor_int,
        w_new // factor_int, factor_int,
    )
    with np.errstate(invalid='ignore'):
        return np.nanmean(reshaped, axis=(1, 3))


def walds_protocol_downsample(data, factor, psf_sigma=None,
                              sigma_method='standard'):
    """Wald's Protocol: PSF simulation (Gaussian blur) + area-average downsampling.

    Args:
        data: (H, W) high-resolution array
        factor: downsampling factor (e.g., 32 for 256->8)
        psf_sigma: Manual sigma (if None, calculated from factor)
        sigma_method: Method to calculate sigma if psf_sigma is None
    """
    if psf_sigma is None:
        psf_sigma = calculate_psf_sigma(factor, method=sigma_method)
    blurred = ndimage.gaussian_filter(data, sigma=psf_sigma, mode='nearest')
    return downsample_area_average(blurred, factor)
