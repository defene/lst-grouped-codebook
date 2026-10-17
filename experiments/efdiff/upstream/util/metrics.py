import math
import numpy as np
from skimage.metrics import structural_similarity

from lst_dataset import LST_RANGE


def calculate_psnr(img1, img2, data_range=LST_RANGE):
    """Calculate PSNR. Default data_range uses fixed LST range (120 degC)."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(data_range / math.sqrt(mse))


def calculate_bias(pred, gt):
    """Calculate mean bias (pred - gt) in degC. Positive = predicted too hot."""
    return float(np.mean(pred.astype(np.float64) - gt.astype(np.float64)))


def calculate_ssim(img1, img2, data_range=LST_RANGE):
    """Calculate SSIM via skimage. Uses GT dynamic range by default."""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return float(structural_similarity(
        img1.squeeze(), img2.squeeze(),
        data_range=max(float(data_range), 1e-6)))
