"""
Compute z-score normalization statistics from training data.

Computes mean/std for:
  - LST condition  (bicubic-upsampled LR, used as SR condition input)
  - Residual       (HR - bicubic_upsampled_LR, the diffusion target)

Saves to a nested JSON file for use in load_normalization_stats().

Usage:
    python scripts/compute_residual_stats.py
    python scripts/compute_residual_stats.py \\
        --data_dir $SCRATCH/prithvi_data \\
        --output $SCRATCH/lst-super-resolution/normalization_stats.json
"""
import argparse
import json
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lst_dataset import MAX_NAN_RATIO, LST_MIN, LST_MAX, LST_FILL_VALUE
from util.downsampling import (
    walds_protocol_downsample, downsample_area_average, calculate_psf_sigma,
)

FMASK_CLOUD_BITS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)

HOME_DIR = os.path.expanduser("~")
DEFAULT_SCRATCH = os.getenv("SCRATCH", os.path.join(HOME_DIR, "scratch"))
DEFAULT_WORK_DIR = os.path.join(DEFAULT_SCRATCH, "lst-super-resolution")
DEFAULT_DATA_DIR = os.path.join(DEFAULT_SCRATCH, "prithvi_data")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_WORK_DIR, "normalization_stats.json")



def compute_stats(data_dir, hr_size=224, lr_size=7, downsample_factor=32.0,
                  downsample_method='strict', sigma_method='standard',
                  max_nan_ratio=MAX_NAN_RATIO, max_patches=None):
    """Compute z-score stats for both LST condition and residual over training patches."""
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    lst_files = sorted(train_dir.rglob('lst.npy'))
    print(f"Found {len(lst_files)} LST patches in {train_dir}")

    if max_patches and max_patches < len(lst_files):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(lst_files), max_patches, replace=False)
        lst_files = [lst_files[i] for i in sorted(indices)]
        print(f"Subsampled to {len(lst_files)} patches")

    psf_sigma = calculate_psf_sigma(downsample_factor, method=sigma_method)

    # Vectorized accumulators (float64 for numerical stability)
    res_sum, res_sum_sq, res_count = np.float64(0), np.float64(0), 0
    lst_sum, lst_sum_sq, lst_count = np.float64(0), np.float64(0), 0

    res_min, res_max = float('inf'), float('-inf')
    lst_min_obs, lst_max_obs = float('inf'), float('-inf')

    skipped = 0
    for lst_path in tqdm(lst_files, desc="Computing normalization stats"):
        try:
            hr_raw = np.load(lst_path)
        except Exception:
            skipped += 1
            continue

        # Detect fill value on raw int16 BEFORE conversion
        if hr_raw.dtype == np.int16:
            fill_mask = (hr_raw == LST_FILL_VALUE)
            hr_raw = hr_raw.astype(np.float32) / 100.0
            hr_raw[fill_mask] = np.nan
        else:
            hr_raw = hr_raw.astype(np.float32)
        # Mark out-of-range as NaN
        hr_raw[(hr_raw < LST_MIN) | (hr_raw > LST_MAX)] = np.nan

        if hr_raw.ndim == 3:
            hr_raw = hr_raw[0]

        h, w = hr_raw.shape
        if h < hr_size or w < hr_size:
            skipped += 1
            continue
        i = (h - hr_size) // 2
        j = (w - hr_size) // 2
        hr = hr_raw[i:i + hr_size, j:j + hr_size]

        # Load Fmask to mask cloud-contaminated pixels
        cloud_mask = np.zeros((hr_size, hr_size), dtype=bool)
        fmask_path = lst_path.parent / 'fmask.npy'
        if fmask_path.exists():
            try:
                fmask = np.load(fmask_path)
                if fmask.ndim == 3:
                    fmask = fmask[0]
                if fmask.shape[0] >= hr_size and fmask.shape[1] >= hr_size:
                    fmask = fmask[i:i + hr_size, j:j + hr_size]
                    cloud_mask = (fmask.astype(np.uint8) & FMASK_CLOUD_BITS) > 0
            except Exception:
                pass

        # Valid = finite AND within physical range AND cloud-free
        valid_mask = np.isfinite(hr) & (hr >= LST_MIN) & (hr <= LST_MAX) & ~cloud_mask
        valid_ratio = valid_mask.sum() / valid_mask.size
        if valid_ratio < (1.0 - max_nan_ratio):
            skipped += 1
            continue

        valid_pixels = hr[valid_mask]
        if len(valid_pixels) == 0:
            skipped += 1
            continue
        hr_mean_val = float(valid_pixels.mean())

        # Fill invalid pixels with patch mean for smooth downsampling
        hr_filled = np.where(valid_mask, hr, hr_mean_val)

        if downsample_method == 'strict':
            lr = walds_protocol_downsample(hr_filled, downsample_factor,
                                           psf_sigma=psf_sigma,
                                           sigma_method=sigma_method)
        else:
            lr = downsample_area_average(hr_filled, downsample_factor)

        lr = np.clip(lr, LST_MIN, LST_MAX)

        lr_tensor = torch.from_numpy(lr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        sr = F.interpolate(lr_tensor, size=(hr_size, hr_size),
                           mode='bicubic', align_corners=False).squeeze().numpy()

        residual = hr_filled - sr

        # Accumulate stats ONLY on valid pixels (vectorized float64)
        res_valid = residual[valid_mask].astype(np.float64)
        sr_valid = sr[valid_mask].astype(np.float64)

        res_sum += res_valid.sum()
        res_sum_sq += (res_valid ** 2).sum()
        res_count += len(res_valid)
        res_min = min(res_min, float(res_valid.min()))
        res_max = max(res_max, float(res_valid.max()))

        lst_sum += sr_valid.sum()
        lst_sum_sq += (sr_valid ** 2).sum()
        lst_count += len(sr_valid)
        lst_min_obs = min(lst_min_obs, float(sr_valid.min()))
        lst_max_obs = max(lst_max_obs, float(sr_valid.max()))

    if res_count == 0:
        raise RuntimeError("No valid patches found!")

    res_mean = float(res_sum / res_count)
    res_std = float(np.sqrt(res_sum_sq / res_count - res_mean ** 2))
    lst_mean = float(lst_sum / lst_count)
    lst_std = float(np.sqrt(lst_sum_sq / lst_count - lst_mean ** 2))

    stats = {
        "residual": {"mean": res_mean, "std": res_std, "min": float(res_min), "max": float(res_max)},
        "lst":      {"mean": lst_mean, "std": lst_std, "min": float(lst_min_obs), "max": float(lst_max_obs)},
        "n_patches": res_count // (hr_size * hr_size),
        "n_pixels": res_count,
        "n_skipped": skipped,
        "config": {
            "hr_size": hr_size,
            "lr_size": lr_size,
            "downsample_factor": downsample_factor,
            "downsample_method": downsample_method,
            "sigma_method": sigma_method,
        },
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute residual statistics")
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument('--hr_size', type=int, default=224)
    parser.add_argument('--lr_size', type=int, default=7)
    parser.add_argument('--downsample_factor', type=float, default=32.0)
    parser.add_argument('--downsample_method', type=str, default='strict')
    parser.add_argument('--sigma_method', type=str, default='standard')
    parser.add_argument('--max_patches', type=int, default=None,
                        help='Max patches to process (None=all)')
    args = parser.parse_args()

    print(f"Computing normalization statistics from {args.data_dir}...")
    stats = compute_stats(
        data_dir=args.data_dir,
        hr_size=args.hr_size,
        lr_size=args.lr_size,
        downsample_factor=args.downsample_factor,
        downsample_method=args.downsample_method,
        sigma_method=args.sigma_method,
        max_patches=args.max_patches,
    )

    def _print_section(label, s):
        mean, std = s['mean'], s['std']
        print(f"\n  [{label}]")
        print(f"    mean:          {mean:.4f} degC")
        print(f"    std:           {std:.4f} degC")
        print(f"    3-sigma range: [{mean - 3*std:.2f}, {mean + 3*std:.2f}] degC")
        print(f"    min/max:       {s['min']:.2f} / {s['max']:.2f} degC")

    print(f"\n{'='*55}")
    print(f"Normalization Statistics ({stats['n_patches']} patches, {stats['n_pixels']:,} pixels)")
    print(f"{'='*55}")
    _print_section("LST condition (SR bicubic)", stats['lst'])
    _print_section("Residual (HR - SR)", stats['residual'])
    print(f"\n  skipped: {stats['n_skipped']} patches")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
