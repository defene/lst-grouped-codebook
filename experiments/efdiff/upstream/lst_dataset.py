"""
LSTDataset: Dataset for LST super-resolution with optional HLS band conditioning.

Supports:
  - Prithvi data format (lst.npy, lst_lr.npy, metadata.json)
  - Real-time LR generation using Wald's Protocol
  - Split configuration via split_config JSON
  - Optional HLS reflectance bands for conditioning

Expected data directory structure (Prithvi):
    $SCRATCH/prithvi_data/
    +-- train/                               # Training set (90% of data)
    |   |-- T01UGU/
    |       |-- obs_0019/
    |           |-- r00_c00/
    |               +-- lst.npy              # HR LST (256, 256) int16, scale=100
    |               +-- lst_lr.npy           # (optional) LR LST (8, 8) int16, scale=100
    |               +-- metadata.json        # patch metadata
    |               |-- hls_bands.npy        # (optional) HLS reflectance (6, 256, 256)
    |-- test/                                # Test set (10% of data)
        |-- ...
"""
import os
import json
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict

from util.normalization import NormStats
from util.downsampling import walds_protocol_downsample, calculate_psf_sigma

logger = logging.getLogger('base')

# --- Constants ---
LST_FILL_VALUE = -32767   # int16 nodata sentinel (before /100 scaling)
HLS_FILL_VALUE = -32767   # int16 nodata sentinel for HLS bands
LST_MIN = -50.0           # degrees Celsius
LST_MAX = 70.0
LST_RANGE = LST_MAX - LST_MIN  # 120 degC, fixed range for PSNR
MAX_NAN_RATIO = 0.10
N_HLS_BANDS = 6
HLS_BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]


class LSTDataset(Dataset):
    """Dataset for LST super-resolution with optional HLS reflectance bands.

    Expected directory structure::

        data_dir/
        ├── train/
        │   └── {MGRS_TILE}/obs_{NNNN}/r{RR}_c{CC}/
        │       ├── lst.npy           # HR LST (256, 256) int16, scale=100
        │       ├── hls_bands.npy     # HLS reflectance (6, 256, 256) int16, scale=10000
        │       └── fmask.npy         # Cloud mask (256, 256) uint8
        └── test/
            └── ...

    Processing pipeline per sample:
        1. Load HR LST: int16 / 100 -> Celsius, mask nodata (fill=-32767)
        2. Random crop (train) or center crop (test) to (hr_size, hr_size)
        3. Synthesize the low-resolution input via Wald's Protocol, then
           bicubic-upsample back to (hr_size, hr_size) to form the SR condition
        4. Compute residual: HR_celsius - SR_bicubic_celsius
        5. Z-score normalize both residual and SR condition (3-sigma clipping)
        6. Optionally load HLS bands, scale to [0, 1], apply same crop

    Train vs test behavior:
        - train: random crop, data augmentation (random horizontal/vertical flip)
        - test: center crop, no augmentation

    Returns dict:
        'HR': (1, hr_size, hr_size) z-score normalized residual (HR - SR_bicubic)
        'SR': (1, hr_size, hr_size) z-score normalized bicubic-upsampled LR condition
        'HLS': (6, hr_size, hr_size) HLS reflectance in [0, 1] (only when load_hls=True)
        'Index': int sample index

    Args:
        data_dir: Root directory containing train/ and test/ subdirectories
        split: 'train' or 'test' — selects corresponding subdirectory
        norm_stats: NormStats instance with z-score statistics (default: mu=0, sigma=1)
        load_hls: Whether to load HLS reflectance bands for Prithvi conditioning
        data_len: Max samples to use (-1 for all), stratified-sampled across tiles
        hr_size: Crop size for training patches (default: 224)
        lr_size: LR resolution after downsample (default: 7, i.e. 224/32)
        downsample_factor: Spatial downsampling factor (default: 32.0)
        psf_sigma: Manual PSF sigma for Wald's Protocol (None = auto-calculate)
        sigma_method: PSF sigma method: 'standard', 'opencv', or 'conservative'
        max_nan_ratio: Max allowed NaN fraction in HR LST (default: 0.10)
        seed: Random seed for reproducible tile sampling and augmentation
    """

    def __init__(self, data_dir, split='train', norm_stats=None,
                 load_hls=False, data_len=-1,
                 hr_size=224, lr_size=None,
                 downsample_factor=32.0,
                 psf_sigma=None, sigma_method='standard',
                 max_nan_ratio=MAX_NAN_RATIO, seed=42):
        self.data_dir = Path(data_dir)
        self.split = split
        self.norm_stats = norm_stats or NormStats()
        self.hr_size = hr_size
        self.lr_size = lr_size if lr_size is not None else int(hr_size / downsample_factor)
        self.downsample_factor = downsample_factor
        self.psf_sigma = psf_sigma
        self.sigma_method = sigma_method
        self.load_hls = load_hls
        self.seed = seed
        self.max_nan_ratio = max_nan_ratio

        # Collect all patches
        all_patches = self._collect_patches(self.data_dir, max_nan_ratio)

        # Split by directory structure (train/test subdirs)
        train_patches, test_patches = self._split_by_directory(all_patches)

        if split == 'train':
            self.patches = train_patches
        else:
            self.patches = test_patches

        if data_len > 0 and data_len < len(self.patches):
            self.patches = self._stratified_sample_by_tile(
                self.patches, data_len, seed)

        self.data_len = len(self.patches)

        logger.info(f'{split} set: {len(self.patches)} patches')

        if self.load_hls:
            n_hls = sum(1 for p in self.patches
                        if self._find_hls_file(p) is not None)
            logger.info(f'{split} set: {n_hls}/{len(self.patches)} patches have HLS bands')

        # Log tile distribution
        tile_counts = defaultdict(int)
        for p in self.patches:
            tile_counts[self._extract_tile_id(p)] += 1
        logger.info(f'{split} set: {len(self.patches)} patches from {len(tile_counts)} tiles')

        # Log downsampling info
        sigma = self.psf_sigma or calculate_psf_sigma(
            self.downsample_factor, self.sigma_method)
        logger.info(
            f'{split} set: real-time Wald\'s Protocol downsampling '
            f'factor={self.downsample_factor:.2f}, sigma={sigma:.4f}')

    @staticmethod
    def _extract_tile_id(patch_path):
        """Extract MGRS tile ID from patch path (e.g., T01UGU)."""
        for part in Path(patch_path).parts:
            if (len(part) == 6 and part[0] == 'T'
                    and part[1:3].isdigit() and part[3:].isalpha()):
                return part
        return 'unknown'

    def _collect_patches(self, data_dir, max_nan_ratio):
        """Collect all valid patches from data directory.

        Results are cached in data_dir/.valid_patches_cache.json to avoid
        repeated scans across runs.  Uses fast int16 fill-value check
        (memory-mapped) instead of loading full arrays to float32.
        """
        cache_path = data_dir / '.valid_patches_cache.json'

        if cache_path.exists():
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            if cache.get('max_nan_ratio') == max_nan_ratio:
                filtered = cache['patches']
                logger.info(f'Loaded {len(filtered)} valid patches from cache')
                return filtered

        patches = []
        search_dirs = [
            data_dir / s for s in ['train', 'val', 'test']
            if (data_dir / s).exists()
        ]
        if not search_dirs:
            search_dirs = [data_dir]

        for search_path in search_dirs:
            for root, _, files in os.walk(search_path):
                if 'lst.npy' in files:
                    patches.append(str(root))

        logger.info(f'Found {len(patches)} patches, filtering by fill ratio <= {max_nan_ratio}...')

        filtered = []
        for i, patch_path in enumerate(patches):
            hr_path = os.path.join(patch_path, 'lst.npy')
            if not os.path.exists(hr_path):
                continue
            try:
                arr = np.load(hr_path, mmap_mode='r')
                fill_ratio = (arr == LST_FILL_VALUE).sum() / arr.size
                if fill_ratio <= max_nan_ratio:
                    filtered.append(patch_path)
            except Exception as e:
                logger.debug(f'Failed to load {hr_path}: {e}')
            if (i + 1) % 50000 == 0:
                logger.info(f'  Scanned {i + 1}/{len(patches)} patches...')

        logger.info(
            f'After filtering: {len(filtered)} patches '
            f'(dropped {len(patches) - len(filtered)}) - saving cache')
        with open(cache_path, 'w') as f:
            json.dump({
                'max_nan_ratio': max_nan_ratio,
                'patches': filtered,
            }, f)
        return filtered

    @staticmethod
    def _split_by_directory(patches):
        """Split patches by directory structure (train/test subdirs)."""
        train_patches, test_patches = [], []
        for patch_path in patches:
            parts = Path(patch_path).parts
            if 'test' in parts or 'val' in parts:
                test_patches.append(patch_path)
            else:
                train_patches.append(patch_path)
        return train_patches, test_patches

    def _stratified_sample_by_tile(self, patches, data_len, seed):
        """Sample patches ensuring representation from every tile."""
        rng = np.random.RandomState(seed)
        tile_to_indices = defaultdict(list)
        for i, p in enumerate(patches):
            tile_to_indices[self._extract_tile_id(p)].append(i)

        tiles = sorted(tile_to_indices.keys())
        n_tiles = len(tiles)

        selected = []
        if data_len >= n_tiles:
            for t in tiles:
                selected.append(rng.choice(tile_to_indices[t]))
            remaining = data_len - n_tiles
        else:
            chosen_tiles = rng.choice(tiles, size=data_len, replace=False)
            for t in chosen_tiles:
                selected.append(rng.choice(tile_to_indices[t]))
            remaining = 0

        if remaining > 0:
            selected_set = set(selected)
            pool_per_tile = {
                t: [i for i in tile_to_indices[t] if i not in selected_set]
                for t in tiles
            }
            total_pool = sum(len(v) for v in pool_per_tile.values())

            for t in tiles:
                if total_pool == 0:
                    break
                n_extra = min(
                    max(0, int(round(remaining * len(pool_per_tile[t]) / total_pool))),
                    len(pool_per_tile[t]),
                )
                if n_extra > 0:
                    chosen = rng.choice(pool_per_tile[t], size=n_extra, replace=False)
                    selected.extend(chosen.tolist())

            if len(selected) < data_len:
                all_remaining = [
                    i for t in tiles for i in pool_per_tile[t]
                    if i not in set(selected)
                ]
                gap = data_len - len(selected)
                if gap > 0 and all_remaining:
                    extra = rng.choice(
                        all_remaining, size=min(gap, len(all_remaining)),
                        replace=False)
                    selected.extend(extra.tolist())

            selected = selected[:data_len]

        return [patches[i] for i in sorted(set(selected))]

    # ---- File loading helpers ----

    @staticmethod
    def _load_lst_file(file_path):
        """Load LST data from .npy file. int16 with scale=100 -> float32 Celsius."""
        lst = np.load(file_path)
        if lst.dtype == np.int16:
            fill_mask = (lst == LST_FILL_VALUE)
            lst = lst.astype(np.float32) / 100.0
            lst[fill_mask] = np.nan
        else:
            lst = lst.astype(np.float32)
        return lst

    @staticmethod
    def _find_hls_file(patch_dir):
        """Find HLS data file in a patch directory."""
        for name in ['bands.npy', 'hls_bands.npy', 't0_bands.npy']:
            path = os.path.join(patch_dir, name)
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def _load_hls(patch_dir):
        """Load HLS reflectance bands. Returns (6, H, W) float32 or None."""
        hls_path = LSTDataset._find_hls_file(patch_dir)
        if hls_path is None:
            return None
        try:
            raw = np.load(hls_path)
            if raw.dtype == np.int16:
                fill_mask = (raw == HLS_FILL_VALUE)
                hls = raw.astype(np.float32) / 10000.0
                hls[fill_mask] = np.nan
            else:
                hls = raw.astype(np.float32)

            if hls.ndim == 2:
                return None
            if hls.shape[0] > N_HLS_BANDS:
                hls = hls[:N_HLS_BANDS]
            elif hls.shape[0] < N_HLS_BANDS:
                return None

            hls = np.where(np.isfinite(hls), hls, 0.0)
            return np.clip(hls, 0.0, 1.0)
        except Exception as e:
            logger.warning(f'Failed to load HLS from {hls_path}: {e}')
            return None

    # ---- Main data loading ----

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        patch_dir = self.patches[index]

        # ---- Load HR LST ----
        hr_path = os.path.join(patch_dir, 'lst.npy')
        hr = self._load_lst_file(hr_path)  # float32 Celsius
        if hr.ndim == 3:
            hr = hr[0]

        # ---- Handle fill values ----
        if not np.all(np.isfinite(hr)):
            fill_val = np.nanmean(hr)
            if np.isnan(fill_val):
                fill_val = 0.0
            hr = np.nan_to_num(hr, nan=fill_val)

        # ---- Load HLS early so crop is synchronized ----
        hls_data = self._load_hls(patch_dir) if self.load_hls else None

        # ---- Crop (same region for HR, HLS) ----
        h, w = hr.shape
        if self.split == 'train':
            i = random.randint(0, h - self.hr_size)
            j = random.randint(0, w - self.hr_size)
        else:
            i = (h - self.hr_size) // 2
            j = (w - self.hr_size) // 2

        hr = hr[i:i + self.hr_size, j:j + self.hr_size]
        if hls_data is not None:
            hls_data = hls_data[:, i:i + self.hr_size, j:j + self.hr_size]

        # ---- Downsample cropped HR -> LR -> SR (real-time, matches old pipeline) ----
        lr = walds_protocol_downsample(
            hr, self.downsample_factor,
            psf_sigma=self.psf_sigma, sigma_method=self.sigma_method,
        )
        lr_tensor = torch.from_numpy(lr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        sr_celsius = F.interpolate(
            lr_tensor, size=(self.hr_size, self.hr_size),
            mode='bicubic', align_corners=False,
        ).squeeze(0).numpy()  # (1, 224, 224)

        hr_out = hr[np.newaxis]  # (1, 224, 224)

        # ---- Normalize via NormStats ----
        sr_norm = self.norm_stats.normalize_lst(sr_celsius)
        sr_final = torch.from_numpy(sr_norm.astype(np.float32)).clamp(-1, 1)

        residual = hr_out - sr_celsius
        residual_norm = self.norm_stats.normalize_residual(residual)
        hr_final = torch.from_numpy(residual_norm.astype(np.float32)).clamp(-1, 1)

        result = {'HR': hr_final, 'SR': sr_final, 'Index': index}

        # ---- Attach HLS bands ----
        if self.load_hls:
            if hls_data is not None:
                result['HLS'] = torch.from_numpy(hls_data)
            else:
                result['HLS'] = torch.zeros(
                    N_HLS_BANDS, self.hr_size, self.hr_size, dtype=torch.float32)

        return result
