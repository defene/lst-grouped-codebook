"""Canonical masks, sample-balanced statistics, and normalization."""
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = 'eo-independent-reconstruction-v1'
BANDS = ('Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2')
ALIASES = {'prithvi': 'prithvi', 'prithvi_data': 'prithvi',
           'lstsr_tb': 'lstsr_tb', 'lstsr': 'lstsr_tb',
           'lstsr_tb_final_upload_20260721': 'lstsr_tb'}

def dataset_name(name):
    try:
        return ALIASES[name]
    except KeyError:
        raise ValueError(f'Unknown dataset {name!r}; use prithvi or lstsr_tb') from None

def modality_name(name):
    if name in ('lst', 'lst_hr'):
        return 'lst_hr'
    if name == 'hls':
        return name
    raise ValueError('Use hls or lst_hr (alias lst)')

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def json_write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def canonical(raw, modality, dataset, fmask=None, source_valid=None):
    """Decode each source without clipping, resizing, or per-image fitting."""
    modality = modality_name(modality)
    raw = np.asarray(raw)
    if modality == 'hls':
        if raw.shape != (6, 256, 256) or raw.dtype != np.int16:
            raise ValueError(f'Unexpected HLS schema {raw.shape}, {raw.dtype}')
        if fmask is None or fmask.shape != (256, 256) or fmask.dtype != np.uint8:
            raise ValueError('Expected uint8 HLS Fmask [256,256]')
        fill = (-9999, -32767, -32768) if dataset == 'prithvi' else (-32768,)
        valid = ~np.isin(raw, fill).any(axis=0)
        valid &= (fmask != 255) & ((fmask & 14) == 0)
        scale = 0.0001
    else:
        if raw.shape != (256, 256) or raw.dtype != np.int16:
            raise ValueError(f'Unexpected HR LST schema {raw.shape}, {raw.dtype}')
        fill = (-32767,) if dataset == 'prithvi' else (-32768,)
        valid = ~np.isin(raw, fill)
        raw = raw[None]
        scale = 1.0  # Explicit raw-coded LST; no unverified temperature unit.
    if source_valid is not None:
        if source_valid.shape != (256, 256):
            raise ValueError('Invalid source mask shape')
        valid &= source_valid.astype(bool)
    image = raw.astype(np.float32) * np.float32(scale)
    return image, valid[None]

def sample_moments(image, mask):
    n = int(mask.sum())
    if n == 0:
        return None
    values = image[:, mask[0]].astype(np.float64)
    return {'pixels': n, 'mean': values.mean(axis=1).tolist(),
            'second': np.square(values).mean(axis=1).tolist()}

def empty_accumulator(channels):
    return {'samples': 0, 'pixels': 0, 'sum_mean': [0.0]*channels, 'sum_second': [0.0]*channels}

def accumulate(acc, moments):
    if moments is None:
        return
    acc['samples'] += 1
    acc['pixels'] += moments['pixels']
    acc['sum_mean'] = (np.asarray(acc['sum_mean']) + moments['mean']).tolist()
    acc['sum_second'] = (np.asarray(acc['sum_second']) + moments['second']).tolist()

def finish_stats(acc, modality):
    if not acc['samples']:
        raise ValueError(f'No valid training samples for {modality}')
    mean = np.asarray(acc['sum_mean']) / acc['samples']
    variance = np.maximum(np.asarray(acc['sum_second']) / acc['samples'] - mean**2, 0)
    std = np.sqrt(variance)
    return {'mean': mean.tolist(), 'std': np.maximum(std, 1e-6).tolist(),
            'unfloored_std': std.tolist(), 'std_floor': 1e-6,
            'samples': acc['samples'], 'valid_spatial_pixels': acc['pixels'],
            'fit_split': 'train', 'weighting': 'equal_samples_then_equal_valid_pixels_within_sample',
            'units': 'reflectance' if modality == 'hls' else 'source_raw_int16_units',
            'physical_temperature_unit_confirmed': False if modality == 'lst_hr' else None}

def normalize(image, mask, stats):
    mean = np.asarray(stats['mean'], np.float32)[:, None, None]
    std = np.asarray(stats['std'], np.float32)[:, None, None]
    return np.where(mask, (image - mean) / std, np.float32(0)).astype(np.float32)

def denormalize(image, stats):
    """Channel axis is -3; supports [C,H,W] and [B,C,H,W]."""
    shape = [1] * np.ndim(image)
    shape[-3] = len(stats['mean'])
    return np.asarray(image) * np.asarray(stats['std'], np.float32).reshape(shape) + np.asarray(stats['mean'], np.float32).reshape(shape)
