"""Deterministic, model-independent synthetic corruption in Celsius space.

This is a controlled restoration benchmark, not a calibrated sensor simulator.
LST is a retrieved temperature product: applying the RGB atmospheric scattering
equation to LST would not establish a physically valid cloud/haze model.
``cloud_missing`` uses cloud-shaped missing observations with a known mask;
``cloud_bias`` is explicitly a blind synthetic regional temperature-error test.
The latter's affected support is for metrics only and MUST NOT enter a model.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np


CORRUPTION_VERSION = "lst-celsius-synthetic-v1-20260925"

PROFILE_METADATA: dict[str, dict[str, Any]] = {
    "clean": {"known_missing": False, "blind_error": False, "description": "Unchanged valid reference observations."},
    "pixel10": {"known_missing": True, "blind_error": False, "description": "Exactly round(0.1*N_valid) randomly missing valid pixels."},
    "sensor_gaussian": {"known_missing": False, "blind_error": True, "description": "Synthetic independent zero-mean Gaussian temperature error; not sensor-calibrated."},
    "sensor_stripe": {"known_missing": False, "blind_error": True, "description": "Synthetic column-wise additive temperature offsets; not sensor-calibrated."},
    "cloud_missing": {"known_missing": True, "blind_error": False, "description": "Smooth-field ranked cloud-shaped missing support, with an explicit observation-validity mask."},
    "cloud_bias": {"known_missing": False, "blind_error": True, "description": "Synthetic signed regional temperature bias; not a radiative-transfer or validated haze simulation."},
    "joint": {"known_missing": True, "blind_error": True, "description": "Independent Gaussian plus column-stripe error and cloud-shaped missing support."},
    "patch16": {"known_missing": True, "blind_error": False, "description": "Whole 16x16 patches: round(0.1*N_patches), hence 26/256 at 256x256; separate MAE track."},
}


def profile_metadata(kind: str) -> dict[str, Any]:
    if kind not in PROFILE_METADATA:
        raise ValueError(f"Unknown corruption profile: {kind!r}")
    return {"kind": kind, "version": CORRUPTION_VERSION, **PROFILE_METADATA[kind]}


def _nonnegative(config: dict[str, Any], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be finite and nonnegative")
    return value


def _coverage(config: dict[str, Any]) -> float:
    value = float(config.get("coverage", 0.1))
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("coverage must lie in [0,1)")
    return value


def _seed(config: dict[str, Any], kind: str, sample_id: str, split: str, epoch: int, view: int) -> int:
    payload = {
        "version": CORRUPTION_VERSION,
        "seed": int(config.get("seed", 20260925)),
        "kind": kind,
        "sample_id": str(sample_id),
        "split": split,
        "view": int(view),
    }
    if split == "train":
        payload["epoch"] = int(epoch)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).digest()
    # Default PyTorch collation represents Python integers as signed int64.
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _smooth_field(rng: np.random.Generator, height: int, width: int, grid: Any) -> np.ndarray:
    """Bilinearly interpolate a coarse Gaussian grid using NumPy alone."""
    if isinstance(grid, (tuple, list)):
        if len(grid) != 2:
            raise ValueError("field_grid must be an integer or [rows,columns]")
        gh, gw = int(grid[0]), int(grid[1])
    else:
        gh = gw = int(grid)
    if gh < 2 or gw < 2 or gh > height or gw > width:
        raise ValueError("field_grid must be at least 2 and at most the image size on each axis")
    coarse = rng.standard_normal((gh, gw))
    xknots = np.linspace(0.0, float(width - 1), gw)
    yknots = np.linspace(0.0, float(height - 1), gh)
    xs, ys = np.arange(width), np.arange(height)
    along_x = np.stack([np.interp(xs, xknots, row) for row in coarse], axis=0)
    return np.stack([np.interp(ys, yknots, along_x[:, column]) for column in range(width)], axis=1).astype(np.float32)


def _rank_support(field: np.ndarray, valid: np.ndarray, fraction: float) -> np.ndarray:
    """Select exact coverage among reference-valid pixels; stable tie order."""
    support = np.zeros_like(valid, dtype=bool)
    valid_flat = np.flatnonzero(valid.reshape(-1))
    count = int(round(fraction * valid_flat.size))
    if count:
        # Stable sort makes tied field values resolve by flat pixel index.
        order = np.argsort(-field.reshape(-1)[valid_flat], kind="stable")
        support.reshape(-1)[valid_flat[order[:count]]] = True
    return support


def corrupt(
    target_celsius: np.ndarray,
    reference_valid: np.ndarray,
    config: dict[str, Any],
    *,
    sample_id: str,
    split: str,
    epoch: int = 0,
    view: int = 0,
) -> dict[str, Any]:
    """Return one reproducible physical observation and separate audit masks.

    Models receive ``observation`` and ``observed_valid``; the independent
    reference-valid mask may also be an explicitly shared model input. The
    ``affected_mask`` is privileged corruption-support metadata for stratified
    metrics, never an input. ``missing_mask`` excludes naturally invalid pixels.
    ``patch_missing`` describes intentionally dropped whole patches only.

    Seeds exclude all model/representation names. Train seeds change per epoch;
    val/test seeds ignore epoch. Severity values do not change the random seed,
    enabling coupled severity sweeps with the same underlying random draws.
    Every returned array is a fresh allocation: the input is not mutated.
    """
    kind = str(config.get("kind", "clean"))
    profile_metadata(kind)
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    if int(epoch) < 0 or int(view) < 0:
        raise ValueError("epoch and view must be nonnegative")
    target = np.asarray(target_celsius, dtype=np.float32)
    valid = np.asarray(reference_valid, dtype=bool)
    if target.ndim != 3 or target.shape[0] != 1 or valid.shape != target.shape:
        raise ValueError("target_celsius and reference_valid must both have shape [1,H,W]")
    if not valid.any():
        raise ValueError("Cannot corrupt an image without reference-valid pixels")
    if not np.isfinite(target[valid]).all():
        raise ValueError("Reference-valid temperatures must be finite")
    height, width = target.shape[-2:]
    seed = _seed(config, kind, sample_id, split, epoch, view)
    rng = np.random.Generator(np.random.PCG64(seed))
    observation = np.where(valid, target, np.float32(0.0)).astype(np.float32, copy=True)
    missing = np.zeros_like(valid)
    affected = np.zeros_like(valid)
    patch_missing = np.zeros((height // 16, width // 16), dtype=bool)

    if kind in {"sensor_gaussian", "joint"}:
        sigma = _nonnegative(config, "sigma_celsius", 0.5)
        noise = rng.standard_normal(target.shape).astype(np.float32) * np.float32(sigma)
        observation[valid] += noise[valid]
        if sigma > 0:
            affected |= valid

    if kind in {"sensor_stripe", "joint"}:
        sigma = _nonnegative(config, "stripe_sigma_celsius", 0.25)
        offsets = rng.standard_normal((1, 1, width)).astype(np.float32) * np.float32(sigma)
        observation += offsets * valid
        if sigma > 0:
            affected |= valid

    if kind == "pixel10":
        if "coverage" in config and float(config["coverage"]) != 0.1:
            raise ValueError("pixel10 is frozen at 10%; use a separately named profile for a new task")
        positions = np.flatnonzero(valid.reshape(-1))
        count = int(round(0.1 * positions.size))
        if count:
            missing.reshape(-1)[rng.choice(positions, size=count, replace=False)] = True

    if kind in {"cloud_missing", "cloud_bias", "joint"}:
        field = _smooth_field(rng, height, width, config.get("field_grid", 8))[None]
        support = _rank_support(field, valid, _coverage(config))
        if kind == "cloud_bias":
            magnitude = _nonnegative(config, "bias_celsius", 2.0)
            if support.any() and magnitude > 0:
                selected = field[support]
                spread = float(selected.max() - selected.min())
                amplitude = np.ones_like(selected) if spread <= 1e-12 else (selected - selected.min()) / spread
                sign = np.float32(1.0 if rng.integers(0, 2) else -1.0)
                observation[support] += sign * np.float32(magnitude) * amplitude
                affected |= support
        else:
            missing |= support

    if kind == "patch16":
        if height % 16 or width % 16:
            raise ValueError("patch16 requires image dimensions divisible by 16")
        if "coverage" in config and float(config["coverage"]) != 0.1:
            raise ValueError("patch16 uses round(10% of patches); use a new profile to change it")
        total_patches = patch_missing.size
        count = int(round(0.1 * total_patches))
        if count:
            patch_missing.reshape(-1)[rng.choice(total_patches, size=count, replace=False)] = True
        dropped = np.repeat(np.repeat(patch_missing, 16, axis=0), 16, axis=1)[None]
        missing = dropped & valid

    affected |= missing
    observed = valid & ~missing
    if not observed.any():
        raise ValueError("Corruption removed all valid observations; no silent resampling is allowed")
    observation[~observed] = np.float32(0.0)
    if not np.isfinite(observation[observed]).all():
        raise ValueError("Corruption produced non-finite observed temperatures")
    return {
        "observation": observation,
        "observed_valid": observed.copy(),
        "missing_mask": missing.copy(),
        "affected_mask": affected.copy(),
        "patch_missing": patch_missing,
        "seed": seed,
        "profile": kind,
        "version": CORRUPTION_VERSION,
        "reference_valid_count": int(valid.sum()),
        "observed_valid_count": int(observed.sum()),
        "missing_fraction": float(missing.sum() / valid.sum()),
        "affected_fraction": float(affected.sum() / valid.sum()),
    }
