"""Shared constants, GEE initialization, and utility functions."""

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import ee
import numpy as np


# ---------------------------------------------------------------------------
# GEE initialization
# ---------------------------------------------------------------------------

_ee_initialized = False


def init_gee():
    """Initialize Google Earth Engine."""
    global _ee_initialized
    if _ee_initialized:
        return
    project = os.getenv("EE_PROJECT", "").strip() or "around-438815"
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    _ee_initialized = True
    print(f"GEE initialized (project: {project})")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HLS Prithvi bands (6 reflectance + Fmask)
HLS_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]

# Patch parameters
PATCH_SIZE = 256
MAX_MISSING_RATIO = 0.01   # <1% missing pixels per band
MAX_CLOUD_RATIO = 0.20     # <20% cloudy pixels
MAX_SAMPLES_PER_AREA = 10
MAX_PATCHES_PER_TILE = 92  # ~1MB/patch x 92 x 2176 tiles ~ 200GB

# LST conversion: DN * scale + offset -> Kelvin, then - 273.15 -> Celsius
LST_SCALE = 0.00341802
LST_OFFSET = 149.0

# GEE collection IDs
COL_HLSL30 = "NASA/HLS/HLSL30/v002"
COL_LC08 = "LANDSAT/LC08/C02/T1_L2"
COL_LC09 = "LANDSAT/LC09/C02/T1_L2"
COL_LULC = "COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019"
COL_ECOREGIONS = "RESOLVE/ECOREGIONS/2017"

# Copernicus LC 100m class mapping (merge 12 forest -> 2)
LULC_CLASSES = {
    200: "sea", 80: "permanent_water", 70: "ice",
    60: "bare_vegetation", 40: "managed_vegetation", 50: "urban",
    111: "closed_forest", 112: "closed_forest", 113: "closed_forest",
    114: "closed_forest", 115: "closed_forest", 116: "closed_forest",
    121: "open_forest", 122: "open_forest", 123: "open_forest",
    124: "open_forest", 125: "open_forest", 126: "open_forest",
    30: "herbaceous_vegetation", 20: "shrubs", 90: "wetland", 100: "moss",
}
LULC_CLASS_NAMES = sorted(set(LULC_CLASSES.values()))

# Tile sampling parameters (from Prithvi-EO-2.0 paper)
TILES_PER_CLASS = 100
TOP_TILES_PER_CLASS = 500
URBAN_TILES = 1000
HIGH_ENTROPY_TILES = 1000
TRAIN_RATIO = 0.95
VAL_RATIO = 0.05

# Fmask bits (HLS v2.0)
FMASK_CLOUD_BITS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
FMASK_WATER_BIT = 1 << 5

# GEE rate limiting
GEE_DELAY = 0.3
GEE_RETRY_DELAY = 30
GEE_MAX_RETRIES = 3

HOME_DIR = str(Path.home())
DEFAULT_SCRATCH = os.getenv("SCRATCH", os.path.join(HOME_DIR, "scratch"))
DEFAULT_ARCHIVE = os.getenv("ARCHIVE", os.path.join(HOME_DIR, "archive"))
DEFAULT_WORK_DIR = os.path.join(DEFAULT_SCRATCH, "lst-super-resolution")
DEFAULT_TILE_CSV = os.path.join(DEFAULT_WORK_DIR, "selected_tiles.csv")
DEFAULT_OBS_CSV = os.path.join(DEFAULT_WORK_DIR, "observations.csv")
DEFAULT_DATA_DIR = os.path.join(DEFAULT_SCRATCH, "prithvi_data")
DEFAULT_FINAL_DIR = os.path.join(DEFAULT_ARCHIVE, "prithvi_final")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def shannon_entropy(proportions: Dict[str, float]) -> float:
    h = 0.0
    for p in proportions.values():
        if p > 0:
            h -= p * np.log2(p)
    return h


def safe_getinfo(ee_object, description="", max_attempts=GEE_MAX_RETRIES):
    """getInfo with retry on rate-limit errors."""
    for attempt in range(max_attempts):
        try:
            time.sleep(GEE_DELAY)
            return ee_object.getInfo()
        except ee.EEException as e:
            msg = str(e).lower()
            if any(w in msg for w in ("rate", "quota", "limit", "429")):
                wait = GEE_RETRY_DELAY * (attempt + 1)
                print(f"  Rate limited ({description}), waiting {wait}s...")
                time.sleep(wait)
                continue
            if attempt < max_attempts - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(5)
                continue
            raise
    raise RuntimeError(f"getInfo failed after {max_attempts} attempts: {description}")


def safe_compute_pixels(request, description="", max_attempts=GEE_MAX_RETRIES):
    """computePixels with retry on rate-limit errors."""
    for attempt in range(max_attempts):
        try:
            time.sleep(GEE_DELAY)
            return ee.data.computePixels(request)
        except ee.EEException as e:
            msg = str(e).lower()
            if any(w in msg for w in ("rate", "quota", "limit", "429")):
                wait = GEE_RETRY_DELAY * (attempt + 1)
                print(f"  Rate limited ({description}), waiting {wait}s...")
                time.sleep(wait)
                continue
            if attempt < max_attempts - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(5)
                continue
            raise
    raise RuntimeError(f"computePixels failed: {description}")


def fmask_cloud_fraction(fmask_arr: np.ndarray) -> float:
    cloud = (fmask_arr.astype(np.uint8) & FMASK_CLOUD_BITS) > 0
    return float(cloud.sum()) / max(cloud.size, 1)


def fmask_is_sea(fmask_arr: np.ndarray, threshold: float = 0.95) -> bool:
    water = (fmask_arr.astype(np.uint8) & FMASK_WATER_BIT) > 0
    return float(water.sum()) / max(water.size, 1) >= threshold


