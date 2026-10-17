"""Phase 3: Download HLS + LST Patches."""

import csv
import json
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import ee
import numpy as np

from common import (
    BAND_NAMES, COL_HLSL30, COL_LC08, COL_LC09, DEFAULT_DATA_DIR,
    HLS_BANDS, LST_OFFSET, LST_SCALE, MAX_CLOUD_RATIO, MAX_MISSING_RATIO,
    MAX_PATCHES_PER_TILE, MAX_SAMPLES_PER_AREA, PATCH_SIZE, ensure_dir,
    fmask_cloud_fraction, init_gee, safe_compute_pixels, safe_getinfo,
)


def _get_hlsl30_image(bbox: List[float], datetime_str: str, max_cloud: float = 20.0):
    """
    Find the best HLSL30 image for the given bbox and datetime on GEE.

    Returns (ee.Image, image_id, projection_info) or (None, None, None).
    """
    dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
    t0 = (dt - timedelta(days=3)).strftime("%Y-%m-%d")
    t1 = (dt + timedelta(days=4)).strftime("%Y-%m-%d")

    point = ee.Geometry.Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    col = (
        ee.ImageCollection(COL_HLSL30)
        .filterBounds(point)
        .filterDate(t0, t1)
        .filter(ee.Filter.lt("CLOUD_COVERAGE", max_cloud))
        .sort("CLOUD_COVERAGE")
        .limit(1)
    )

    count = safe_getinfo(col.size(), "hlsl30 count")
    if count == 0:
        return None, None, None

    img = ee.Image(col.first())
    img_id = safe_getinfo(img.get("system:index"), "hlsl30 id")

    # Get native projection from B2 band
    proj_info = safe_getinfo(img.select("B2").projection(), "hlsl30 proj")

    return img, img_id, proj_info


def download_hls_patches_compute_pixels(
    image: "ee.Image",
    proj_info: dict,
    patch_size: int = PATCH_SIZE,
) -> List[Dict]:
    """
    Download the full HLSL30 tile via computePixels, then extract
    non-overlapping 256x256 patches with quality filtering.

    Downloads band-by-band to stay under the 50MB computePixels limit.
    (3660x3660 x 1 band x int16 = ~27MB per request)

    Returns list of patch dicts with "bands", "fmask", "row", "col".
    """
    crs = proj_info.get("crs") or proj_info.get("wkt")
    transform = proj_info["transform"]
    # Standard MGRS tile: 3660x3660 at 30m
    tile_w, tile_h = 3660, 3660

    grid = {
        "dimensions": {"width": tile_w, "height": tile_h},
        "affineTransform": {
            "scaleX": transform[0], "shearX": transform[1],
            "translateX": transform[2],
            "shearY": transform[3], "scaleY": transform[4],
            "translateY": transform[5],
        },
        "crsCode": crs,
    }

    # Download each band separately (27MB each, under 50MB limit)
    band_arrays = []
    for band_name in HLS_BANDS:
        expr = image.select(band_name).multiply(10000).toInt16()
        request = {
            "expression": expr,
            "fileFormat": "NUMPY_NDARRAY",
            "grid": grid,
        }
        arr = safe_compute_pixels(request, f"HLS {band_name}")
        band_arrays.append(arr[band_name])  # (H, W) int16
    band_stack = np.stack(band_arrays, axis=0)  # (6, H, W) int16

    # Download Fmask (27MB, under limit)
    fmask_request = {
        "expression": image.select("Fmask").toUint8(),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": grid,
    }
    fmask_arr = safe_compute_pixels(fmask_request, "Fmask")
    fmask = fmask_arr["Fmask"]  # (H, W) uint8

    _, h, w = band_stack.shape
    n_rows = h // patch_size
    n_cols = w // patch_size

    patches = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            y0, x0 = ri * patch_size, ci * patch_size
            y1, x1 = y0 + patch_size, x0 + patch_size

            patch_bands = band_stack[:, y0:y1, x0:x1]  # (6, 256, 256) int16
            patch_fmask = fmask[y0:y1, x0:x1]          # (256, 256) uint8

            # Quality check 1: missing pixels (<1%)
            # HLS nodata: -9999 (scaled to -99990000) or 0 (fill at tile edges)
            # Pixel is nodata if ALL 6 bands == 0, or any band is out of range
            all_zero = np.all(patch_bands == 0, axis=0)  # (256, 256)
            out_of_range = (patch_bands < -500) | (patch_bands > 15000)
            nodata_mask = out_of_range | all_zero[np.newaxis, :, :]
            missing_per_band = nodata_mask.sum(axis=(1, 2)) / (patch_size * patch_size)
            max_missing = float(missing_per_band.max())
            if max_missing > MAX_MISSING_RATIO:
                continue

            # Quality check 2: cloud cover (<20%)
            cloud_ratio = fmask_cloud_fraction(patch_fmask)
            if cloud_ratio > MAX_CLOUD_RATIO:
                continue

            # Clamp to valid range [0, 10000]
            patch_bands = np.clip(patch_bands, 0, 10000)

            patches.append({
                "bands": patch_bands,
                "fmask": patch_fmask,
                "row": ri,
                "col": ci,
                "missing_ratio": max_missing,
                "cloud_ratio": cloud_ratio,
            })

    return patches


def download_lst_compute_pixels(
    bbox: List[float],
    datetime_str: str,
    proj_info: dict,
    window_days: int = 1,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Download Landsat C2 L2 ST_B10 + QA_PIXEL via computePixels, aligned to HLS grid.

    Returns (lst, qa_pixel) tuple:
        lst:      float32 (3660, 3660) in Celsius, or None
        qa_pixel: uint16  (3660, 3660) bit-packed QA, or None

    QA_PIXEL bits (Landsat C2 L2):
        Bit 0: Fill
        Bit 1: Dilated Cloud
        Bit 2: Cirrus
        Bit 3: Cloud
        Bit 4: Cloud Shadow
        Bit 5: Snow
        Bit 6: Clear
        Bit 7: Water
    """
    dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
    t0 = (dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
    t1 = (dt + timedelta(days=window_days + 1)).strftime("%Y-%m-%d")

    crs = proj_info.get("crs") or proj_info.get("wkt")
    transform = proj_info["transform"]

    point = ee.Geometry.Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    try:
        col = (
            ee.ImageCollection(COL_LC08)
            .merge(ee.ImageCollection(COL_LC09))
            .select(["ST_B10", "QA_PIXEL"])
            .filterBounds(point)
            .filterDate(t0, t1)
        )
        count = safe_getinfo(col.size(), "lst count")
        if count == 0:
            return None, None

        # Mosaic: nearest time first
        target_ms = int(dt.timestamp() * 1000)
        img = (
            col.map(lambda i: i.set(
                "t_diff", i.date().millis().subtract(target_ms).abs()))
            .sort("t_diff", False)
            .mosaic()
        )

        grid = {
            "dimensions": {"width": 3660, "height": 3660},
            "affineTransform": {
                "scaleX": transform[0], "shearX": transform[1],
                "translateX": transform[2],
                "shearY": transform[3], "scaleY": transform[4],
                "translateY": transform[5],
            },
            "crsCode": crs,
        }

        # Download ST_B10 as int16
        # Convert DN -> Celsius on server, then x100 -> int16
        # DN * 0.00341802 + 149.0 - 273.15 = Celsius
        # Celsius * 100 -> int16 (range -5000..7000, fits int16)
        # This keeps request under 50MB: 3660x3660x2bytes = 27MB
        lst_i16 = (img.select("ST_B10")
                   .multiply(LST_SCALE).add(LST_OFFSET - 273.15)
                   .multiply(100).toInt16())
        request = {
            "expression": lst_i16,
            "fileFormat": "NUMPY_NDARRAY",
            "grid": grid,
        }
        arr = safe_compute_pixels(request, "LST")
        lst = arr["ST_B10"].astype(np.float32) / 100.0  # back to Celsius

        # Mark invalid
        lst[lst < -50] = np.nan
        lst[lst > 70] = np.nan

        valid_frac = np.sum(np.isfinite(lst)) / max(lst.size, 1)
        if valid_frac < 0.01:
            return None, None

        # Download QA_PIXEL as uint16 (separate request to stay under 50MB)
        qa_pixel = None
        try:
            qa_request = {
                "expression": img.select("QA_PIXEL").toUint16(),
                "fileFormat": "NUMPY_NDARRAY",
                "grid": grid,
            }
            qa_arr = safe_compute_pixels(qa_request, "QA_PIXEL")
            qa_pixel = qa_arr["QA_PIXEL"]  # (3660, 3660) uint16
        except Exception as e:
            print(f"    QA_PIXEL download failed (LST still ok): {e}")

        return lst, qa_pixel

    except Exception as e:
        print(f"    LST download failed: {e}")
        return None, None


def _download_one_observation(obs_info, outdir, enable_lst, state):
    """
    Download and process one observation. Thread-safe via state['lock'].
    Returns number of saved patches.
    """
    i, obs_row = obs_info
    tile_id = obs_row["mgrs_tile"]
    split = obs_row["split"]
    obs_idx = obs_row["obs_idx"]
    ts = obs_row["datetime"]
    key = f"{tile_id}_{obs_idx}"
    bbox = [float(obs_row[k]) for k in ("west", "south", "east", "north")]

    print(f"[{i+1}/{state['total_obs']}] {tile_id} obs#{obs_idx} ({split}) {ts[:10]}")

    with state["lock"]:
        if state["tile_patch_counts"].get(tile_id, 0) >= MAX_PATCHES_PER_TILE:
            return 0

    # Find HLSL30 image
    img, img_id, proj_info = _get_hlsl30_image(bbox, ts)
    if img is None:
        print(f"  [{tile_id}#{obs_idx}] No HLSL30 image")
        return 0

    # Download and extract patches
    try:
        patches = download_hls_patches_compute_pixels(img, proj_info)
    except Exception as e:
        print(f"  [{tile_id}#{obs_idx}] HLS download failed: {e}")
        return 0

    print(f"  [{tile_id}#{obs_idx}] {len(patches)} valid patches")
    if not patches:
        return 0

    # Download LST + QA_PIXEL
    lst_tile = None
    qa_tile = None
    if enable_lst:
        lst_tile, qa_tile = download_lst_compute_pixels(bbox, ts, proj_info)
        if lst_tile is None:
            print(f"  [{tile_id}#{obs_idx}] LST unavailable, skipping")
            return 0

    # Determine which patches to save (thread-safe)
    max_patches = state.get("max_patches", 0)
    positions_to_save = []
    with state["lock"]:
        tile_count = state["tile_patch_counts"].get(tile_id, 0)
        for p in sorted(patches, key=lambda x: (x["row"], x["col"])):
            if max_patches > 0 and state["total_patches"] + len(positions_to_save) >= max_patches:
                break
            if tile_count + len(positions_to_save) >= MAX_PATCHES_PER_TILE:
                break
            area_key = f"{tile_id}_r{p['row']}_c{p['col']}"
            cur = state["patch_counts"].get(area_key, 0)
            if cur >= MAX_SAMPLES_PER_AREA:
                continue
            state["patch_counts"][area_key] = cur + 1
            positions_to_save.append(p)
        state["tile_patch_counts"][tile_id] = tile_count + len(positions_to_save)

    # Save patches (check LST BEFORE writing anything to disk)
    obs_dir = outdir / split / tile_id / f"obs_{int(obs_idx):04d}"
    saved_count = 0
    for p in positions_to_save:
        row, col = p["row"], p["col"]

        # Check LST first — skip patch if LST unavailable
        lst_i16 = None
        if enable_lst and lst_tile is not None:
            h, w = lst_tile.shape
            y1 = row * PATCH_SIZE + PATCH_SIZE
            x1 = col * PATCH_SIZE + PATCH_SIZE
            if y1 > h or x1 > w:
                continue
            lst_patch = lst_tile[row * PATCH_SIZE:y1, col * PATCH_SIZE:x1]
            lst_missing = np.sum(~np.isfinite(lst_patch)) / lst_patch.size
            if lst_missing > MAX_MISSING_RATIO:
                continue
            lst_i16_arr = np.nan_to_num(lst_patch, nan=-9999.0)
            lst_i16 = (lst_i16_arr * 100).clip(-32767, 32767).astype(np.int16)

        # All checks passed — now write to disk
        patch_dir = obs_dir / f"r{row:02d}_c{col:02d}"
        ensure_dir(patch_dir)

        np.save(patch_dir / "bands.npy", p["bands"])  # int16
        np.save(patch_dir / "fmask.npy", p["fmask"])  # uint8

        has_lst = False
        if lst_i16 is not None:
            np.save(patch_dir / "lst.npy", lst_i16)
            has_lst = True

        # Save QA_PIXEL patch
        if qa_tile is not None:
            h_qa, w_qa = qa_tile.shape
            if y1 <= h_qa and x1 <= w_qa:
                qa_patch = qa_tile[row * PATCH_SIZE:y1, col * PATCH_SIZE:x1]
                np.save(patch_dir / "qa_pixel.npy", qa_patch)  # uint16

        meta = {
            "tile": tile_id, "split": split, "obs_idx": int(obs_idx),
            "row": row, "col": col,
            "datetime": ts, "patch_size": PATCH_SIZE,
            "bands": BAND_NAMES,
            "bands_dtype": "int16", "bands_scale": 10000,
            "hls_image_id": img_id,
        }
        if enable_lst:
            meta["has_lst"] = has_lst
            meta["lst_dtype"] = "int16"
            meta["lst_scale"] = 100

        with open(patch_dir / "metadata.json", "w") as mf:
            json.dump(meta, mf, indent=2)

        saved_count += 1

    # Update progress
    with state["lock"]:
        state["done_keys"].add(key)
        state["total_patches"] += saved_count
        state["progress_path"].write_text(json.dumps({
            "completed": sorted(state["done_keys"]),
            "patch_counts": dict(state["patch_counts"]),
            "tile_patch_counts": dict(state["tile_patch_counts"]),
            "total_patches": state["total_patches"],
        }, indent=2))

    print(f"  [{tile_id}#{obs_idx}] Saved {saved_count} (total: {state['total_patches']})")
    return saved_count


def cmd_download_patches(args):
    """CLI: download-patches (parallel, via GEE computePixels)."""
    init_gee()

    enable_lst = not getattr(args, "no_lst", False)
    max_patches = getattr(args, "max_patches", 0) or 0
    workers = getattr(args, "workers", 4)

    if enable_lst:
        print("LST enabled (Landsat C2 L2 ST_B10 via GEE)")
    print(f"Workers: {workers}" + (f", Max patches: {max_patches}" if max_patches else ""))

    observations = []
    with open(args.sequences, newline="") as f:
        for row in csv.DictReader(f):
            observations.append(row)
    print(f"Loaded {len(observations)} observations from {args.sequences}")

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Resume progress
    progress_path = outdir / "download_progress.json"
    done_keys: Set[str] = set()
    patch_counts: Dict[str, int] = defaultdict(int)
    tile_patch_counts: Dict[str, int] = defaultdict(int)
    prev_total = 0

    if progress_path.exists():
        prog = json.loads(progress_path.read_text())
        done_keys = set(prog.get("completed", []))
        patch_counts = defaultdict(int, prog.get("patch_counts", {}))
        tile_patch_counts = defaultdict(int, prog.get("tile_patch_counts", {}))
        prev_total = prog.get("total_patches", 0)

    state = {
        "lock": threading.Lock(),
        "done_keys": done_keys,
        "patch_counts": patch_counts,
        "tile_patch_counts": tile_patch_counts,
        "total_patches": prev_total,
        "total_obs": len(observations),
        "progress_path": progress_path,
        "max_patches": max_patches,
    }

    pending = [
        (i, row) for i, row in enumerate(observations)
        if f"{row['mgrs_tile']}_{row['obs_idx']}" not in done_keys
    ]
    random.shuffle(pending)
    print(f"Pending: {len(pending)} (skipping {len(observations) - len(pending)} done)")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_one_observation, item, outdir, enable_lst, state,
            ): item
            for item in pending
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                _, row = futures[future]
                print(f"  Error [{row['mgrs_tile']}#{row['obs_idx']}]: {e}")

            if max_patches > 0:
                with state["lock"]:
                    if state["total_patches"] >= max_patches:
                        print(f"\nReached {max_patches} patches, stopping.")
                        for fut in futures:
                            fut.cancel()
                        break

    print(f"\nDone -- {state['total_patches']} patches saved")


