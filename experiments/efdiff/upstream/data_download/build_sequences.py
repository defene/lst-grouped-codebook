"""Phase 2: Build Observations."""

import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import ee
import numpy as np

from common import (
    COL_HLSL30, DEFAULT_OBS_CSV, ensure_dir, init_gee, iso_z, safe_getinfo,
)


def build_observations_for_tile(
    tile_id: str,
    bbox: List[float],
    start_year: int = 2014,
    end_year: int = 2023,
    max_obs: int = 100,
    max_cloud: float = 20.0,
) -> List[str]:
    """
    Find cloud-free HLSL30 observation dates for a tile via GEE.

    Searches quarter-by-quarter, collects clear observation dates,
    deduplicates, and subsamples evenly for temporal diversity.

    Returns list of ISO datetime strings.
    """
    point = ee.Geometry.Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    all_dates: List[int] = []  # milliseconds since epoch

    for year in range(start_year, end_year + 1):
        for q_start in [1, 4, 7, 10]:
            q_end = q_start + 3
            t0 = f"{year}-{q_start:02d}-01"
            t1 = f"{year + 1}-01-01" if q_end > 12 else f"{year}-{q_end:02d}-01"

            try:
                col = (
                    ee.ImageCollection(COL_HLSL30)
                    .filterBounds(point)
                    .filterDate(t0, t1)
                    .filter(ee.Filter.lt("CLOUD_COVERAGE", max_cloud))
                    .limit(10)
                )
                dates = safe_getinfo(
                    col.aggregate_array("system:time_start"),
                    f"{tile_id} {year}Q{(q_start-1)//3+1}",
                )
                if dates:
                    all_dates.extend(dates)
            except Exception:
                continue

    # Deduplicate by date (day-level)
    date_dict: Dict[str, datetime] = {}
    for ms in all_dates:
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        day_key = dt.strftime("%Y-%m-%d")
        if day_key not in date_dict:
            date_dict[day_key] = dt

    unique_obs = sorted(date_dict.values())
    if len(unique_obs) <= max_obs:
        return [iso_z(dt) for dt in unique_obs]

    # Subsample evenly
    indices = np.linspace(0, len(unique_obs) - 1, max_obs, dtype=int)
    return [iso_z(unique_obs[i]) for i in indices]


def _build_obs_for_tile(tile_info):
    """Worker: build observations for one tile."""
    i, tile_row, start_year, end_year, max_obs, max_cloud, total = tile_info
    tile_id = tile_row["mgrs_tile"]
    split = tile_row["split"]
    bbox = [float(tile_row[k]) for k in ("west", "south", "east", "north")]

    print(f"[{i+1}/{total}] {tile_id} ({split}) finding observations...")
    observations = build_observations_for_tile(
        tile_id, bbox, start_year, end_year, max_obs, max_cloud,
    )

    rows = []
    for j, dt_str in enumerate(observations):
        rows.append({
            "mgrs_tile": tile_id, "split": split, "obs_idx": j,
            "datetime": dt_str,
            "west": bbox[0], "south": bbox[1],
            "east": bbox[2], "north": bbox[3],
        })
    print(f"  {tile_id} -> {len(observations)} observations")
    return rows


def cmd_build_sequences(args):
    """CLI: build-sequences (finds individual HLSL30 observations per tile)."""
    init_gee()

    tiles = []
    with open(args.tiles, newline="") as f:
        for row in csv.DictReader(f):
            tiles.append(row)
    print(f"Loaded {len(tiles)} tiles from {args.tiles}")

    workers = getattr(args, "workers", 4)
    max_obs = getattr(args, "max_obs_per_tile", 40)
    print(f"Workers: {workers}, Max obs/tile: {max_obs}")

    fieldnames = ["mgrs_tile", "split", "obs_idx", "datetime",
                   "west", "south", "east", "north"]

    tile_infos = [
        (i, row, args.start_year, args.end_year,
         max_obs, args.max_cloud, len(tiles))
        for i, row in enumerate(tiles)
    ]

    lock = threading.Lock()
    total_obs = [0]
    ensure_dir(Path(args.output).parent)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_build_obs_for_tile, info): info
                for info in tile_infos
            }
            for future in as_completed(futures):
                try:
                    rows = future.result()
                    with lock:
                        writer.writerows(rows)
                        f.flush()
                        total_obs[0] += len(rows)
                except Exception as e:
                    info = futures[future]
                    print(f"  Error [{info[1]['mgrs_tile']}]: {e}")

    print(f"\nSaved {total_obs[0]} observations -> {args.output}")


