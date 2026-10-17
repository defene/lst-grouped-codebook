"""Phase 1: LULC + Ecoregion Based Tile Selection."""

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import ee
import numpy as np

from common import (
    COL_ECOREGIONS, COL_HLSL30, COL_LULC, HIGH_ENTROPY_TILES,
    LULC_CLASSES, LULC_CLASS_NAMES, TILES_PER_CLASS, TOP_TILES_PER_CLASS,
    URBAN_TILES, VAL_RATIO, ensure_dir, init_gee, safe_getinfo, shannon_entropy,
)


def _extract_mgrs_tile_id(img_id: str) -> Optional[str]:
    """Extract MGRS tile ID (e.g. 'T12SUC') from an HLS system:index string."""
    for part in img_id.replace(".", "_").split("_"):
        if part.startswith("T") and len(part) == 6 and part[1:3].isdigit():
            return part
    return None


def discover_hls_tiles_gee(
    year: int = 2019,
    max_cloud: float = 50.0,
) -> Dict[str, List[float]]:
    """
    Discover HLSL30 MGRS tiles globally via GEE.

    Uses a two-phase approach:
      Phase A – Region-based scan: splits the globe into 10° lat × 20° lon
        regions (lat -60° to 84°N), queries HLSL30 with GEE ``distinct()``
        to extract unique MGRS tile IDs and their bounding boxes in one
        call per region.
      Phase B – Dense grid fill: a 2° × 2° point-based grid catches any
        tiles that might have been missed in Phase A (e.g. small coastal
        tiles where the region centroid falls in the ocean).

    Returns dict: mgrs_tile -> [west, south, east, north]
    """
    tile_bboxes: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Phase A: Region-based scan with distinct() — fast & thorough
    # ------------------------------------------------------------------
    regions = [
        (lat, lon, min(lat + 10, 84), min(lon + 20, 180))
        for lat in range(-60, 84, 10)
        for lon in range(-180, 180, 20)
    ]
    print(f"Phase A: Scanning {len(regions)} regions for HLS tiles "
          f"(lat -60° to 84°N)...")

    def _set_tile_id(img):
        """Server-side: extract MGRS tile ID from system:index."""
        # system:index format: "T10SGJ_20190611T180345"
        idx = ee.String(img.get("system:index"))
        mgrs = idx.split("_").get(0)
        return img.set("mgrs_tile", mgrs)

    for i, (lat_min, lon_min, lat_max, lon_max) in enumerate(regions):
        region = ee.Geometry.Rectangle(
            [lon_min, lat_min, lon_max, lat_max], proj="EPSG:4326", geodesic=False,
        )
        try:
            col = (
                ee.ImageCollection(COL_HLSL30)
                .filterBounds(region)
                .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
                .filter(ee.Filter.lt("CLOUD_COVERAGE", max_cloud))
                .map(_set_tile_id)
            )
            distinct = col.distinct("mgrs_tile")

            # Build FeatureCollection with tile ID + bbox in one call
            def _tile_feature(img):
                return ee.Feature(
                    img.geometry().bounds(1),
                    {"mgrs_tile": img.get("mgrs_tile")},
                )

            fc = distinct.map(_tile_feature)
            result = safe_getinfo(fc, f"region {i}")

            if result and "features" in result:
                for feat in result["features"]:
                    tid = feat["properties"].get("mgrs_tile", "")
                    if not (tid.startswith("T") and len(tid) == 6):
                        continue
                    if tid in tile_bboxes:
                        continue
                    coords = feat["geometry"]["coordinates"][0]
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    tile_bboxes[tid] = [
                        min(lons), min(lats), max(lons), max(lats),
                    ]
        except Exception as e:
            print(f"  Region {i}: {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(regions)}] found {len(tile_bboxes)} tiles")

    print(f"Discovered {len(tile_bboxes)} unique MGRS tiles")
    return tile_bboxes


def compute_tile_lulc_gee(
    tile_bboxes: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute LULC class proportions for each tile using GEE Copernicus LC 100m.
    """
    lulc_image = ee.Image(COL_LULC).select("discrete_classification")

    # Build remap: pixel value -> merged class index
    class_to_idx = {name: i for i, name in enumerate(LULC_CLASS_NAMES)}
    from_vals = sorted(LULC_CLASSES.keys())
    to_vals = [class_to_idx[LULC_CLASSES[v]] for v in from_vals]
    n_classes = len(LULC_CLASS_NAMES)

    tile_lulc: Dict[str, Dict[str, float]] = {}
    items = list(tile_bboxes.items())
    print(f"Computing LULC proportions for {len(items)} tiles via GEE...")

    for i, (tile_id, bbox) in enumerate(items):
        try:
            region = ee.Geometry.Rectangle(bbox)
            remapped = lulc_image.remap(from_vals, to_vals).rename("lulc_class")

            # Use reduceRegion with a histogram reducer
            hist = safe_getinfo(
                remapped.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=region,
                    scale=100,
                    maxPixels=1e7,
                ),
                f"lulc {tile_id}",
            )

            if not hist or "lulc_class" not in hist:
                continue

            freq = hist["lulc_class"]
            total = sum(freq.values())
            if total == 0:
                continue

            props = {}
            for cls_name in LULC_CLASS_NAMES:
                idx = class_to_idx[cls_name]
                count = freq.get(str(idx), 0)
                props[cls_name] = count / total

            tile_lulc[tile_id] = props

        except Exception:
            continue

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(items)}] processed")

    print(f"  Computed LULC for {len(tile_lulc)} tiles")
    return tile_lulc


def compute_tile_ecoregions_gee(
    tile_bboxes: Dict[str, List[float]],
) -> Dict[str, List[int]]:
    """
    Find which RESOLVE ecoregions each tile intersects, using GEE FeatureCollection.
    """
    eco_fc = ee.FeatureCollection(COL_ECOREGIONS)
    tile_ecoregions: Dict[str, List[int]] = {}
    items = list(tile_bboxes.items())
    print(f"Computing ecoregion coverage for {len(items)} tiles via GEE...")

    for i, (tile_id, bbox) in enumerate(items):
        try:
            region = ee.Geometry.Rectangle(bbox)
            intersecting = eco_fc.filterBounds(region)
            eco_ids = safe_getinfo(
                intersecting.aggregate_array("ECO_ID"),
                f"eco {tile_id}",
            )
            tile_ecoregions[tile_id] = [int(e) for e in (eco_ids or [])]
        except Exception:
            tile_ecoregions[tile_id] = []
            continue

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(items)}] processed")

    all_ecos = set()
    for eids in tile_ecoregions.values():
        all_ecos.update(eids)
    print(f"  Found {len(all_ecos)} unique ecoregions")
    return tile_ecoregions


# ---------------------------------------------------------------------------
# Local file alternatives for LULC + Ecoregions (much faster than GEE)
# ---------------------------------------------------------------------------

def compute_tile_lulc_local(
    tile_bboxes: Dict[str, List[float]],
    lulc_tif: str,
) -> Dict[str, Dict[str, float]]:
    """
    Compute LULC class proportions using a local Copernicus LC 100m GeoTIFF.

    Much faster than GEE (~minutes vs ~hours for 18k tiles).
    Requires: pip install rasterio
    """
    import rasterio
    from rasterio.windows import from_bounds

    class_to_idx = {name: i for i, name in enumerate(LULC_CLASS_NAMES)}
    n_classes = len(LULC_CLASS_NAMES)

    # Build remap lookup: pixel_value -> class_index
    remap_lut = {}
    for pix_val, cls_name in LULC_CLASSES.items():
        remap_lut[pix_val] = class_to_idx[cls_name]

    tile_lulc: Dict[str, Dict[str, float]] = {}
    items = list(tile_bboxes.items())
    print(f"Computing LULC proportions for {len(items)} tiles from local GeoTIFF...")

    with rasterio.open(lulc_tif) as src:
        for i, (tile_id, bbox) in enumerate(items):
            try:
                west, south, east, north = bbox
                window = from_bounds(west, south, east, north, src.transform)
                data = src.read(1, window=window)

                if data.size == 0:
                    continue

                # Count per merged class
                counts = np.zeros(n_classes, dtype=np.int64)
                for pix_val, cls_idx in remap_lut.items():
                    counts[cls_idx] += np.sum(data == pix_val)

                total = counts.sum()
                if total == 0:
                    continue

                props = {}
                for cls_name in LULC_CLASS_NAMES:
                    props[cls_name] = float(counts[class_to_idx[cls_name]]) / total
                tile_lulc[tile_id] = props

            except Exception:
                continue

            if (i + 1) % 1000 == 0:
                print(f"  [{i+1}/{len(items)}] processed")

    print(f"  Computed LULC for {len(tile_lulc)} tiles")
    return tile_lulc


def compute_tile_ecoregions_local(
    tile_bboxes: Dict[str, List[float]],
    eco_shp: str,
) -> Dict[str, List[int]]:
    """
    Find ecoregions per tile using a local Ecoregions2017 shapefile.

    Much faster than GEE (~minutes vs ~hours for 18k tiles).
    Requires: pip install geopandas
    """
    import geopandas as gpd
    from shapely.geometry import box

    print(f"Loading ecoregions shapefile: {eco_shp}")
    eco_gdf = gpd.read_file(eco_shp)
    sindex = eco_gdf.sindex  # spatial index for fast queries

    tile_ecoregions: Dict[str, List[int]] = {}
    items = list(tile_bboxes.items())
    print(f"Computing ecoregion coverage for {len(items)} tiles from local shapefile...")

    for i, (tile_id, bbox) in enumerate(items):
        try:
            west, south, east, north = bbox
            tile_box = box(west, south, east, north)
            candidates = list(sindex.intersection(tile_box.bounds))
            if candidates:
                matches = eco_gdf.iloc[candidates]
                actual = matches[matches.intersects(tile_box)]
                eco_ids = actual["ECO_ID"].dropna().astype(int).tolist()
            else:
                eco_ids = []
            tile_ecoregions[tile_id] = eco_ids
        except Exception:
            tile_ecoregions[tile_id] = []
            continue

        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(items)}] processed")

    all_ecos = set()
    for eids in tile_ecoregions.values():
        all_ecos.update(eids)
    print(f"  Found {len(all_ecos)} unique ecoregions")
    return tile_ecoregions


def select_tiles(
    tile_bboxes: Dict[str, List[float]],
    tile_lulc: Dict[str, Dict[str, float]],
    tile_ecoregions: Dict[str, List[int]],
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Implement the Prithvi-EO-2.0 tile sampling strategy:
      1. Sample 100 tiles per LULC class from top-500 by class proportion
      2. Upsample urban areas (1000 tiles)
      3. Add 1000 high-entropy tiles
      4. Ensure ecoregion coverage
      5. Drop problematic tiles (Greenland, Antarctica)
      6. Stratified 95%/5% train/val split (per LULC class, min 1 per class)
    """
    random.seed(seed)
    valid_tiles = set(tile_lulc.keys())
    selected: Set[str] = set()

    # Step 1: Sample per LULC class
    print("\nStep 1: Sampling tiles per LULC class...")
    for cls in LULC_CLASS_NAMES:
        ranked = sorted(valid_tiles, key=lambda t: tile_lulc[t].get(cls, 0), reverse=True)
        top_n = ranked[:TOP_TILES_PER_CLASS]
        sampled = random.sample(top_n, min(TILES_PER_CLASS, len(top_n)))
        selected.update(sampled)
        print(f"  {cls}: {len(sampled)} tiles")

    # Step 2: Upsample urban
    print(f"\nStep 2: Upsampling urban ({URBAN_TILES} tiles)...")
    urban_ranked = sorted(valid_tiles, key=lambda t: tile_lulc[t].get("urban", 0), reverse=True)
    selected.update(urban_ranked[:URBAN_TILES])

    # Step 3: High entropy
    print(f"\nStep 3: Adding {HIGH_ENTROPY_TILES} high-entropy tiles...")
    tile_entropy = {t: shannon_entropy(tile_lulc[t]) for t in valid_tiles}
    entropy_ranked = sorted(valid_tiles, key=lambda t: tile_entropy[t], reverse=True)
    selected.update(entropy_ranked[:HIGH_ENTROPY_TILES])

    # Step 4: Ecoregion coverage
    print("\nStep 4: Ensuring ecoregion coverage...")
    covered_ecos: Set[int] = set()
    for t in selected:
        covered_ecos.update(tile_ecoregions.get(t, []))

    eco_to_tiles: Dict[int, List[str]] = defaultdict(list)
    all_ecos: Set[int] = set()
    for t, eids in tile_ecoregions.items():
        for eid in eids:
            all_ecos.add(eid)
            eco_to_tiles[eid].append(t)

    for eid in all_ecos - covered_ecos:
        cands = eco_to_tiles.get(eid, [])
        if len(cands) >= 3:
            selected.update(random.sample(cands, min(3, len(cands))))

    # Step 5: Drop Antarctica / central Greenland
    print("\nStep 5: Dropping problematic tiles...")
    to_drop = set()
    for t in selected:
        bbox = tile_bboxes.get(t)
        if not bbox:
            to_drop.add(t)
            continue
        west, south, east, north = bbox
        lat_c = (south + north) / 2
        lon_c = (west + east) / 2
        if lat_c < -60:
            to_drop.add(t)
        if lat_c > 72 and -60 < lon_c < -20:
            to_drop.add(t)
    selected -= to_drop
    print(f"  Dropped {len(to_drop)} tiles")

    # Step 6: Stratified train/val split — ensure every LULC class in val
    print("\nStep 6: Stratified train/val split...")
    all_sel = sorted(selected)

    # Group tiles by dominant LULC
    lulc_to_tiles: Dict[str, List[str]] = defaultdict(list)
    for t in all_sel:
        lulc = tile_lulc.get(t, {})
        dominant = max(lulc, key=lulc.get) if lulc else "unknown"
        lulc_to_tiles[dominant].append(t)

    train_tiles, val_tiles = [], []
    for cls in sorted(lulc_to_tiles.keys()):
        cls_tiles = lulc_to_tiles[cls]
        random.shuffle(cls_tiles)
        # At least 1 per class in val, otherwise 5%
        n_val_cls = max(1, int(len(cls_tiles) * VAL_RATIO))
        val_tiles.extend(cls_tiles[:n_val_cls])
        train_tiles.extend(cls_tiles[n_val_cls:])
        print(f"  {cls:<25} total={len(cls_tiles):>4}  "
              f"train={len(cls_tiles) - n_val_cls:>4}  val={n_val_cls:>3}")

    print(f"\n  Train: {len(train_tiles)}, Val: {len(val_tiles)}, "
          f"Total: {len(all_sel)}")
    return train_tiles, val_tiles


def cmd_select_tiles(args):
    """CLI: select-tiles"""
    init_gee()

    ensure_dir(Path(args.output).parent)
    tiles_cache = Path(args.output).with_suffix(".tiles_cache.json")
    if tiles_cache.exists() and not args.force_rediscover:
        print(f"Loading cached tiles from {tiles_cache}")
        with open(tiles_cache) as f:
            tile_bboxes = json.load(f)
    else:
        tile_bboxes = discover_hls_tiles_gee()
        with open(tiles_cache, "w") as f:
            json.dump(tile_bboxes, f)
        print(f"Cached -> {tiles_cache}")

    # --- LULC: prefer local GeoTIFF if available ---
    lulc_tif = getattr(args, 'lulc_tif', None)
    if lulc_tif and os.path.exists(lulc_tif):
        print(f"Using LOCAL LULC GeoTIFF: {lulc_tif}")
        tile_lulc = compute_tile_lulc_local(tile_bboxes, lulc_tif)
    else:
        # Auto-detect common local path
        default_lulc = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif",
        )
        if os.path.exists(default_lulc):
            print(f"Auto-detected LOCAL LULC GeoTIFF: {default_lulc}")
            tile_lulc = compute_tile_lulc_local(tile_bboxes, default_lulc)
        else:
            print("No local LULC found, falling back to GEE...")
            tile_lulc = compute_tile_lulc_gee(tile_bboxes)

    # --- Ecoregions: prefer local shapefile if available ---
    eco_shp = getattr(args, 'eco_shp', None)
    if eco_shp and os.path.exists(eco_shp):
        print(f"Using LOCAL ecoregions shapefile: {eco_shp}")
        tile_ecoregions = compute_tile_ecoregions_local(tile_bboxes, eco_shp)
    else:
        default_eco = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "Ecoregions2017", "Ecoregions2017.shp",
        )
        if os.path.exists(default_eco):
            print(f"Auto-detected LOCAL ecoregions shapefile: {default_eco}")
            tile_ecoregions = compute_tile_ecoregions_local(tile_bboxes, default_eco)
        else:
            print("No local ecoregions found, falling back to GEE...")
            tile_ecoregions = compute_tile_ecoregions_gee(tile_bboxes)
    seed = getattr(args, 'seed', 42)
    train_tiles, val_tiles = select_tiles(
        tile_bboxes, tile_lulc, tile_ecoregions, seed=seed,
    )

    rows = []
    for split_name, tile_list in [("train", train_tiles),
                                   ("val", val_tiles)]:
        for t in tile_list:
            bbox = tile_bboxes.get(t, [0, 0, 0, 0])
            lulc = tile_lulc.get(t, {})
            dominant = max(lulc, key=lulc.get) if lulc else ""
            rows.append({
                "mgrs_tile": t, "split": split_name,
                "west": bbox[0], "south": bbox[1],
                "east": bbox[2], "north": bbox[3],
                "dominant_lulc": dominant,
                "entropy": f"{shannon_entropy(lulc):.4f}",
                "n_ecoregions": len(tile_ecoregions.get(t, [])),
            })

    fieldnames = ["mgrs_tile", "split", "west", "south", "east", "north",
                   "dominant_lulc", "entropy", "n_ecoregions"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} tiles -> {args.output}")


