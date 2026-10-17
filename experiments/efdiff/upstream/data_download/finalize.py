"""Phase 4: Finalization."""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import (
    BAND_NAMES, DEFAULT_FINAL_DIR, MAX_CLOUD_RATIO, MAX_MISSING_RATIO,
    PATCH_SIZE, ensure_dir, fmask_is_sea,
)


def cmd_finalize(args):
    """
    CLI: finalize
    1. Verify train/val/test spatial non-overlap
    2. Subsample ocean/desert to 10%
    3. Write manifest + sample lists
    """
    datadir = Path(args.datadir)
    outdir = Path(args.output)
    ensure_dir(outdir)

    print("Scanning patch metadata...")
    all_patches = []
    for meta_path in datadir.rglob("metadata.json"):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["_path"] = str(meta_path.parent)
        all_patches.append(meta)

    print(f"  Found {len(all_patches)} patches")

    train_patches = [p for p in all_patches if p.get("split") == "train"]
    val_patches = [p for p in all_patches if p.get("split") == "val"]
    print(f"  Train: {len(train_patches)}, Val: {len(val_patches)}")

    # Step 1: Spatial non-overlap — drop train samples overlapping val areas
    print("\nStep 1: Ensuring spatial non-overlap...")
    held_out = set()
    for p in val_patches:
        held_out.add(f"{p['tile']}_r{p['row']}_c{p['col']}")

    train_filtered = [
        p for p in train_patches
        if f"{p['tile']}_r{p['row']}_c{p['col']}" not in held_out
    ]
    print(f"  Dropped {len(train_patches) - len(train_filtered)} overlapping train samples")
    train_patches = train_filtered

    # Step 2: Subsample ocean/desert to 10%
    print("\nStep 2: Subsampling ocean/desert...")
    sea_patches, desert_patches, normal_patches = [], [], []

    for p in train_patches:
        patch_dir = Path(p["_path"])
        is_sea = False
        is_desert = False

        fmask_path = patch_dir / "fmask.npy"
        if fmask_path.exists():
            fmask = np.load(fmask_path)
            is_sea = fmask_is_sea(fmask, threshold=0.95)

        if not is_sea:
            bands_path = patch_dir / "bands.npy"
            if bands_path.exists():
                bands = np.load(bands_path).astype(np.float32) / 10000.0
                nir, red, swir1 = bands[3], bands[2], bands[4]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red), 0.0)
                low_veg = float((ndvi < 0.1).sum()) / max(ndvi.size, 1)
                if low_veg > 0.8 and float(np.nanmean(swir1)) > 0.25:
                    is_desert = True

        if is_sea:
            sea_patches.append(p)
        elif is_desert:
            desert_patches.append(p)
        else:
            normal_patches.append(p)

    print(f"  Sea: {len(sea_patches)}, Desert: {len(desert_patches)}, "
          f"Normal: {len(normal_patches)}")

    sea_keep = max(1, len(sea_patches) // 10)
    desert_keep = max(1, len(desert_patches) // 10)
    if len(sea_patches) > sea_keep:
        random.shuffle(sea_patches)
        sea_patches = sea_patches[:sea_keep]
    if len(desert_patches) > desert_keep:
        random.shuffle(desert_patches)
        desert_patches = desert_patches[:desert_keep]

    final_train = normal_patches + sea_patches + desert_patches
    print(f"\nFinal: Train={len(final_train)}, Val={len(val_patches)}")

    # Write manifest
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patch_size": PATCH_SIZE,
        "bands": BAND_NAMES,
        "bands_dtype": "int16", "bands_scale": 10000,
        "quality_filters": {
            "max_missing_ratio": MAX_MISSING_RATIO,
            "max_cloud_ratio": MAX_CLOUD_RATIO,
        },
        "subsampling": {"sea_ratio": 0.10, "desert_ratio": 0.10},
        "train_count": len(final_train),
        "val_count": len(val_patches),
    }

    for name, data in [("train", final_train), ("val", val_patches)]:
        with open(outdir / f"{name}_samples.json", "w") as f:
            json.dump([p["_path"] for p in data], f, indent=2)

    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved to {outdir}/")

