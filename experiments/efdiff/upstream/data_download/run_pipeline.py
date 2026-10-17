"""
Prithvi-EO-2.0 Full Pipeline - All 4 Phases

Phase 1 skipped if selected_tiles.csv exists.
Phase 2 skipped if observations.csv exists.
Phase 3 is resumable (progress saved in $SCRATCH/prithvi_data/download_progress.json).
"""
import os
import sys
import time
from argparse import Namespace

from common import DEFAULT_DATA_DIR, DEFAULT_FINAL_DIR, DEFAULT_OBS_CSV, DEFAULT_TILE_CSV
from select_tiles import cmd_select_tiles
from build_sequences import cmd_build_sequences
from download_patches import cmd_download_patches
from finalize import cmd_finalize

HOME_DIR = os.path.expanduser("~")
SCRATCH_DIR = os.getenv("SCRATCH", os.path.join(HOME_DIR, "scratch"))
ARCHIVE_DIR = os.getenv("ARCHIVE", os.path.join(HOME_DIR, "archive"))
WORK_DIR = os.path.join(SCRATCH_DIR, "lst-super-resolution")
TILES = os.path.join(WORK_DIR, "selected_tiles.csv")
OBS = os.path.join(WORK_DIR, "observations.csv")
DATA = os.path.join(SCRATCH_DIR, "prithvi_data")
FINAL = os.path.join(ARCHIVE_DIR, "prithvi_final")

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
os.makedirs(FINAL, exist_ok=True)


def timed(label, func):
    print(f"\n{'=' * 60}")
    print(f"[{label}] Starting...")
    t0 = time.time()
    try:
        func()
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)
    elapsed = time.time() - t0
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"  [{label}] Elapsed: {h}h{m:02d}m{s:02d}s")


# Phase 1
if os.path.exists(TILES):
    print(f"\n{'=' * 60}\n[Phase 1] SKIP - {TILES} already exists")
else:
    timed("Phase 1", lambda: cmd_select_tiles(Namespace(
        output=TILES,
        force_rediscover=False,
        lulc_tif=None,
        eco_shp=None,
        seed=42,
    )))

# Phase 2
if os.path.exists(OBS):
    print(f"\n{'=' * 60}\n[Phase 2] SKIP - {OBS} already exists")
else:
    timed("Phase 2", lambda: cmd_build_sequences(Namespace(
        tiles=TILES,
        start_year=2014,
        end_year=2026,
        max_obs_per_tile=48,
        max_cloud=20.0,
        output=OBS,
        workers=4,
    )))

# Phase 3
timed("Phase 3", lambda: cmd_download_patches(Namespace(
    sequences=OBS,
    outdir=DATA,
    no_lst=False,
    max_patches=0,
    workers=4,
)))

# Phase 4
timed("Phase 4", lambda: cmd_finalize(Namespace(
    datadir=DATA,
    output=FINAL,
)))

print(f"\n{'=' * 60}")
print(f"DONE - Dataset in {FINAL}/")
