"""Build a separate, verified HLC train/val geolocation sidecar (no image reads)."""
import argparse
import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

PROTOCOL = 'hlc-center224-calendar-geo-adaln-20261004-v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def crop_center(transform, x0, y0):
    """Pixel-edge affine: midpoint of [16,240) is 128, not 127.5."""
    a, b, c, d, e, f = map(float, transform)
    col, row = x0 + 128., y0 + 128.
    return a * col + b * row + c, d * col + e * row + f


def build(root, output):
    from rasterio.warp import transform as project
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite metadata: {output}')
    prepared = root / 'processed_data/v1/lstsr_tb'
    source = root / 'data_inspection/extracted_metadata/lstsr_tb_final_dataset_20260719'
    paths = {name: source / name for name in (
        'phase3_tile_variant_months_merged.csv', 'phase4_dataset_manifest.csv',
        'hls_l8_conditions_2019_2025_lc08/hls_condition_manifest.csv')}
    paths['index.sqlite'] = prepared / 'index.sqlite'
    bundle = json.loads((prepared / 'bundle.json').read_text())
    source_hashes = {name: sha256(path) for name, path in paths.items()}
    if source_hashes['index.sqlite'] != bundle['artifacts']['index.sqlite']:
        raise ValueError('Frozen record index hash mismatch')
    grids = {}
    with paths['phase3_tile_variant_months_merged.csv'].open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key = row['tile_variant_id']
            value = (row['grid_crs'].replace('\\n', '\n'),
                     tuple(json.loads(row['grid_transform'])), int(row['tile_pixels']))
            if key in grids and grids[key] != value:
                raise ValueError(f'Month-dependent geometry: {key}')
            grids[key] = value
    with sqlite3.connect(paths['index.sqlite'].as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        records = {key: (rid, split, tile, json.loads(meta), json.loads(loc))
                   for rid, key, split, tile, meta, loc in db.execute(
                       "SELECT id,key,split,tile,meta,loc FROM records WHERE split IN ('train','val')")}
        max_id = db.execute('SELECT max(id) FROM records').fetchone()[0]
    hls_offsets = {}
    with paths['hls_l8_conditions_2019_2025_lc08/hls_condition_manifest.csv'].open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key = row['sequence_id']
            if key not in records:
                continue
            value = (int(row['x0']), int(row['y0']), int(row['patch_size']), row['mgrs_tile'])
            if key in hls_offsets:
                raise ValueError(f'Duplicate HLS record: {key}')
            hls_offsets[key] = value
    groups, seen = defaultdict(list), set()
    with paths['phase4_dataset_manifest.csv'].open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key = row['sequence_id']
            if key not in records:
                continue
            if key in seen:
                raise ValueError(f'Duplicate LST record: {key}')
            seen.add(key)
            rid, split, tile, meta, loc = records[key]
            x0, y0, size = (int(row[k]) for k in ('x0', 'y0', 'patch_size'))
            assert (x0, y0, size, tile) == hls_offsets[key], f'HLS/LST geometry mismatch: {key}'
            assert size == 256 and split == row['split'] and tile == row['mgrs_tile']
            assert row['archive'] == loc['lst'] and key == meta['sequence_id']
            assert sorted(int(m)-1 for m in row['available_months'].split(',')) == loc['slots']
            assert json.loads(row['month_dates_json']) == meta['month_dates']
            crs, affine, pixels = grids[row['tile_variant_id']]
            assert 0 <= x0 <= pixels-size and 0 <= y0 <= pixels-size
            x, y = crop_center(affine, x0, y0)
            groups[crs].append((rid, x, y))
    if seen != set(records):
        raise ValueError(f'Missing geometry for {len(set(records)-seen)} records')
    # Test records remain NaN and cannot be consumed by the new training loader.
    latlon = np.full((max_id+1, 2), np.nan, dtype=np.float64)
    max_roundtrip = 0.
    for crs, rows in groups.items():
        ids, x, y = zip(*rows)
        lon, lat = project(crs, 'EPSG:4326', x, y)
        back_x, back_y = project('EPSG:4326', crs, lon, lat)
        error = np.hypot(np.asarray(back_x)-x, np.asarray(back_y)-y)
        max_roundtrip = max(max_roundtrip, float(error.max()))
        latlon[np.asarray(ids)] = np.column_stack((lat, lon))
    assert max_roundtrip < .01, max_roundtrip
    selected = latlon[[r[0] for r in records.values()]]
    assert np.isfinite(selected).all() and (np.abs(selected) <= [90, 180]).all()
    splits = {}
    for split in ('train', 'val'):
        entry = bundle['samples'][f'{split}/lst_hr']
        path = prepared / entry['index']
        assert sha256(path) == entry['sha256']
        index = np.load(path, mmap_mode='r', allow_pickle=False)
        ids = np.unique(index[:, 0])
        assert set(map(int, ids)) == {r[0] for r in records.values() if r[1] == split}
        assert np.isfinite(latlon[ids]).all()
        splits[split] = dict(records=len(index), sequences=len(ids), sample_index_sha256=entry['sha256'])
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'latlon_by_record.npy', latlon, allow_pickle=False)
    audit = dict(protocol=PROTOCOL, source_hashes=source_hashes, splits=splits,
                 artifact='latlon_by_record.npy', artifact_sha256=sha256(output/'latlon_by_record.npy'),
                 builder_sha256=sha256(__file__), shape=list(latlon.shape), order=['latitude','longitude'],
                 units='degrees', center_pixel_edge_coordinates=[128,128], crop=[16,240,16,240],
                 coordinate_source='phase3 grid_crs/grid_transform + phase4 x0/y0; HLS offsets checked',
                 max_projected_roundtrip_error_m=max_roundtrip, geographic_crs='EPSG:4326',
                 test_used=False, training_launched=False,
                 limitation='Source georeferencing verified structurally; not an independent image-coregistration audit.')
    (output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output), indent=2))
