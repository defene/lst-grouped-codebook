"""Read-only LST unit audit, including independent Landsat point comparisons.

Run from the project root: python experiments/data/check_lst_encoding.py
Outputs contain public scene identifiers and point responses, never signed assets.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import io
import json
import sys

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eo_data import open_dataset
from eo_data.core import sha256_file

DATA = ROOT / 'data_inspection/extracted_metadata/lstsr_tb_final_dataset_20260719'
STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1'
POINT = 'https://planetarycomputer.microsoft.com/api/data/v1/item/point'
PIXELS = [(r, c) for r in (32, 128, 224) for c in (32, 128, 224)]


def fetch(url, params):
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    return response.json()


def point_check(job):
    item, epsg, x, y, row, col, stored = job
    params = {'collection': 'landsat-c2-l2', 'item': item,
              'assets': 'lwir11', 'coord_crs': f'EPSG:{epsg}', 'unscale': 'false'}
    result = {'item': item, 'row': row, 'col': col, 'stored': stored,
              'input_coordinates': [x, y], 'coord_crs': params['coord_crs']}
    try:
        answer = fetch(f'{POINT}/{x},{y}', params)
        result['response'] = answer
        dn = answer['values'][0]
        if dn is None or dn == 0:
            result['status'] = 'source_nodata'
        else:
            temperature = float(dn) * 0.00341802 + 149.0 - 273.15
            error = stored / 100.0 - temperature
            result.update(status='compared', source_dn=dn,
                          source_celsius=temperature, stored_div_100=stored / 100.0,
                          error_celsius=error, agrees_within_0_011_C=abs(error) <= 0.011)
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        result.update(status='request_failed', error=str(exc))
    return result


def main():
    grids = {}
    months = {}
    with (DATA / 'phase3_tile_variant_months_merged.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            grids.setdefault(row['mgrs_tile'], row)
            if row['tile_variant_id'] == 'T01WCP_v000':
                months[int(row['month'])] = row
    with (DATA / 'phase4_dataset_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        sequence = next(row for row in csv.DictReader(stream)
                        if row['sequence_id'] == 'T01WCP_v000_p0000')

    ds = open_dataset('prithvi', 'lst', 'train')
    scale_counts = ds._connect().execute(
        "SELECT json_extract(meta,'$.lst_scale'),count(*) FROM records GROUP BY 1").fetchall()
    report = {
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Distinguish stored Celsius x100 from original Landsat ST DN',
        'source_code_found_in_delivered_workspace': False,
        'prithvi_all_record_lst_scale_counts': dict(scale_counts),
        'sources': {
            'landsat_formula': 'https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature',
            'modis_formula': 'https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A1',
            'point_api_schema': 'https://planetarycomputer.microsoft.com/api/data/v1/openapi.json',
        },
        'formula': {'landsat_dn_to_kelvin': 'DN * 0.00341802 + 149.0',
                    'landsat_dn_to_celsius': 'DN * 0.00341802 - 124.15',
                    'stored_to_celsius': 'stored / 100.0'},
        'source_hashes': {}, 'cases': [], 'frozen_train_stats': {},
        'normalization_code_changed': False, 'source_lookup_failures': [],
    }
    out = ROOT / 'experiments/data/lst_encoding_verification.json'
    def save():
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for name in ('prithvi', 'lstsr_tb'):
        path = ROOT / 'processed_data/v1' / name / 'normalization.json'
        stats = json.loads(path.read_text())['lst_hr']
        report['source_hashes'][path.relative_to(ROOT).as_posix()] = sha256_file(path)
        report['frozen_train_stats'][name] = {
            'samples': stats['samples'], 'raw_mean': stats['mean'][0], 'raw_std': stats['std'][0],
            'celsius_mean_under_x100_decode': stats['mean'][0] / 100,
            'celsius_std_under_x100_decode': stats['std'][0] / 100,
        }
    for path in (DATA / 'phase3_tile_variant_months_merged.csv', DATA / 'phase4_dataset_manifest.csv'):
        report['source_hashes'][path.relative_to(ROOT).as_posix()] = sha256_file(path)

    # The exact datetime and same HLS tile select candidate source scenes. The
    # shared tile grid is an explicit geolocation hypothesis, tested at all nine
    # preselected pixels; no pixel offset is fitted to improve agreement.
    for key in ('train/T03VXK/obs_0014/r00_c00', 'train/T03WXM/obs_0025/r00_c00'):
        metadata, loc, digest = ds._connect().execute(
            'SELECT meta,loc,digest FROM records WHERE key=?', (key,)).fetchone()
        meta = json.loads(metadata)
        assert meta['hls_image_id'].startswith(meta['tile'] + '_')
        raw = np.load(io.BytesIO(ds._parts(json.loads(loc), digest)[2]), allow_pickle=False)
        dt = datetime.fromisoformat(meta['datetime'].replace('Z', '+00:00'))
        params = {'collections': 'landsat-c2-l2', 'limit': 10,
                  'datetime': f'{(dt-timedelta(seconds=1)).isoformat()}/{(dt+timedelta(seconds=2)).isoformat()}'}
        items = fetch(f'{STAC}/search', params)['features']
        case = {'dataset': 'prithvi', 'sample': key, 'metadata': meta,
                'source_record_sha256': digest, 'source_selection': params,
                'grid_evidence': 'Same HLS tile ID; grid from LSTSR tile manifest; zero-based r00_c00',
                'grid_transform': json.loads(grids[meta['tile']]['grid_transform']),
                'epsg': (32600 if meta['tile'][3] >= 'N' else 32700) + int(meta['tile'][1:3]),
                'patch_origin_pixels': [meta['col'] * 256, meta['row'] * 256],
                'items': [{'id': item['id'], 'datetime': item['properties']['datetime'],
                           'stac_url': f"{STAC}/collections/landsat-c2-l2/items/{item['id']}",
                           'band_metadata': item['assets'].get('lwir11', {}).get('raster:bands')}
                          for item in items], 'points': []}
        run_case(case, raw, raw != -32767)
        # HLS tile metadata can name a scene whose ST footprint misses the
        # patch. Check the immediately adjacent WRS rows on the same date;
        # preserve the initial no-data results and do not fit pixel offsets.
        if case['items'] and not case['best_absolute_errors_celsius']:
            case['initial_points'] = case['points']
            base = case['items'][0]['id'].split('_')
            pathrow = base[2]
            case['source_selection_expansion'] = 'Adjacent WRS rows, same path/date/platform, because initial ST pixels are all unavailable'
            for delta in (-1, 1):
                neighbor = base.copy()
                neighbor[2] = pathrow[:3] + f'{int(pathrow[3:]) + delta:03d}'
                item_id = '_'.join(neighbor)
                try:
                    item = fetch(f'{STAC}/collections/landsat-c2-l2/items/{item_id}', {})
                except requests.RequestException as exc:
                    report['source_lookup_failures'].append({'item': item_id, 'error': str(exc)})
                    continue
                case['items'].append({'id': item_id, 'datetime': item['properties']['datetime'],
                                      'stac_url': f'{STAC}/collections/landsat-c2-l2/items/{item_id}',
                                      'band_metadata': item['assets'].get('lwir11', {}).get('raster:bands')})
            run_case(case, raw, raw != -32767)
        report['cases'].append(case)
        save()
    ds.close()

    # For LSTSR, month source IDs, patch offset, and full grid transform are
    # explicit in the delivered manifests. Compare all listed source scenes.
    path = DATA / sequence['archive']
    report['source_hashes'][path.relative_to(ROOT).as_posix()] = sha256_file(path)
    with np.load(path, allow_pickle=False) as arrays:
        for month in (4, 8):
            assert month in {int(v) for v in sequence['available_months'].split(',')}
            provenance = months[month]
            items = []
            for gee_id in provenance['landsat_scene_ids'].split(';'):
                satellite, pathrow, day = gee_id.rsplit('/', 1)[1].split('_')
                item_id = f'{satellite}_L2SP_{pathrow}_{day}_02_T1'
                try:
                    item = fetch(f'{STAC}/collections/landsat-c2-l2/items/{item_id}', {})
                except requests.RequestException as exc:
                    report['source_lookup_failures'].append({'item': item_id, 'error': str(exc)})
                    continue
                items.append({'id': item_id, 'gee_id': gee_id,
                              'datetime': item['properties']['datetime'],
                              'stac_url': f'{STAC}/collections/landsat-c2-l2/items/{item_id}',
                              'band_metadata': item['assets']['lwir11']['raster:bands']})
            case = {'dataset': 'lstsr_tb', 'sample': sequence['sequence_id'], 'month': month,
                    'grid_evidence': 'Delivered phase3/phase4 manifests',
                    'grid_transform': json.loads(provenance['grid_transform']), 'epsg': 32601,
                    'patch_origin_pixels': [int(sequence['x0']), int(sequence['y0'])],
                    'items': items, 'points': []}
            run_case(case, arrays['hr_lst'][month-1],
                     arrays['hr_valid'][month-1].astype(bool) & (arrays['hr_lst'][month-1] != -32768))
            report['cases'].append(case)
            save()

    # Reuse a documented deterministic sample (2048 train, 256 val, 256 test
    # sequences). MODIS uses 0.02 K/DN; Celsius x100 yields 2*DN-27315, odd.
    report['modis_lr_quantization_evidence'] = []
    for split in ('train', 'val', 'test'):
        path = ROOT / f'data_inspection/normalization/{split}_histograms.npz'
        report['source_hashes'][path.relative_to(ROOT).as_posix()] = sha256_file(path)
        with np.load(path, allow_pickle=False) as arrays:
            counts = arrays['hist'][arrays['names'].tolist().index('lr_lst')]
            values = np.arange(-32768, 32768)
            total = int(counts.sum())
            odd = int(counts[values % 2 == 1].sum())
            report['modis_lr_quantization_evidence'].append(
                {'split': split, 'valid_sampled_pixels': total, 'odd_pixels': odd,
                 'odd_fraction': odd / total, 'interpretation': 'Consistent with 2 * MODIS_DN - 27315; supporting evidence'})

    report['summary'] = {}
    for dataset in ('prithvi', 'lstsr_tb'):
        cases = [v for v in report['cases'] if v['dataset'] == dataset]
        pixels = sum(v['valid_preselected_pixels'] for v in cases)
        matched = sum(v['matched_pixels_any_listed_scene'] for v in cases)
        errors = [e for v in cases for e in v['best_absolute_errors_celsius']]
        unit_agreement = sum(v <= .02 for v in errors)
        report['summary'][dataset] = {
            'patch_month_cases': len(cases), 'valid_preselected_pixels': pixels,
            'pixels_with_available_source_dn': len(errors),
            'pixels_without_available_source_dn': pixels - len(errors),
            'matched_within_0_011_C': matched,
            'matched_within_0_02_C': unit_agreement,
            'max_best_absolute_error_celsius': max(errors) if errors else None,
            'unit_conclusion': 'Celsius x100 supported by independent source pixels' if unit_agreement == len(errors) and len(errors) >= 8 else 'See individual cases; incomplete source match',
            'scope': 'Sample verification, not an all-pixel proof or recovered exporter implementation',
            'tolerance_note': '0.011 C checks one stored integer step; 0.02 C allows small source resampling/quantization differences for a unit check, not a pixel-exact export claim',
        }
    save()
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))
    return report


def run_case(case, raw, valid):
    a, b, c, d, e, f = case['grid_transform']
    x0, y0 = case['patch_origin_pixels']
    pixels = [(row, col) for row, col in PIXELS if valid[row, col]]
    jobs = []
    for item in case['items']:
        for row, col in pixels:
            u, v = x0 + col + .5, y0 + row + .5
            jobs.append((item['id'], case['epsg'], a*u+b*v+c, d*u+e*v+f,
                         row, col, int(raw[row, col])))
    with ThreadPoolExecutor(max_workers=4) as pool:
        case['points'] = list(pool.map(point_check, jobs))
    best = []
    matched = 0
    for row, col in pixels:
        errors = [abs(v['error_celsius']) for v in case['points']
                  if v['row'] == row and v['col'] == col and v['status'] == 'compared']
        if errors:
            best.append(min(errors))
            matched += min(errors) <= .011
    case.update(valid_preselected_pixels=len(pixels), matched_pixels_any_listed_scene=int(matched),
                best_absolute_errors_celsius=best)
    print(case['dataset'], case['sample'], case.get('month'), f'{matched}/{len(pixels)} matched', flush=True)


if __name__ == '__main__':
    main()
