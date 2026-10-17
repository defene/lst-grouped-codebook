"""Read-only full membership/metadata audit; does not decompress pixel arrays."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3

import numpy as np


def audit(root):
    source = root / 'processed_data/v1/prithvi'
    connection = sqlite3.connect((source / 'index.sqlite').as_uri() + '?mode=ro', uri=True)
    report = {
        'dataset': 'prithvi',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Complete paired sample indices and source metadata, not a pixel registration audit',
        'cross_view_rule': 'HLS source tile equals record tile and timestamp difference < 1 second',
        'splits': {},
    }
    split_tiles = {}
    split_ids = {}
    for split in ('train', 'val'):
        hls = np.load(source / 'indices' / f'{split}.hls.npy', mmap_mode='r')[:, 0]
        lst = np.load(source / 'indices' / f'{split}.lst_hr.npy', mmap_mode='r')[:, 0]
        assert len(np.unique(hls)) == len(hls) and len(np.unique(lst)) == len(lst)
        pairs = np.intersect1d(hls, lst)
        remaining = set(map(int, pairs))
        split_ids[split] = remaining.copy()
        counts, tiles, examples = Counter(), set(), []
        for record_id, key, tile, raw_meta in connection.execute(
                'SELECT id,key,tile,meta FROM records WHERE split=?', (split,)):
            if record_id not in remaining:
                continue
            remaining.remove(record_id)
            meta = json.loads(raw_meta)
            assert meta['tile'] == tile and meta['split'] == split
            source_tile, stamp = meta['hls_image_id'].rsplit('_', 1)
            hls_time = datetime.strptime(stamp, '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
            recorded = datetime.fromisoformat(meta['datetime'].replace('Z', '+00:00'))
            delta = (hls_time - recorded).total_seconds()
            tile_match, time_match = source_tile == tile, abs(delta) < 1
            counts['metadata_eligible_for_cross_view'] += int(tile_match and time_match)
            counts['source_tile_mismatch'] += int(not tile_match)
            counts['source_time_mismatch'] += int(not time_match)
            counts['both_tile_and_time_mismatch'] += int(not tile_match and not time_match)
            if not (tile_match and time_match) and len(examples) < 5:
                examples.append({'record_id': record_id, 'key': key, 'record_tile': tile,
                                 'hls_source_tile': source_tile, 'time_delta_seconds': delta})
            tiles.add(tile)
        assert not remaining, 'Paired index points to absent source records'
        split_tiles[split] = tiles
        report['splits'][split] = {
            'hls_samples': len(hls), 'lst_samples': len(lst), 'pairs': len(pairs),
            'hls_unpaired': len(hls) - len(pairs), 'lst_unpaired': len(lst) - len(pairs),
            'tiles': len(tiles),
            'paired_record_ids_sha256': hashlib.sha256(pairs.astype('<i8').tobytes()).hexdigest(),
            **counts,
            'metadata_eligible_fraction': counts['metadata_eligible_for_cross_view'] / len(pairs),
            'ineligible_examples': examples,
        }
    report['train_val_record_overlap'] = len(split_ids['train'] & split_ids['val'])
    report['train_val_record_tile_overlap'] = len(split_tiles['train'] & split_tiles['val'])
    assert report['train_val_record_overlap'] == 0
    assert report['train_val_record_tile_overlap'] == 0
    connection.close()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('paired_metadata_audit.json'))
    args = parser.parse_args()
    result = audit(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({s: {k: v for k, v in value.items() if k != 'ineligible_examples'}
                      for s, value in result['splits'].items()}, indent=2))
