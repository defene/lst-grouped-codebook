"""Extract split-local QA connected components; keep provenance and hashes."""
from collections import Counter
from pathlib import Path
import hashlib
import io
import json
import numpy as np
from scipy import ndimage
from eo_data import open_dataset
from eo_data.core import sha256_file, json_write


def build_bank(root, destination, split, max_records=4096, max_shapes=256, seed=20260910):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if split not in ('train', 'val', 'test'):
        raise ValueError('Expected HLP v2 train/val/test')
    if max_records < 1 or max_shapes < 1:
        raise ValueError('Positive bank bounds required')
    destination.mkdir(parents=True, exist_ok=True)
    archive, manifest = destination/f'{split}.npz', destination/f'{split}.json'
    if archive.exists() or manifest.exists():
        raise FileExistsError(f'Bank already exists: {destination / split}')
    ds = open_dataset('prithvi', 'hls', split, root=root, prepared=root/'processed_data/hlp_split_v2_20260914')
    positions = np.random.default_rng(seed).permutation(len(ds))[:max_records]
    shapes, records, seen = [], [], set()
    scanned = 0
    for index in positions:
        _, rid, _, _, row = ds._record(int(index))
        key, tile, _, loc, digest = row
        parts = ds._parts(json.loads(loc), digest)
        fmask = np.load(io.BytesIO(parts[1]), allow_pickle=False)
        scanned += 1
        for label, bit in (('cloud', 2), ('shadow', 8), ('adjacency', 4)):
            components, count = ndimage.label((fmask != 255) & ((fmask & bit) != 0))
            sizes = np.bincount(components.ravel())
            candidates = np.argsort(sizes[1:])[::-1][:2]+1
            for component in candidates:
                size = int(sizes[component])
                if size < 32 or size > int(fmask.size*.8):
                    continue
                mask = components == component
                digest_mask = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
                if digest_mask in seen:
                    continue
                seen.add(digest_mask)
                shapes.append(mask)
                records.append({'shape_id': digest_mask, 'class': label, 'source_key': key,
                                'record_id': rid, 'tile': tile, 'split': split,
                                'source_record_sha256': digest, 'pixels': size})
                if len(shapes) >= max_shapes:
                    break
            if len(shapes) >= max_shapes:
                break
        if len(shapes) >= max_shapes:
            break
    if not shapes:
        raise ValueError('No QA components found; increase scan bound or explicitly select procedural_missing')
    temporary = archive.with_suffix('.npz.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, masks=np.stack(shapes))
    temporary.replace(archive)
    result = {'version': 1, 'dataset': 'prithvi', 'split': split, 'seed': seed,
              'scanned_records': scanned, 'max_records': max_records, 'max_shapes': max_shapes,
              'source_index_sha256': ds.fingerprint['source_index'],
              'source_sample_index_sha256': ds.fingerprint['sample_index'],
              'sha256': sha256_file(archive), 'shapes': records,
              'class_counts': dict(Counter(v['class'] for v in records)),
              'interpretation': 'QA-derived missing-region geometry; not paired real cloudy/clear imagery'}
    json_write(manifest, result)
    ds.close()
    return {k: v for k, v in result.items() if k != 'shapes'}

