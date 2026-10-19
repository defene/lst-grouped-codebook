"""Deterministic input-space corruption; reference values never enter inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from eo_data.core import sha256_file

NOISE_VERSION = 'reference-controlled-corruption-v1'


def stable_seed(*parts):
    payload = json.dumps(parts, separators=(',', ':'), ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')


class MaskBank:
    def __init__(self, root, split, source_index_hash, source_sample_hash):
        directory = Path(root)
        info = json.loads((directory / f'{split}.json').read_text(encoding='utf-8'))
        if info['split'] != split or info['source_index_sha256'] != source_index_hash:
            raise ValueError('Mask bank split/source fingerprint mismatch')
        if info['source_sample_index_sha256'] != source_sample_hash or any(s['split'] != split for s in info['shapes']):
            raise ValueError('Mask bank sample split mismatch')
        path = directory / f'{split}.npz'
        if sha256_file(path) != info['sha256']:
            raise ValueError('Mask bank bytes changed')
        with np.load(path, allow_pickle=False) as arrays:
            self.masks = arrays['masks'].astype(bool)
        if len(self.masks) != len(info['shapes']) or not len(self.masks):
            raise ValueError('Empty or invalid mask bank')
        for mask, record in zip(self.masks, info['shapes']):
            if not mask.any() or hashlib.sha256(np.packbits(mask).tobytes()).hexdigest() != record['shape_id']:
                raise ValueError('Mask bank shape provenance mismatch')
        self.info = info
        self.manifest_sha256 = sha256_file(directory/f'{split}.json')

    def transplant(self, valid, coverage, rng):
        h, w = valid.shape
        wanted = int(round(coverage * valid.sum()))
        hidden = np.zeros_like(valid)
        used = []
        for _ in range(24):
            remaining = wanted - int(hidden.sum())
            if remaining <= max(1, int(valid.sum() * .002)):
                break
            index = int(rng.integers(len(self.masks)))
            shape = self.masks[index]
            rows, cols = np.nonzero(shape)
            shape = shape[rows.min():rows.max()+1, cols.min():cols.max()+1]
            shape = np.rot90(shape, int(rng.integers(4)))
            if rng.random() < .5:
                shape = shape[:, ::-1]
            scale = np.sqrt(remaining / max(1, shape.sum())) * rng.uniform(.6, 1.0)
            nh, nw = max(1, min(h, int(shape.shape[0]*scale))), max(1, min(w, int(shape.shape[1]*scale)))
            shape = shape[np.minimum((np.arange(nh)*shape.shape[0]/nh).astype(int), shape.shape[0]-1)[:, None],
                          np.minimum((np.arange(nw)*shape.shape[1]/nw).astype(int), shape.shape[1]-1)[None, :]]
            top, left = int(rng.integers(h-nh+1)), int(rng.integers(w-nw+1))
            candidate = hidden.copy()
            candidate[top:top+nh, left:left+nw] |= shape
            candidate &= valid
            if abs(int(candidate.sum()) - wanted) < abs(int(hidden.sum()) - wanted):
                hidden = candidate
                used.append(self.info['shapes'][index]['shape_id'])
        return hidden, used


def procedural_mask(valid, coverage, rng):
    h, w = valid.shape
    yy, xx = np.mgrid[:h, :w]
    score = np.zeros((h, w))
    for _ in range(6):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        sy, sx = rng.uniform(.05, .3)*h, rng.uniform(.05, .3)*w
        score += np.exp(-.5*((yy-cy)/sy)**2 - .5*((xx-cx)/sx)**2)
    locations = np.flatnonzero(valid)
    count = int(round(len(locations)*coverage))
    hidden = np.zeros_like(valid)
    if count:
        selected = locations[np.argpartition(score.flat[locations], -count)[-count:]]
        hidden.flat[selected] = True
    return hidden


class Corruptor:
    def __init__(self, config, split, bank=None):
        self.config, self.split, self.bank = config, split, bank
        if config['kind'] == 'bank_missing' and bank is None:
            raise ValueError('Build a split-specific mask bank before using bank_missing')

    def __call__(self, image, reference_mask, sample_id, modality, epoch, view, coverage=None):
        c = self.config
        # Evaluation excludes epoch; seed does not include architecture/loss.
        seed = stable_seed(NOISE_VERSION, c['seed'], self.split, sample_id, modality,
                           c['kind'], epoch if self.split == 'train' else 0, view,
                           coverage if coverage is not None else 'train_schedule')
        rng = np.random.default_rng(seed)
        fraction = float(rng.choice(c['coverages'])) if coverage is None else float(coverage)
        kind = c['kind']
        if coverage is None and self.split == 'train' and rng.random() < c['clean_probability']:
            kind, fraction = 'none', 0.0
        if fraction == 0 or kind == 'none':
            kind, fraction = 'none', 0.0
        valid = reference_mask[0].astype(bool)
        input_mask = reference_mask.copy()
        changed = np.zeros_like(valid)
        x = np.where(reference_mask, image, 0).astype(np.float32).copy()
        shapes = []
        if kind in ('bank_missing', 'procedural_missing'):
            if kind == 'bank_missing':
                changed, shapes = self.bank.transplant(valid, fraction, rng)
            else:
                changed = procedural_mask(valid, fraction, rng)
            input_mask[0] &= ~changed
            x[:, changed] = 0
        elif kind == 'gaussian':
            changed = valid.copy() if c['sigma'] > 0 else changed
            x += (rng.standard_normal(x.shape)*c['sigma']*reference_mask).astype(np.float32)
        elif kind == 'stripes':
            offsets = rng.standard_normal((x.shape[0], 1, x.shape[2]))*c['sigma']
            # The column offset is shared vertically; chosen columns form stripes.
            columns = rng.random(x.shape[2]) < fraction
            changed = valid & columns[None, :]
            x += (offsets*changed[None]).astype(np.float32)
        return {'image': x, 'input_mask': input_mask, 'corruption_mask': changed[None],
                'requested_coverage': fraction, 'actual_coverage': float(changed.sum()/max(1, valid.sum())),
                'noise_kind': kind, 'noise_seed': str(seed), 'shape_ids': ','.join(shapes)}
