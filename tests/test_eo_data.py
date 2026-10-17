import csv
import io
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import zstandard
from eo_data import codec, open_dataset, open_datasets
from eo_data import prepare
from eo_data.core import BANDS, canonical, sample_moments, empty_accumulator, accumulate, finish_stats
from eo_data.verify import verify_all, restore_prithvi


def npy(array):
    f = io.BytesIO()
    np.save(f, array, allow_pickle=False)
    return f.getvalue()


def test_lossless_codec_and_corruption():
    rng = np.random.default_rng(82)
    parts = [npy(rng.integers(-32768, 32768, (6, 256, 256), dtype=np.int16)),
             npy(rng.integers(0, 256, (256, 256), dtype=np.uint8)),
             npy(np.resize(np.array([-32768, 32767, 0, -1], np.int16), (256, 256))),
             b'{ "original formatting": true }\n']
    frame = codec.encode(parts)
    assert codec.decode(frame, codec.original_digest(parts)) == parts
    damaged = bytearray(frame)
    damaged[-1] ^= 1
    with pytest.raises(zstandard.ZstdError):
        codec.decode(damaged)
    with pytest.raises(ValueError, match='hash'):
        codec.decode(frame, '0' * 64)


def test_mask_policy_and_sample_equal_statistics():
    raw = np.zeros((6, 256, 256), np.int16)
    raw[0, 0, :3] = [-9999, -32767, -32768]
    fmask = np.zeros((256, 256), np.uint8)
    fmask[1, :7] = [2, 4, 8, 255, 16, 32, 192]
    image, mask = canonical(raw, 'hls', 'prithvi', fmask=fmask)
    assert not mask[0, 0, :3].any()
    assert not mask[0, 1, :4].any()
    assert mask[0, 1, 4:7].all() and mask[0, 2].all()
    lst = np.zeros((256, 256), np.int16)
    lst[0, :2] = [-32767, -32768]
    assert canonical(lst, 'lst', 'prithvi')[1][0, 0, :2].tolist() == [False, True]
    assert canonical(lst, 'lst', 'lstsr_tb')[1][0, 0, :2].tolist() == [True, False]
    acc = empty_accumulator(1)
    for value, valid_count in [(2, 1), (6, 8)]:
        x = np.full((1, 3, 3), value, np.float32)
        m = np.arange(9).reshape(1, 3, 3) < valid_count
        accumulate(acc, sample_moments(x, m))
    stats = finish_stats(acc, 'lst_hr')
    assert stats['mean'] == [4] and stats['std'] == [2]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def sources(tmp_path):
    protocol = tmp_path / 'experiments/data/protocol.json'
    protocol.parent.mkdir(parents=True)
    protocol.write_text(json.dumps({'training_scope': 'independent_datasets'}))
    for i, (split, value) in enumerate([('train', 1000), ('train', 3000), ('val', 12000)]):
        tile = f'T{i:02d}AAA'
        folder = tmp_path / 'prithvi_data' / split / tile / 'obs_0000/r00_c00'
        folder.mkdir(parents=True)
        bands = np.full((6, 256, 256), value, np.int16)
        bands[:, 0, 0] = -9999
        lst = np.full((256, 256), value, np.int16)
        lst[0, 0] = -32767
        np.save(folder/'bands.npy', bands)
        np.save(folder/'fmask.npy', np.zeros((256, 256), np.uint8))
        np.save(folder/'lst.npy', lst)
        (folder/'metadata.json').write_text(json.dumps(dict(split=split, tile=tile,
            obs_idx=0, row=0, col=0, bands=list(BANDS), bands_scale=10000, lst_scale=100)))
    (tmp_path/'prithvi_data/download_progress.json').write_text('{"kept": true}\n')
    source = tmp_path/'data_inspection/extracted_metadata/lstsr_tb_final_dataset_20260719'
    hls_dir = source/'hls_l8_conditions_2019_2025_lc08'
    hrows, lrows = [], []
    for i, (split, value) in enumerate([('train', 2000), ('train', 6000), ('val', 15000), ('test', 20000)]):
        tile, key = f'T{i:02d}AAA', f'sequence_{i}'
        rel = f'arrays/{key}.npz'
        (source/'arrays').mkdir(parents=True, exist_ok=True)
        (hls_dir/'arrays').mkdir(parents=True, exist_ok=True)
        b = np.full((6, 256, 256), value, np.int16)
        v = np.ones((256, 256), np.uint8)
        v[0, 0] = 0
        np.savez_compressed(hls_dir/rel, hls_reflectance_i16=b, hls_valid=v,
                            hls_fmask=np.zeros((256, 256), np.uint8))
        l = np.full((12, 256, 256), 30000, np.int16)
        l[0] = value
        l[2] = value + 1000
        lv = np.ones_like(l, dtype=np.uint8)
        lv[:, 0, 0] = 0
        np.savez_compressed(source/rel, hr_lst=l, hr_valid=lv)
        hrows.append(dict(sequence_id=key, archive=rel, split=split, mgrs_tile=tile,
            hls_datetime_utc='2020-01-01', hls_image_id=key, reflectance_bands='B2,B3,B4,B5,B6,B7', reflectance_scale='0.0001'))
        lrows.append(dict(sequence_id=key, archive=rel, split=split, mgrs_tile=tile,
            nodata_i16=-32768, patch_size=256, available_months='1,3', available_month_count=2,
            month_dates_json=json.dumps({'01': '2020-01-01', '03': '2020-03-01'}), qa_version='fixture'))
    write_csv(source/'phase4_dataset_manifest.csv', lrows)
    write_csv(hls_dir/'hls_condition_manifest.csv', hrows)
    return tmp_path


def test_both_sources_train_only_and_read_interface(sources):
    output = sources/'processed_data/v1'
    for name in ('prithvi', 'lstsr_tb'):
        prepare.prepare_dataset(sources, name, output, workers=2)
    datasets = open_datasets(root=sources)
    assert len(datasets['prithvi']['hls']) == 2
    assert len(datasets['lstsr_tb']['hls']) == 2
    assert len(datasets['lstsr_tb']['lst']) == 4
    assert datasets['prithvi']['lst'].stats['mean'] == [2000]
    assert datasets['lstsr_tb']['lst'].stats['mean'] == [4500]
    np.testing.assert_allclose(datasets['prithvi']['hls'].stats['mean'], .2, atol=1e-7)
    np.testing.assert_allclose(datasets['lstsr_tb']['hls'].stats['mean'], .4, atol=1e-7)
    for group in datasets.values():
        for ds in group.values():
            item = ds[0]
            assert not np.shares_memory(item['image'], item['target'])
            assert np.isfinite(item['image']).all()
            assert not item['corruption_mask'].any()
            other = pickle.loads(pickle.dumps(ds))
            np.testing.assert_array_equal(other[0]['image'], item['image'])
            np.testing.assert_array_equal(ds.epoch_indices(2), other.epoch_indices(2))
            other.close()
            ds.close()
    assert verify_all(sources, samples=3)['lstsr_tb']['status'] == 'passed'
    restored = sources/'restored'
    assert restore_prithvi(sources, None, restored)['full_dataset']
    for file in (sources/'prithvi_data').rglob('*'):
        if file.is_file():
            assert file.read_bytes() == (restored/file.relative_to(sources/'prithvi_data')).read_bytes()
    ds = open_dataset('lstsr', 'lst', root=sources)
    assert [ds[i]['month'] for i in range(len(ds))] == [1, 3, 1, 3]
    path = ds.source/'arrays/sequence_0.npz'
    with path.open('ab') as f:
        f.write(b'drift')
    with pytest.raises(ValueError, match='changed'):
        ds[0]
    ds.close()


def test_transaction_resume_and_development_guard(sources, monkeypatch):
    monkeypatch.setattr(prepare, 'CHUNK', 2)
    original = prepare.inspect_record
    def fail(record, *args):
        if record['id'] == 3:
            raise RuntimeError('simulated interruption')
        return original(record, *args)
    monkeypatch.setattr(prepare, 'inspect_record', fail)
    output = sources/'resumable'
    with pytest.raises(RuntimeError, match='simulated'):
        prepare.prepare_dataset(sources, 'prithvi', output, workers=1)
    progress = json.loads((output/'prithvi/progress.json').read_text())
    assert progress['completed_records'] == 2
    monkeypatch.setattr(prepare, 'inspect_record', original)
    bundle = prepare.prepare_dataset(sources, 'prithvi', output, workers=2)
    assert bundle['records'] == 3 and sum(s['records'] for s in bundle['shards']) == 3
    assert verify_all(sources, output, 'prithvi')['prithvi']['status'] == 'passed'
    dev = sources/'development'
    prepare.prepare_dataset(sources, 'prithvi', dev, limit=1)
    with pytest.raises(ValueError, match='Development'):
        open_dataset('prithvi', 'hls', root=sources, prepared=dev)
    with pytest.raises(ValueError, match='scope'):
        prepare.prepare_dataset(sources, 'prithvi', dev)


def test_all_invalid_excluded_per_modality_and_month(sources):
    p = sources/'prithvi_data/train/T00AAA/obs_0000/r00_c00/fmask.npy'
    np.save(p, np.full((256, 256), 255, np.uint8))
    lst_path = sources/'data_inspection/extracted_metadata/lstsr_tb_final_dataset_20260719/arrays/sequence_1.npz'
    with np.load(lst_path) as z:
        lst, valid = z['hr_lst'], z['hr_valid']
    valid[2] = 0
    np.savez_compressed(lst_path, hr_lst=lst, hr_valid=valid)
    output = sources/'processed_data/v1'
    for name in ('prithvi', 'lstsr_tb'):
        prepare.prepare_dataset(sources, name, output, workers=2)
    datasets = open_datasets(root=sources)
    assert len(datasets['prithvi']['hls']) == 1
    assert len(datasets['prithvi']['lst']) == 2
    assert len(datasets['lstsr_tb']['hls']) == 2
    assert len(datasets['lstsr_tb']['lst']) == 3
    np.testing.assert_allclose(datasets['lstsr_tb']['lst'].stats['mean'], [11000/3])
    for name, expected in [('prithvi', 'hls'), ('lstsr_tb', 'lst_hr')]:
        with sqlite3.connect(output/name/'index.sqlite') as db:
            assert db.execute('SELECT modality,reason FROM rejected').fetchall() == [(expected, 'no_valid_reference_pixels')]
    for group in datasets.values():
        for ds in group.values(): ds.close()


def test_archive_direct_reads_restore_and_relocation(sources):
    from eo_data.archive import create_archive, attach_archive
    output = sources/'processed_data/v1'
    prepare.prepare_dataset(sources, 'prithvi', output, workers=2)
    ds = open_dataset('prithvi','hls',root=sources)
    original = ds[0]['image'].copy()
    ds.close()
    archive = sources/'latest.zip'
    report = create_archive(sources, archive)
    attach_archive(sources, archive, verified_report=report)
    for p in (output/'prithvi/packs').glob('*.eoz'):
        assert p.resolve().is_relative_to(sources.resolve())
        p.unlink()
    ds = open_dataset('prithvi','hls',root=sources)
    np.testing.assert_array_equal(ds[0]['image'], original)
    assert ds.fingerprint['record_index'] == ds.bundle['artifacts']['index.sqlite']
    child = pickle.loads(pickle.dumps(ds))
    np.testing.assert_array_equal(child[0]['image'], original)
    child.close(); ds.close()
    assert verify_all(sources,dataset='prithvi')['prithvi']['status'] == 'passed'
    restored = sources/'restored_from_archive'
    assert restore_prithvi(sources,None,restored)['full_dataset']
    for p in (sources/'prithvi_data').rglob('*'):
        if p.is_file(): assert p.read_bytes() == (restored/p.relative_to(sources/'prithvi_data')).read_bytes()
    relocated = sources/'new_project'
    attach_archive(relocated, archive, verified_report=report)
    ds = open_dataset('prithvi','hls',root=relocated)
    np.testing.assert_array_equal(ds[0]['image'], original)
    ds.close()


def test_writer_lock_and_processing_code_guard(sources):
    from eo_data.locking import build_lock
    output = sources/'processed_data/v1'
    with build_lock(output/'prithvi'):
        with pytest.raises(RuntimeError,match='writer'):
            prepare.prepare_dataset(sources,'prithvi',output)
    prepare.prepare_dataset(sources,'prithvi',output)
    path = output/'prithvi/bundle.json'
    bundle = json.loads(path.read_text())
    bundle['identity']['pipeline_sha256']['core.py'] = '0'*64
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='Processing code'):
        open_dataset('prithvi','hls',root=sources)
