"""Derive a tile-disjoint HLP train/val/test version without rewriting source data."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import copy
import csv
import hashlib
import io
import json
import shutil
import sqlite3
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eo_data import open_dataset
from eo_data.archive import ArchiveFrames
from eo_data import codec
from eo_data.core import canonical, sample_moments, empty_accumulator, accumulate, finish_stats, sha256_file

SEED = 20260914
FRACTION = 0.05
SOURCE = ROOT / 'processed_data/v1/prithvi'
DEST = ROOT / 'processed_data/hlp_split_v2_20260914/prithvi'
EVIDENCE = ROOT / 'data_inspection/hlp_split_20260914'
MODS = ('hls', 'lst_hr')

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def init_worker(storage):
    global archive
    archive = ArchiveFrames(ROOT, storage)

def moments_chunk(rows):
    accumulators = {m: empty_accumulator(6 if m == 'hls' else 1) for m in MODS}
    for rid, loc, digest, hls_count, lst_count in rows:
        parts = codec.decode(archive.read(json.loads(loc)), digest)
        fmask = np.load(io.BytesIO(parts[1]), allow_pickle=False)
        for mod, part, count in [('hls', 0, hls_count), ('lst_hr', 2, lst_count)]:
            raw = np.load(io.BytesIO(parts[part]), allow_pickle=False)
            image, mask = canonical(raw, mod, 'prithvi', fmask=fmask if mod == 'hls' else None)
            assert int(mask.sum()) == count, (rid, mod)
            accumulate(accumulators[mod], sample_moments(image, mask))
    return accumulators

def merge_accumulators(target, part):
    for mod in MODS:
        for field in ('samples', 'pixels'):
            target[mod][field] += part[mod][field]
        for field in ('sum_mean', 'sum_second'):
            target[mod][field] = (np.asarray(target[mod][field]) + part[mod][field]).tolist()

def main():
    started = time.time()
    EVIDENCE.mkdir(exist_ok=True)
    if DEST.exists():
        raise RuntimeError(f'Completed or partial destination already exists: {DEST}')
    db = sqlite3.connect((SOURCE/'index.sqlite').as_uri()+'?mode=ro&immutable=1', uri=True)
    original_bundle = json.loads((SOURCE/'bundle.json').read_text())
    protected = {p.name: sha256_file(p) for p in SOURCE.iterdir() if p.is_file() and p.suffix not in ('.wal', '.shm') and not p.name.endswith(('-wal','-shm'))}
    source_rows = db.execute('SELECT id,key,split,tile FROM records ORDER BY id').fetchall()
    assert len(source_rows) == 242416
    original_tiles = {split: {r[3] for r in source_rows if r[2] == split} for split in ('train','val')}
    assert len(original_tiles['train']) == 2943 and len(original_tiles['val']) == 151
    assert not original_tiles['train'] & original_tiles['val']
    ranked = sorted(original_tiles['train'], key=lambda t: (hashlib.sha256(f'{SEED}:{t}'.encode()).hexdigest(), t))
    val_tiles = set(ranked[:round(FRACTION*len(ranked))])
    assignments = {rid: ('test' if split == 'val' else 'val' if tile in val_tiles else 'train') for rid,key,split,tile in source_rows}
    split_ids = {s: {rid for rid,split in assignments.items() if split==s} for s in ('train','val','test')}
    split_tiles = {s: {tile for rid,key,old,tile in source_rows if assignments[rid]==s} for s in split_ids}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not split_ids[a]&split_ids[b]
        assert not split_tiles[a]&split_tiles[b]
    assert split_ids['test'] == {r[0] for r in source_rows if r[2]=='val'}
    assert split_ids['train'] | split_ids['val'] == {r[0] for r in source_rows if r[2]=='train'}
    counts = {s: {'records': len(split_ids[s]), 'tiles': len(split_tiles[s]), 'hls_samples': len(split_ids[s]), 'fine_lst_samples': len(split_ids[s])} for s in split_ids}
    print('Selected splits:', json.dumps(counts), flush=True)

    progress = json.loads(db.execute("SELECT value FROM settings WHERE key='progress'").fetchone()[0])
    old_stats = json.loads((SOURCE/'normalization.json').read_text())
    for mod in MODS:
        assert finish_stats(progress['train_accumulators'][mod], mod) == old_stats[mod]
    sample_counts = {(rid,mod):count for rid,mod,count in db.execute('SELECT record_id,modality,valid_pixels FROM samples')}
    validation_rows = [(rid,loc,digest,sample_counts[rid,'hls'],sample_counts[rid,'lst_hr'])
                       for rid,loc,digest in db.execute('SELECT id,loc,digest FROM records ORDER BY id') if rid in split_ids['val']]
    chunks = [validation_rows[i:i+256] for i in range(0,len(validation_rows),256)]
    cache = EVIDENCE/'moments'; cache.mkdir(exist_ok=True)
    storage = json.loads((SOURCE/'storage.json').read_text())
    results = {}
    with ProcessPoolExecutor(max_workers=6, initializer=init_worker, initargs=(storage,)) as pool:
        jobs = {}
        for i, chunk in enumerate(chunks):
            fingerprint = hashlib.sha256(json.dumps(chunk).encode()).hexdigest()
            path = cache/f'{i:04d}.json'
            if path.exists():
                part = json.loads(path.read_text()); assert part['source_sha256'] == fingerprint
                results[i] = part['accumulators']
            else:
                jobs[pool.submit(moments_chunk, chunk)] = (i, fingerprint, path)
        for job in as_completed(jobs):
            i,fingerprint,path = jobs[job]
            results[i] = job.result()
            write_json(path, {'source_sha256':fingerprint, 'records':len(chunks[i]), 'accumulators':results[i]})
            print(f'Validated held-out raw records: {len(results)}/{len(chunks)} batches', flush=True)
    heldout = {m:empty_accumulator(6 if m=='hls' else 1) for m in MODS}
    for i in range(len(chunks)):
        merge_accumulators(heldout, results[i])
    train_acc = copy.deepcopy(progress['train_accumulators'])
    for mod in MODS:
        for field in ('samples','pixels'):
            train_acc[mod][field] -= heldout[mod][field]
        for field in ('sum_mean','sum_second'):
            train_acc[mod][field] = (np.asarray(train_acc[mod][field])-heldout[mod][field]).tolist()
        assert train_acc[mod]['samples'] == counts['train']['records']
        expected_pixels = sum(sample_counts[rid,mod] for rid in split_ids['train'])
        assert train_acc[mod]['pixels'] == expected_pixels
    normalization = {m:finish_stats(train_acc[m],m) for m in MODS}
    write_json(EVIDENCE/'normalization_derivation.json', {
        'method':'Original train sufficient-statistic sums minus raw recomputed held-out sums; float64 arithmetic.',
        'validation_records_recomputed_and_digest_verified':len(validation_rows),
        'original_train':progress['train_accumulators'], 'heldout_validation':heldout,
        'remaining_train':train_acc,
        'test_contribution':0,
        'source_normalization_sha256':protected['normalization.json'],
        'note':'Same canonical masks and equal-sample weighting; no rounded published means used. Floating-point subtraction has normal roundoff.'})

    staging = DEST.with_name('prithvi.building')
    if staging.exists():
        raise RuntimeError(f'Inspect previous staging directory before rerunning: {staging}')
    (staging/'indices').mkdir(parents=True)
    target_db = sqlite3.connect(staging/'index.sqlite')
    db.backup(target_db)
    target_db.execute('CREATE TEMP TABLE new_split(id INTEGER PRIMARY KEY, split TEXT)')
    target_db.executemany('INSERT INTO new_split VALUES (?,?)', assignments.items())
    target_db.execute('UPDATE records SET split=(SELECT split FROM new_split WHERE id=records.id)')
    target_db.execute('UPDATE samples SET split=(SELECT split FROM new_split WHERE id=samples.record_id)')
    for rid,meta in target_db.execute('SELECT id,meta FROM records').fetchall():
        obj=json.loads(meta); obj['source_split']=obj['split']; obj['experiment_split']=assignments[rid]
        target_db.execute('UPDATE records SET meta=? WHERE id=?', (json.dumps(obj,separators=(',',':')),rid))
    samples = {}
    for split in split_ids:
        for mod in MODS:
            rows = target_db.execute('SELECT record_id,slot,valid_pixels FROM samples WHERE split=? AND modality=? ORDER BY record_id,slot',(split,mod)).fetchall()
            array = np.asarray(rows,dtype=np.int64)
            assert set(array[:,0]) == split_ids[split]
            file = staging/'indices'/f'{split}.{mod}.npy'; np.save(file,array,allow_pickle=False)
            samples[f'{split}/{mod}']={'samples':len(rows),'index':file.relative_to(staging).as_posix(),'sha256':sha256_file(file)}
    for mod in MODS:
        assert np.array_equal(np.load(SOURCE/f'indices/val.{mod}.npy'),np.load(staging/f'indices/test.{mod}.npy'))
    with (SOURCE/'source_index.jsonl').open(encoding='utf-8') as src, (staging/'source_index.jsonl').open('w',encoding='utf-8') as dst:
        for line in src:
            obj=json.loads(line); obj['source_split']=obj['split'];obj['split']=assignments[obj['id']]
            dst.write(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n')
    identity = copy.deepcopy(original_bundle['identity'])
    identity['source_index_sha256']=sha256_file(staging/'source_index.jsonl')
    identity['split_version']='hlp-tile-validation-v2-20260914'
    info = json.loads((SOURCE/'source_info.json').read_text())
    info['record_counts']={s:v['records'] for s,v in counts.items()}
    info['candidate_samples']={key:value['samples'] for key,value in samples.items()}
    info['index_sha256']=identity['source_index_sha256']
    info['source_split_semantics']={'train':'original training pool','val':'fixed test'}
    write_json(staging/'source_info.json',info)
    write_json(staging/'normalization.json',normalization)
    shutil.copyfile(SOURCE/'protocol.json',staging/'protocol.json')
    shutil.copyfile(SOURCE/'storage.json',staging/'storage.json')
    progress['train_accumulators']=train_acc
    progress['split_version']=identity['split_version']
    target_db.execute("UPDATE settings SET value=? WHERE key='progress'",(json.dumps(progress),))
    target_db.execute("UPDATE settings SET value=? WHERE key='identity'",(json.dumps(identity),))
    target_db.commit()
    assert target_db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert not target_db.execute('SELECT tile FROM records GROUP BY tile HAVING COUNT(DISTINCT split)>1').fetchall()
    target_db.close(); db.close()
    manifest_files={}
    for split in split_ids:
        file=EVIDENCE/f'{split}_records.csv'
        with file.open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['record_id','record_key','mgrs_tile','source_split','experiment_split'])
            writer.writerows((rid,key,tile,old,split) for rid,key,old,tile in source_rows if assignments[rid]==split)
        tile_file=EVIDENCE/f'{split}_tiles.txt';tile_file.write_text('\n'.join(sorted(split_tiles[split]))+'\n',encoding='utf-8')
        manifest_files[file.name]=sha256_file(file);manifest_files[tile_file.name]=sha256_file(tile_file)
    bundle=copy.deepcopy(original_bundle)
    bundle['samples']=samples;bundle['identity']=identity
    bundle['artifacts']={name:sha256_file(staging/name) for name in original_bundle['artifacts']}
    bundle['derived_from']={'prepared':'processed_data/v1/prithvi','bundle_sha256':protected['bundle.json'],'split_version':identity['split_version']}
    write_json(staging/'bundle.json',bundle)
    write_json(staging/'progress.json',{'status':'complete','dataset':'prithvi','records':len(source_rows),'samples':{k:v['samples'] for k,v in samples.items()},'split_version':identity['split_version']})
    verification=[]
    # Reader expects a child named prithvi; publish locally before read checks.
    staging.rename(DEST)
    try:
        for split in split_ids:
            for mod in MODS:
                ds=open_dataset('prithvi',mod,split,prepared=DEST.parent)
                assert len(ds)==counts[split]['records']
                for i in (0,len(ds)//2,len(ds)-1):
                    sample=ds[i]
                    assert sample['tile_id'] in split_tiles[split]
                    assert sample['split']==split
                    assert np.isfinite(sample['image']).all()
                    assert ds.get_metadata(i)['experiment_split']==split
                verification.append({'split':split,'modality':mod,'samples':len(ds),'raw_read_checks':3})
                ds.close()
        for name,digest in protected.items():
            assert sha256_file(SOURCE/name)==digest, name
    except Exception:
        DEST.rename(staging)
        raise
    report={'status':'complete','version':identity['split_version'],'date':'2026-09-14','seed':SEED,
            'method':'Sort original-train MGRS tiles by (SHA256(seed:tile), tile); first round(0.05*N) become val.',
            'target_train_tile_fraction':FRACTION,'eligible_train_tiles':len(ranked),'selected_val_tiles':len(val_tiles),
            'counts':counts,'test_membership_and_index_arrays_unchanged':True,'old_train_equals_new_train_union_val':True,
            'pairwise_record_overlap':0,'pairwise_tile_id_overlap':0,'spatial_footprint_overlap_checked':False,
            'prepared_directory':DEST.parent.relative_to(ROOT).as_posix(),'reader_splits':{'train':'train','validation':'val','test':'test'},
            'normalization':'Remaining-train sufficient statistics; see normalization_derivation.json.',
            'old_checkpoints_independent_of_new_validation':False,
            'source_files_unchanged':protected,'membership_sha256':manifest_files,'read_verification':verification,
            'script_sha256':sha256_file(Path(__file__)),'elapsed_seconds':round(time.time()-started,2)}
    write_json(EVIDENCE/'split_summary.json',report)
    write_json(DEST/'verification.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    main()
