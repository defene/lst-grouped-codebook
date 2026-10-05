"""Resumable full census, train-only fitting, and verified lossless packing."""
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import time
import numpy as np
from . import codec
from .core import (VERSION, BANDS, accumulate, canonical, empty_accumulator,
                   finish_stats, json_write, sample_moments, sha256_file)

PROJECT = Path(__file__).resolve().parents[1]
CHUNK = 512

def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path escapes source root: {relative}')
    return path

def bounded_map(pool, fn, items, width):
    it = iter(items)
    pending = deque()
    for _ in range(width):
        try: pending.append(pool.submit(fn, next(it)))
        except StopIteration: break
    while pending:
        yield pending.popleft().result()
        try: pending.append(pool.submit(fn, next(it)))
        except StopIteration: pass

def file_signature(path):
    s = path.stat()
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns}

def create_index(project, name, out, limit=None):
    index = out / 'source_index.jsonl'
    if index.exists() and (out/'source_info.json').exists():
        info = json.loads((out/'source_info.json').read_text(encoding='utf-8'))
        if info['limit'] != limit:
            raise ValueError('Cannot change a partial/full build scope in an existing output')
        for rel, digest in info['source_manifest_hashes'].items():
            if sha256_file(project/rel) != digest:
                raise ValueError(f'Source manifest changed: {rel}')
        if sha256_file(index) != info['index_sha256']:
            raise ValueError('Source index changed')
        return info
    temp = index.with_suffix('.building')
    hashes = {}
    counts = Counter()
    candidate = Counter()
    count = 0
    if name == 'prithvi':
        source = project/'prithvi_data'
        if shutil.which('rg'):
            paths = subprocess.check_output(['rg','--files','--hidden','--no-ignore',str(source)], text=True).splitlines()
            rels = [Path(p).relative_to(source).as_posix() for p in paths]
        else:
            rels = [p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()]
        groups = {}
        extra = []
        for rel in rels:
            if rel.startswith(('train/','val/')):
                parent, base = rel.rsplit('/', 1)
                groups.setdefault(parent, set()).add(base)
            else:
                extra.append(rel)
        with temp.open('w', encoding='utf-8', newline='\n') as f:
            for key, members in sorted(groups.items()):
                if set(codec.FILENAMES) != members:
                    raise ValueError(f'Unexpected/missing files at {key}: {members}')
                split, tile, obs, patch = key.split('/')
                int(obs.removeprefix('obs_'))
                row, col = patch.split('_')
                int(row[1:]); int(col[1:])
                record = {'id': count+1, 'key': key, 'split': split, 'tile': tile,
                          'loc': {'path': key}}
                f.write(json.dumps(record, separators=(',',':'))+'\n')
                count += 1; counts[split] += 1
                candidate[f'{split}/hls'] += 1; candidate[f'{split}/lst_hr'] += 1
                if limit and count >= limit: break
        aux = out/'original_aux'
        aux.mkdir(exist_ok=True)
        auxiliary = {}
        for rel in extra:
            src = safe_path(source, rel)
            dst = safe_path(aux, rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            auxiliary[rel] = {'sha256': sha256_file(dst), **file_signature(src)}
        root_rel = 'prithvi_data'
    else:
        root_rel = 'data_inspection/extracted_metadata/lstsr_tb_final_dataset_20260719'
        source = project/root_rel
        hls_rel = 'hls_l8_conditions_2019_2025_lc08'
        manifests = [source/'phase4_dataset_manifest.csv', source/hls_rel/'hls_condition_manifest.csv']
        for p in manifests: hashes[p.relative_to(project).as_posix()] = sha256_file(p)
        hls = {}
        with manifests[1].open(encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                key = r['sequence_id']
                if key in hls: raise ValueError(f'Duplicate HLS key: {key}')
                if r['reflectance_bands'] != 'B2,B3,B4,B5,B6,B7' or float(r['reflectance_scale']) != 0.0001:
                    raise ValueError(f'Unexpected HLS band/scale at {key}')
                hls[key] = {k:r[k] for k in ['archive','split','mgrs_tile','hls_datetime_utc','hls_image_id']}
        seen = set()
        with manifests[0].open(encoding='utf-8', newline='') as src, temp.open('w', encoding='utf-8', newline='\n') as dst:
            for r in csv.DictReader(src):
                key = r['sequence_id']
                if key in seen: raise ValueError(f'Duplicate LST key: {key}')
                seen.add(key)
                h = hls[key]
                if (r['split'],r['mgrs_tile']) != (h['split'],h['mgrs_tile']):
                    raise ValueError(f'LST/HLS membership mismatch at {key}')
                if int(r['nodata_i16']) != -32768 or int(r['patch_size']) != 256:
                    raise ValueError(f'Unexpected source encoding at {key}')
                months = [int(m)-1 for m in r['available_months'].split(',')]
                if len(set(months)) != len(months) or any(m not in range(12) for m in months) or len(months)!=int(r['available_month_count']):
                    raise ValueError(f'Invalid availability at {key}')
                loc = {'lst': r['archive'], 'hls': hls_rel+'/'+h['archive'], 'slots': sorted(months)}
                safe_path(source, loc['lst']); safe_path(source, loc['hls'])
                record = {'id':count+1,'key':key,'split':r['split'],'tile':r['mgrs_tile'],'loc':loc,
                          'meta':{'sequence_id':key,'month_dates':json.loads(r['month_dates_json']),
                                  'hls_datetime':h['hls_datetime_utc'],'hls_image_id':h['hls_image_id'],
                                  'qa_version':r['qa_version'],'source_split':r['split']}}
                dst.write(json.dumps(record, separators=(',',':'))+'\n')
                count += 1; counts[r['split']] += 1
                candidate[f"{r['split']}/hls"] += 1
                candidate[f"{r['split']}/lst_hr"] += len(months)
                if limit and count >= limit: break
        if not limit and seen != set(hls): raise ValueError('LST and HLS manifest key sets differ')
        auxiliary = {}
    temp.replace(index)
    info = {'dataset':name,'source_root':root_rel,'records':count,'record_counts':dict(counts),
            'candidate_samples':dict(candidate),'source_manifest_hashes':hashes,
            'index_sha256':sha256_file(index),'auxiliary':auxiliary,'limit':limit}
    json_write(out/'source_info.json', info)
    return info

def inspect_record(record, source, name, level):
    """Each worker loads one complete source record; no validation fitting."""
    fit = record['split'] == 'train'
    result = {'record':record,'samples':[], 'frame':None, 'raw_bytes':0}
    if name == 'prithvi':
        folder = safe_path(source, record['loc']['path'])
        paths = [folder/n for n in codec.FILENAMES]
        signatures = [file_signature(p) for p in paths]
        parts = [p.read_bytes() for p in paths]
        if [file_signature(p) for p in paths] != signatures:
            raise ValueError(f'Source changed during read: {folder}')
        meta = json.loads(parts[3])
        b, f, l = [np.load(io.BytesIO(p), allow_pickle=False) for p in parts[:3]]
        split,tile,obs,patch = record['key'].split('/')
        row,col = patch.split('_')
        if (meta['split'],meta['tile'],meta['obs_idx'],meta['row'],meta['col']) != (split,tile,int(obs[4:]),int(row[1:]),int(col[1:])):
            raise ValueError(f'Metadata/path mismatch: {record["key"]}')
        if tuple(meta['bands']) != BANDS or meta['bands_scale'] != 10000 or meta['lst_scale'] != 100:
            raise ValueError(f'Unexpected scale/bands: {record["key"]}')
        result.update(meta=meta,digest=codec.original_digest(parts),frame=codec.encode(parts,level),
                      signatures=signatures,raw_bytes=sum(map(len,parts)))
        arrays = [('hls',-1,*canonical(b,'hls',name,fmask=f)),
                  ('lst_hr',-1,*canonical(l,'lst_hr',name))]
    else:
        loc = record['loc']
        hp, lp = safe_path(source,loc['hls']), safe_path(source,loc['lst'])
        signatures = {'hls':file_signature(hp),'lst':file_signature(lp)}
        with np.load(hp,allow_pickle=False) as z:
            b, f, v = z['hls_reflectance_i16'], z['hls_fmask'], z['hls_valid']
        with np.load(lp,allow_pickle=False) as z:
            l, lv = z['hr_lst'], z['hr_valid']
        if l.shape != (12,256,256) or lv.shape != l.shape or l.dtype != np.int16:
            raise ValueError(f'Invalid month bundle: {record["key"]}')
        if {'hls':file_signature(hp),'lst':file_signature(lp)} != signatures:
            raise ValueError(f'Source changed during read: {record["key"]}')
        digest = hashlib.sha256()
        for array in (b,f,v,l,lv): digest.update(array.tobytes(order='C'))
        result.update(meta=record['meta'],digest=digest.hexdigest(),signatures=signatures,
                      raw_bytes=signatures['hls']['size']+signatures['lst']['size'])
        arrays = [('hls',-1,*canonical(b,'hls',name,fmask=f,source_valid=v))]
        for slot in loc['slots']:
            arrays.append(('lst_hr',slot,*canonical(l[slot],'lst_hr',name,source_valid=lv[slot])))
    for modality,slot,image,mask in arrays:
        n = int(mask.sum())
        result['samples'].append((modality,slot,n,sample_moments(image,mask) if fit and n else None))
    return result

def init_db(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS records(id INTEGER PRIMARY KEY, key TEXT UNIQUE, split TEXT, tile TEXT, meta TEXT, loc TEXT, digest TEXT);
    CREATE TABLE IF NOT EXISTS samples(modality TEXT, split TEXT, record_id INTEGER, slot INTEGER, valid_pixels INTEGER, UNIQUE(modality,record_id,slot));
    CREATE TABLE IF NOT EXISTS rejected(record_id INTEGER, modality TEXT, slot INTEGER, reason TEXT);
    CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE IF NOT EXISTS shards(name TEXT PRIMARY KEY, bytes INTEGER, sha256 TEXT, records INTEGER, verified INTEGER);
    ''')
    return db

def _read_setting(db, key):
    row = db.execute('SELECT value FROM settings WHERE key=?',(key,)).fetchone()
    return json.loads(row[0]) if row else None

def prepare_dataset(project, name, output, workers=4, level=3, limit=None):
    from .locking import build_lock
    with build_lock(Path(output)/name):
        return _prepare_dataset(project,name,output,workers,level,limit)

def _prepare_dataset(project, name, output, workers=4, level=3, limit=None):
    started = time.time()
    out = output/name
    out.mkdir(parents=True, exist_ok=True)
    if (out/'bundle.json').exists():
        bundle = json.loads((out/'bundle.json').read_text(encoding='utf-8'))
        if bundle['identity']['limit'] != limit or bundle['identity']['zstd_level'] != level:
            raise ValueError('Completed output has a different build scope/configuration')
        print(f'{name}: already complete; keeping frozen output',flush=True)
        return bundle
    info = create_index(project,name,out,limit)
    source = project/info['source_root']
    db = init_db(out/'index.sqlite')
    protocol = json.loads((project/'experiments/data/protocol.json').read_text(encoding='utf-8'))
    if protocol['training_scope'] != 'independent_datasets': raise ValueError('This pipeline requires independent datasets')
    identity = {'protocol_sha256':sha256_file(project/'experiments/data/protocol.json'),
                'pipeline_sha256':{p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
                'source_index_sha256':info['index_sha256'],'zstd_level':level,'limit':limit}
    saved = _read_setting(db,'identity')
    if saved and saved != identity: raise ValueError('Build identity changed; use a new output version')
    if not saved:
        db.execute('INSERT INTO settings VALUES (?,?)',('identity',json.dumps(identity)))
        db.commit()
        json_write(out/'protocol.json',protocol)
    state = _read_setting(db,'progress') or {'completed_records':0,'raw_bytes':0,'packed_bytes':0,
        'train_accumulators':{'hls':empty_accumulator(6),'lst_hr':empty_accumulator(1)},'elapsed_seconds':0.0}
    prior_time = state['elapsed_seconds']
    def records():
        with (out/'source_index.jsonl').open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r['id'] > state_start: yield r
    state_start = state['completed_records']
    it = iter(records())
    packs = out/'packs'
    if name == 'prithvi': packs.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            chunk=[]
            for _ in range(CHUNK):
                try: chunk.append(next(it))
                except StopIteration: break
            if not chunk: break
            part = f'part_{(chunk[0]["id"]-1)//CHUNK:06d}.eoz'
            tmp = packs/(part+'.partial')
            stream = tmp.open('wb') if name=='prithvi' else None
            locations=[]
            pending_state=json.loads(json.dumps(state))
            try:
                results = bounded_map(pool,lambda r:inspect_record(r,source,name,level),chunk,max(2,workers*2))
                for result in results:
                    r=result['record']; loc=dict(r['loc'])
                    loc['source_signatures']=result['signatures']
                    if stream:
                        offset=stream.tell(); frame=result['frame']; stream.write(frame)
                        loc.update(shard=part,offset=offset,length=len(frame))
                        locations.append((offset,len(frame),result['digest']))
                        pending_state['packed_bytes']+=len(frame)
                    for mod,slot,n,moments in result['samples']:
                        if n:
                            db.execute('INSERT INTO samples VALUES (?,?,?,?,?)',(mod,r['split'],r['id'],slot,n))
                            if r['split']=='train': accumulate(pending_state['train_accumulators'][mod],moments)
                        else:
                            db.execute('INSERT INTO rejected VALUES (?,?,?,?)',(r['id'],mod,slot,'no_valid_reference_pixels'))
                    db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?)',
                        (r['id'],r['key'],r['split'],r['tile'],json.dumps(result['meta'],separators=(',',':')),
                         json.dumps(loc,separators=(',',':')),result['digest']))
                    pending_state['completed_records']+=1
                    pending_state['raw_bytes']+=result['raw_bytes']
                if stream:
                    stream.flush();os.fsync(stream.fileno());stream.close();stream=None
                    with tmp.open('rb') as check:
                        for offset,length,digest in locations:
                            check.seek(offset)
                            codec.decode(check.read(length),digest)
                    target = packs/part
                    tmp.replace(target)
                    db.execute('INSERT OR REPLACE INTO shards VALUES (?,?,?,?,?)',
                               (part,target.stat().st_size,sha256_file(target),len(locations),1))
                pending_state['elapsed_seconds']=prior_time+time.time()-started
                db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)',('progress',json.dumps(pending_state)))
                db.commit()
            except BaseException:
                if stream:stream.close()
                db.rollback()
                raise
            state=pending_state
            progress={'dataset':name,'status':'running','phase':'full_preprocessing',
                      'completed_records':state['completed_records'],'total_records':info['records'],
                      'raw_GiB':state['raw_bytes']/2**30,'packed_GiB':state['packed_bytes']/2**30,
                      'elapsed_seconds':state['elapsed_seconds']}
            json_write(out/'progress.json',progress)
            if state['completed_records']%(CHUNK*8)==0 or state['completed_records']==info['records']:
                print(json.dumps(progress),flush=True)
    if state['completed_records'] != info['records']:raise RuntimeError('Incomplete source census')
    overlap=db.execute('SELECT tile FROM records GROUP BY tile HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchone()
    if overlap:raise ValueError(f'Tile crosses splits within {name}: {overlap[0]}')
    normalization={mod:finish_stats(acc,mod) for mod,acc in state['train_accumulators'].items()}
    json_write(out/'normalization.json',normalization)
    splits={}
    (out/'indices').mkdir(exist_ok=True)
    for mod,split,count in db.execute('SELECT modality,split,COUNT(*) FROM samples GROUP BY modality,split').fetchall():
        path=out/'indices'/f'{split}.{mod}.npy'
        arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.int64,shape=(count,3))
        cur=db.execute('SELECT record_id,slot,valid_pixels FROM samples WHERE modality=? AND split=? ORDER BY record_id,slot',(mod,split))
        pos=0
        while True:
            batch=cur.fetchmany(10000)
            if not batch:break
            arr[pos:pos+len(batch)]=batch;pos+=len(batch)
        arr.flush();del arr
        splits[f'{split}/{mod}']={'samples':count,'index':path.relative_to(out).as_posix(),'sha256':sha256_file(path)}
    rejected=db.execute('SELECT modality,COUNT(*) FROM rejected GROUP BY modality').fetchall()
    shards=[dict(zip(['name','bytes','sha256','records','verified'],r)) for r in db.execute('SELECT * FROM shards ORDER BY name')]
    db.execute('CREATE INDEX IF NOT EXISTS samples_lookup ON samples(modality,split,record_id,slot)')
    db.commit();db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
    artifacts={'index.sqlite':sha256_file(out/'index.sqlite'),'normalization.json':sha256_file(out/'normalization.json'),
               'source_index.jsonl':info['index_sha256'],'source_info.json':sha256_file(out/'source_info.json'),
               'protocol.json':sha256_file(out/'protocol.json')}
    bundle={'version':VERSION,'dataset':name,'status':'ready','scope':'development' if limit else 'full',
            'backend':'eoz' if name=='prithvi' else 'npz','source_root':info['source_root'],
            'records':info['records'],'samples':splits,'normalization':'normalization.json','artifacts':artifacts,
            'identity':identity,'source_manifest_hashes':info['source_manifest_hashes'],
            'source_bytes':state['raw_bytes'],'packed_bytes':state['packed_bytes'],
            'shards':shards,'all_packed_records_byte_verified':name=='prithvi',
            'rejected':dict(rejected),'auxiliary':info['auxiliary'],'elapsed_seconds':state['elapsed_seconds']}
    json_write(out/'bundle.json',bundle)
    json_write(out/'progress.json',{'dataset':name,'status':'complete','records':info['records'],
               'samples':{k:v['samples'] for k,v in splits.items()},'elapsed_seconds':state['elapsed_seconds']})
    print(f'{name}: COMPLETE {info["records"]} records; '+json.dumps({k:v['samples'] for k,v in splits.items()}),flush=True)
    return bundle
