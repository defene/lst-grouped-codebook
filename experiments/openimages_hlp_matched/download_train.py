"""HLP-matched 218975 OpenImages/CVDF train sample, resumable downloads and full SHA audit."""
import argparse
import concurrent.futures as cf
from collections import Counter
import csv
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timezone
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/openimages_train218975_20260917'
BASE = ROOT/'data/openimages_train100k_20260916'
META_URL = 'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'
SEED, COUNT, POPULATION = '20260916', 218975, 1743042
LOCAL = threading.local()


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    part = path.with_suffix(path.suffix+'.tmp')
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    part.replace(path)


def session():
    if not hasattr(LOCAL, 'http'):
        LOCAL.http = requests.Session()
        LOCAL.http.headers['User-Agent'] = 'OpenImages-research-subset/1.0'
    return LOCAL.http


def _fetch_metadata():
    target = OUT/'metadata/train-images-boxable-with-rotation.csv'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        return target
    part = target.with_suffix('.csv.partial')
    for attempt in range(6):
        try:
            offset = part.stat().st_size if part.exists() else 0
            with session().get(META_URL, headers={'Range': f'bytes={offset}-'} if offset else {},
                               timeout=(20, 90), stream=True) as r:
                r.raise_for_status()
                mode = 'ab' if offset and r.status_code == 206 else 'wb'
                if mode == 'wb':
                    offset = 0
                expected = int(r.headers.get('Content-Length', '0'))
                if r.status_code == 206:
                    assert r.headers['Content-Range'].startswith(f'bytes {offset}-')
                received = 0
                last = time.monotonic()
                with part.open(mode) as f:
                    for chunk in r.iter_content(2**20):
                        f.write(chunk)
                        received += len(chunk)
                        if time.monotonic()-last >= 10:
                            save(OUT/'progress.json', {'status':'downloading_metadata','bytes':offset+received,'updated_at':now()})
                            print(f'Metadata {(offset+received)/2**20:.1f} MiB', flush=True)
                            last = time.monotonic()
                assert not expected or received == expected
            part.replace(target)
            return target
        except (requests.RequestException, AssertionError):
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 20))


def metadata():
    old = BASE/'metadata/train-images-boxable-with-rotation.csv'
    target = OUT/'metadata/train-images-boxable-with-rotation.csv'
    target.parent.mkdir(exist_ok=True)
    if not target.exists() and old.exists():
        os.link(old,target)
    return _fetch_metadata()


def seed_previous(rows, db):
    """Extend the same rank prefix; old files remain independently addressable."""
    selected = [r['ImageID'] for r in rows]
    old_info = json.loads((BASE/'selection.json').read_text(encoding='utf-8'))
    old_verification = json.loads((BASE/'verification.json').read_text(encoding='utf-8'))
    assert old_verification['status']=='complete' and old_info['count']==100000
    assert digest(BASE/'download_manifest.jsonl') == old_verification['manifest_sha256']
    assert digest(BASE/'selected_metadata.jsonl') == old_info['selected_metadata_sha256']
    old_ids=[json.loads(s)['ImageID'] for s in (BASE/'selected_metadata.jsonl').read_text(encoding='utf-8').splitlines()]
    assert selected[:100000] == old_ids
    reused=0
    for line in (BASE/'download_manifest.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line);rid=r['image_id'];source=BASE/r['path'];target=image_path(rid)
        assert source.stat().st_size==r['bytes']
        target.parent.mkdir(exist_ok=True)
        if not target.exists():os.link(source,target)
        db.execute('INSERT OR IGNORE INTO images VALUES (?,?,?,?,?,?,?,?)',
            (rid,str(target.relative_to(OUT)).replace('\\','/'),r['bytes'],r['sha256'],r['width'],r['height'],r['mode'],r['verified_at']))
        reused+=1
        if reused%10000==0:
            db.commit();save(OUT/'progress.json',{'status':'reusing_verified_base','reused':reused,'target':COUNT,'updated_at':now()})
            print(f'Reused {reused}/100000 verified base images',flush=True)
    db.commit()
    save(OUT/'extension.json',{'base':str(BASE),'base_count':100000,'target_count':COUNT,
        'new_images':COUNT-100000,'seed':SEED,'exact_base_prefix_preserved':True,
        'storage_reuse':'NTFS hard links; no mutation of source image bytes or historical manifests',
        'base_manifest_sha256':digest(BASE/'download_manifest.jsonl'),'base_selection_sha256':digest(BASE/'selection.json'),
        'matched_hlp_version':'hlp-tile-validation-v2-20260914','matched_split':'train','created_at':now()})


def select(meta):
    frozen = OUT/'selection.json'
    records = OUT/'selected_metadata.jsonl'
    if frozen.exists():
        info = json.loads(frozen.read_text(encoding='utf-8'))
        assert info['seed'] == SEED and info['count'] == COUNT and info['population'] == POPULATION
        assert info['metadata_sha256'] == digest(meta)
        assert info['selected_metadata_sha256'] == digest(records)
        rows = [json.loads(line) for line in records.read_text(encoding='utf-8').splitlines()]
        assert len(rows) == len({r['ImageID'] for r in rows}) == COUNT
        return rows, info
    save(OUT/'progress.json', {'status':'selecting','updated_at':now()})
    heap, seen = [], set()
    with meta.open(encoding='utf-8', newline='') as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            rid = row['ImageID']
            assert len(rid) == 16 and all(c in '0123456789abcdef' for c in rid) and rid not in seen
            seen.add(rid)
            rank = int(hashlib.sha256((SEED+':'+rid).encode()).hexdigest(), 16)
            candidate = (-rank, rid, row)
            if len(heap) < COUNT:
                heapq.heappush(heap, candidate)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, candidate)
    assert i == len(seen) == POPULATION, (i, len(seen))
    rows = [r for _, _, r in sorted(heap, key=lambda x: (-x[0], x[1]))]
    del seen, heap
    old = ROOT/'runs/openimages_var200_20260915/selection.json'
    old_ids = {r['ImageID'] for r in json.loads(old.read_text(encoding='utf-8'))['records']}
    overlap = old_ids.intersection(r['ImageID'] for r in rows)
    assert not overlap
    with records.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    (OUT/'image_list.txt').write_text(''.join('train/'+r['ImageID']+'\n' for r in rows), encoding='utf-8')
    info = {'dataset':'OpenImages','subset':'CVDF boxable train','split':'train','population':POPULATION,
            'count':COUNT,'seed':SEED,'selection_method':'first 218975 ascending SHA256(seed:ImageID); no class or quality filter; no replacement after download failures',
            'metadata_url':META_URL,'metadata_sha256':digest(meta),'selected_metadata_sha256':digest(records),
            'image_list_sha256':digest(OUT/'image_list.txt'),'fixed_at':now(),'overlap_previous_validation_200':0,
            'image_url_template':'https://open-images-dataset.s3.amazonaws.com/train/{ImageID}.jpg',
            'resolution':'CVDF source bytes; mirror documents longest edge at most 1024, but actual dimensions can exceed 1024; see verification and source_format_exceptions; no local resizing or re-encoding',
            'scope_note':'Random sample within the 1743042-image CVDF train subset, not uniform across the full 9M OpenImages dataset'}
    save(frozen, info)
    print(f'Fixed selection {COUNT}/{POPULATION}; overlap previous validation images = 0', flush=True)
    return rows, info


def image_path(rid):
    return OUT/'images'/rid[:2]/(rid+'.jpg')


def inspect_image(path, rid):
    with Image.open(path) as im:
        im.load()
        w, h, mode, fmt = *im.size, im.mode, im.format
    assert fmt in ('JPEG','MPO','PNG','WEBP','TIFF','GIF','BMP') and min(w, h)>0, (fmt,w,h)
    return (rid, str(path.relative_to(OUT)).replace('\\','/'), path.stat().st_size, digest(path), w, h, mode, now())


def download(rid):
    path = image_path(rid)
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        try:
            return inspect_image(path, rid)
        except (OSError, AssertionError):
            pass  # Replace only this sample's invalid local download after a good fetch.
    part = path.with_suffix('.jpg.partial')
    url = f'https://open-images-dataset.s3.amazonaws.com/train/{rid}.jpg'
    for attempt in range(6):
        try:
            with session().get(url, timeout=(15, 45), stream=True) as r:
                r.raise_for_status()
                expected = int(r.headers.get('Content-Length','0'))
                with part.open('wb') as f:
                    for chunk in r.iter_content(262144):
                        f.write(chunk)
                assert not expected or part.stat().st_size == expected
            # Fully decode before atomically exposing a finished JPEG.
            verified = inspect_image(part, rid)
            part.replace(path)
            return (verified[0], str(path.relative_to(OUT)).replace('\\','/'), *verified[2:])
        except (requests.RequestException, OSError, AssertionError) as error:
            if attempt == 5:
                raise RuntimeError(f'{rid}: {type(error).__name__}: {error}') from error
            time.sleep(min(2**attempt,16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    assert 1 <= args.workers <= 64
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'images').mkdir(exist_ok=True)
    # Advisory OS lock is released automatically on exit; no duplicate downloaders.
    import msvcrt
    lock = (OUT/'download.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    save(OUT/'process.json', {'pid':os.getpid(),'started_at':now(),'workers':args.workers})
    if shutil.disk_usage(OUT).free < 60*2**30:
        raise RuntimeError('Need at least 60 GiB free disk before this download')
    rows, info = select(metadata())
    db = sqlite3.connect(OUT/'downloads.sqlite3')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS images (image_id TEXT PRIMARY KEY,path TEXT,bytes INTEGER,sha256 TEXT,width INTEGER,height INTEGER,mode TEXT,verified_at TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS failures (image_id TEXT PRIMARY KEY,error TEXT,updated_at TEXT)')
    seed_previous(rows, db)
    known = {r[0]:r for r in db.execute('SELECT image_id,path,bytes FROM images')}
    ids = [r['ImageID'] for r in rows]
    assert not (set(known)-set(ids))
    pending = []
    for rid in ids:
        k = known.get(rid)
        p = image_path(rid)
        if not k or not p.exists() or p.stat().st_size != k[2]:
            pending.append(rid)
            db.execute('DELETE FROM images WHERE image_id=?',(rid,))
    db.commit()
    initial_count = db.execute('SELECT COUNT(*) FROM images').fetchone()[0]
    started = time.monotonic()
    initial_bytes = db.execute('SELECT COALESCE(SUM(bytes),0) FROM images').fetchone()[0]

    def progress(status='downloading', extra=None):
        count, total = db.execute('SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM images').fetchone()
        seconds = time.monotonic()-started
        speed = (count-initial_count)/max(seconds,1)
        data = {'status':status,'target':COUNT,'completed':count,'remaining':COUNT-count,
                'image_bytes':total,'image_gib':total/2**30,'workers':args.workers,'pid':os.getpid(),
                'session_seconds':seconds,'session_images_per_second':speed,
                'session_mib_per_second':(total-initial_bytes)/max(seconds,1)/2**20,
                'estimated_remaining_seconds':(COUNT-count)/speed if speed>0 else None,
                'failed':db.execute('SELECT COUNT(*) FROM failures').fetchone()[0],'updated_at':now()}
        data.update(extra or {})
        save(OUT/'progress.json',data)
        print(json.dumps(data),flush=True)

    progress()
    for round_no in range(3):
        failures = []
        last_progress = time.monotonic()
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            queue = iter(pending)
            futures = {}
            def fill():
                while len(futures)<args.workers*2:
                    rid = next(queue,None)
                    if rid is None:
                        return
                    futures[pool.submit(download,rid)] = rid
            fill()
            while futures:
                done, _ = cf.wait(futures, timeout=10, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    rid = futures.pop(f)
                    try:
                        result = f.result()
                        db.execute('INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?)', result)
                        db.execute('DELETE FROM failures WHERE image_id=?',(rid,))
                    except Exception as error:
                        failures.append(rid)
                        db.execute('INSERT OR REPLACE INTO failures VALUES (?,?,?)',(rid,str(error),now()))
                db.commit()
                fill()
                if time.monotonic()-last_progress>=10:
                    progress(extra={'download_round':round_no+1})
                    last_progress = time.monotonic()
                    if shutil.disk_usage(OUT).free<20*2**30:
                        raise RuntimeError('Free disk below 20 GiB')
        if not failures:
            break
        pending = failures
        progress(extra={'retrying':len(pending),'download_round':round_no+1})
    count = db.execute('SELECT COUNT(*) FROM images').fetchone()[0]
    if count != COUNT:
        progress('incomplete')
        raise RuntimeError(f'{COUNT-count} fixed samples still unavailable; no replacement made')
    progress('verifying')
    records = {r[0]:r for r in db.execute('SELECT * FROM images')}
    def verify(rid):
        r = records[rid]
        p = OUT/r[1]
        assert p.stat().st_size == r[2] and digest(p) == r[3], rid
        with Image.open(p) as im:
            im.load()
            assert im.size == (r[4], r[5])
            return im.format, max(im.size), rid
    formats, oversize = Counter(), []
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for done, (fmt, long_edge, rid) in enumerate(pool.map(verify,ids),1):
            formats[fmt] += 1
            if long_edge > 1024:
                oversize.append({'image_id':rid,'long_edge':long_edge,'format':fmt})
            if done%10000==0:
                progress('verifying',{'audited':done})
    with (OUT/'download_manifest.jsonl').open('w',encoding='utf-8') as f:
        fields = ['image_id','path','bytes','sha256','width','height','mode','verified_at']
        for rid in ids:
            f.write(json.dumps(dict(zip(fields,records[rid])))+'\n')
    total = sum(r[2] for r in records.values())
    result = {'status':'complete','images':COUNT,'unique_image_ids':len(set(ids)),'total_image_bytes':total,
              'total_image_gib':total/2**30,'all_images_decoded':True,'all_sha256_audited':True,
              'file_formats':dict(formats),'long_edge_over_1024_count':len(oversize),
              'overlap_previous_validation_200':0,'completed_at':now(),
              'selection_sha256':digest(OUT/'selection.json'),'manifest_sha256':digest(OUT/'download_manifest.jsonl')}
    save(OUT/'verification.json',result)
    save(OUT/'source_format_exceptions.json', {'file_formats':dict(formats),'long_edge_over_1024':oversize})
    (OUT/'README_ZH.md').write_text(f'''# OpenImages 训练子集：固定随机218,975张

已完成218,975张下载、默认图像帧解码和全量SHA256复核，图片共{total/2**30:.2f} GiB。

从CVDF提供的1,743,042张boxable train图片中，按SHA256("{SEED}:"+ImageID)升序取前218,975。没有按类别、质量或重建误差筛选；不是从约900万全集中均匀抽取。下载失败不自动替换样本。与前次200张validation图片的ImageID交集为0；没有声称已核验视觉近重复。

图片保存于 `images/前两位ID/ImageID.jpg`，保留CVDF源字节，本地未裁剪、缩放或重编码。镜像文档称最长边不超过1024，但实测有{len(oversize)}个文件超过此尺寸，实际格式统计为{dict(formats)}；文件名沿用镜像，不凭.jpg后缀判断格式。异常详情见 `source_format_exceptions.json`。多图格式仅验证默认图像帧的解码；保留完整源文件。这218,975张全部作为训练集，与HLP v2 train记录数相同，不从中扣除验证或测试。没有加噪、重建或训练。原10万张完全保留为抽样顺序的前缀，通过NTFS硬链接复用存储；只新增下载118,975张，原子集及历史清单不变。验证/测试集需另设独立成员。

`selection.json`记录抽样规则和来源哈希；`image_list.txt`是官方downloader兼容名单；`selected_metadata.jsonl`保留入选图片原始URL、作者、许可等源元数据；`download_manifest.jsonl`保存路径、尺寸、字节数和SHA256。`verification.json`为完整性报告，`progress.json`为状态。

在项目根目录运行 `.venv-paper1/Scripts/python.exe -m experiments.openimages_hlp_matched.download_train` 可断点续传并再次全量校验。脚本只操作本目录，不更改HLP/HLC及此前200张验证结果。

来源：[OpenImages官方](https://storage.googleapis.com/openimages/web/download_v7.html)、[CVDF镜像](https://github.com/cvdfoundation/open-images-dataset)。
''',encoding='utf-8')
    progress('complete',{'audited':COUNT,'verification':result})
    db.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            save(OUT/'last_error.json',{'error':str(error),'time':now()})
        raise
