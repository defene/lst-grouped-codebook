"""Download a fixed OpenImages validation sample and pinned official VAR sources/weights."""
from pathlib import Path
import concurrent.futures as cf
import csv
import hashlib
import json
import subprocess
import sys
import time
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'runs/openimages_var200_20260915'
SEED = '20260915'
META_URL = 'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    for attempt in range(4):
        try:
            with requests.get(url, timeout=(20, 120), stream=True) as response:
                response.raise_for_status()
                part = path.with_suffix(path.suffix+'.partial')
                with part.open('wb') as f:
                    for chunk in response.iter_content(2**20):
                        f.write(chunk)
                part.replace(path)
            return
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = OUT/'metadata/validation-images-with-rotation.csv'
    fetch(META_URL, metadata)
    rows = list(csv.DictReader(metadata.open(encoding='utf-8')))
    assert len(rows) == 41620 and len({r['ImageID'] for r in rows}) == 41620
    selected = sorted(rows, key=lambda r: (hashlib.sha256((SEED+':'+r['ImageID']).encode()).hexdigest(), r['ImageID']))[:200]
    selection = {'dataset': 'OpenImages', 'split': 'validation', 'population': len(rows), 'n': 200,
        'seed': SEED, 'method': 'first 200 by ascending SHA256(seed:ImageID), fixed before inference; no class/quality filter',
        'metadata_url': META_URL, 'metadata_sha256': sha(metadata), 'records': selected,
        'image_source': 'CVDF public S3 mirror, aspect-preserving longest edge <=1024; original downloaded JPEGs retained'}
    save(OUT/'selection.json', selection)
    (OUT/'image_list.txt').write_text(''.join('validation/'+r['ImageID']+'\n' for r in selected), encoding='utf-8')
    print('Fixed selection of 200 out of 41620 validation images', flush=True)

    def image_job(row):
        rid = row['ImageID']
        path = OUT/'images'/f'{rid}.jpg'
        url = f'https://open-images-dataset.s3.amazonaws.com/validation/{rid}.jpg'
        fetch(url, path)
        with Image.open(path) as im:
            im.load()
            size, mode = im.size, im.mode
        return {'image_id': rid, 'url': url, 'sha256': sha(path), 'bytes': path.stat().st_size,
                'size': size, 'mode': mode}

    def weight_job():
        info = requests.get('https://huggingface.co/api/models/FoundationVision/var?blobs=true', timeout=60)
        info.raise_for_status()
        info = info.json()
        filename = 'vae_ch160v4096z32.pth'
        sibling = next(x for x in info['siblings'] if x['rfilename'] == filename)
        revision = info['sha']
        dest = ROOT/'artifacts/pretrained/var'
        subprocess.run([str(Path(sys.executable).with_name('hf.exe')), 'download', 'FoundationVision/var', filename,
                        '--revision', revision, '--local-dir', str(dest)], check=True)
        path = dest/filename
        digest = sha(path)
        assert digest == sibling['lfs']['sha256']
        manifest = {'repository': 'FoundationVision/var', 'revision': revision, 'filename': filename,
                    'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest,
                    'expected_lfs_sha256': sibling['lfs']['sha256'], 'verified': True}
        save(OUT/'checkpoint.json', manifest)
        print('Official VQ-VAE weight SHA256 verified', flush=True)

    def source_job():
        info = requests.get('https://api.github.com/repos/FoundationVision/VAR/commits/main', timeout=60)
        info.raise_for_status()
        revision = info.json()['sha']
        names = ['models/vqvae.py', 'models/quant.py', 'models/basic_vae.py', 'dist.py', 'utils/data.py', 'LICENSE']
        hashes = {}
        for name in names:
            path = OUT/'upstream'/name
            fetch(f'https://raw.githubusercontent.com/FoundationVision/VAR/{revision}/{name}', path)
            hashes[name] = sha(path)
        # Minimal local package marker avoids importing unrelated AR generation dependencies.
        (OUT/'upstream/models/__init__.py').write_text('', encoding='utf-8')
        save(OUT/'upstream_manifest.json', {'repository': 'https://github.com/FoundationVision/VAR',
              'revision': revision, 'sha256': hashes, 'local_package_marker': 'empty models/__init__.py'})
        print('Pinned official VQ-VAE inference sources downloaded', flush=True)

    with cf.ThreadPoolExecutor(max_workers=10) as pool:
        extra = [pool.submit(weight_job), pool.submit(source_job)]
        jobs = [pool.submit(image_job, r) for r in selected]
        done = []
        for future in cf.as_completed(jobs):
            done.append(future.result())
            if len(done) % 25 == 0:
                print(f'Images downloaded and decoded: {len(done)}/200', flush=True)
        for f in extra:
            f.result()
    by_id = {r['image_id']: r for r in done}
    done = [by_id[r['ImageID']] for r in selected]
    save(OUT/'download_manifest.json', {'status': 'complete', 'images': done,
                                       'total_image_bytes': sum(r['bytes'] for r in done)})
    print('PREPARE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
