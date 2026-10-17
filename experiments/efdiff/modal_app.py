"""Modal pipeline: HLP integrity -> CPU stats + GPU smoke -> five complete epochs."""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
app = modal.App('efdiff-hlp-v2-reproduction')
data_volume = modal.Volume.from_name('hls-lst-datasets-20260914', environment_name='main')
runs_volume = modal.Volume.from_name('efdiff-hlp-v2-runs', environment_name='main', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('libgl1', 'libglib2.0-0')
    .pip_install('torch==2.8.0', 'torchvision==0.23.0', 'numpy==2.2.6', 'scipy==1.16.2',
                 'PyYAML==6.0.2', 'zstandard==0.24.0', 'terratorch==1.2.13')
    .env({'PYTHONPATH': '/code:/data', 'HF_HOME': '/runs/huggingface',
          'HF_XET_CACHE': '/tmp/hf-xet', 'OMP_NUM_THREADS': '1'})
    .add_local_dir(HERE, '/code', ignore=['.git', '__pycache__', '*.pyc']))
mounts = {'/data': data_volume, '/runs': runs_volume}
RUN = 'hlp-v2-x0-5epoch-20260915'


@app.function(image=image, volumes=mounts, cpu=4, memory=8192, timeout=7200)
def preflight():
    import hashlib
    import os
    import sys
    from data import HLPPairs
    from eo_data import open_dataset
    root = Path('/data').resolve()
    manifest = root/'_transfer/files.jsonl'
    plan = json.loads((root/'_transfer/plan.json').read_text())
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == plan['manifest_sha256']
    count = 0
    for line in manifest.open():
        r = json.loads(line)
        if r['group'] not in ('HLP', 'reader', 'docs'):
            continue
        p = (root/r['path']).resolve()
        assert p.is_relative_to(root) and p.stat().st_size == r['size']
        os.utime(p, ns=(p.stat().st_atime_ns, r['mtime_ns']))
        count += 1
    archive = root/'artifacts/prithvi_data_v1_20260909.zip'
    print('Verifying full HLP archive SHA256 before any training', flush=True)
    h = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(16*1024*1024), b''):
            h.update(block)
    expected = (root/'artifacts/prithvi_data_v1_20260909.zip.sha256').read_text().split()[0]
    assert h.hexdigest() == expected
    ids = {}
    tiles = {}
    reads = []
    for split in ('train', 'val', 'test'):
        ds = HLPPairs(root, split=split)
        ids[split] = set(map(int, ds.lst.indices[:, 0]))
        tiles[split] = {r[0] for r in ds.lst._connect().execute('SELECT DISTINCT tile FROM records WHERE split=?', (split,))}
        for i in (0, len(ds)//2, len(ds)-1):
            sample = ds[i]
            assert sample['mask'].any() and sample['HLS'].shape == (6,224,224)
        reads.append({'split': split, 'records': len(ds), 'tiles': len(tiles[split])})
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not ids[a] & ids[b] and not tiles[a] & tiles[b]
    assert [len(tiles[s]) for s in ('train','val','test')] == [2796,147,151]
    result = {'status': 'passed', 'scope': 'HLP-only full archive hash, paired reads, split isolation',
              'archive_sha256': h.hexdigest(), 'files_checked': count, 'splits': reads}
    out = Path('/runs')/RUN
    out.mkdir(parents=True, exist_ok=True)
    (out/'preflight.json').write_text(json.dumps(result, indent=2))
    data_volume.commit()
    runs_volume.commit()
    return result


@app.function(image=image, volumes=mounts, cpu=10, memory=16384, timeout=14400, max_containers=8)
def stats_shard(shard: int, shards: int):
    from fit_stats import fit_shard
    result = fit_shard('/data', shard, shards, workers=8)
    return result


@app.function(image=image, volumes=mounts, gpu='H200', cpu=12, memory=65536,
              timeout=28800, single_use_containers=True)
def train_gpu(smoke: bool = True):
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from data import HLPPairs, PROTOCOL
    from train import run, save_json
    runs_volume.reload()
    out = Path('/runs')/RUN
    if smoke:
        # Provisional train-only statistics exclusively for the technical check.
        ds = HLPPairs('/data', load_hls=False)
        sums = np.zeros((2,2), dtype=np.float64)
        subset = np.linspace(0, len(ds)-1, 32, dtype=int).tolist()
        for raw in DataLoader(Subset(ds, subset), batch_size=4, num_workers=4):
            for row, mask in enumerate(raw['mask'].numpy()):
                for j, key in enumerate(('condition_c', 'residual_c')):
                    v = raw[key][row].numpy()[mask].astype(np.float64)
                    sums[j] += [v.mean(), np.square(v).mean()]
        stats = {'protocol': PROTOCOL, 'fit_split': 'train', 'records':32, 'scope':'smoke only'}
        for key, (mean, second) in zip(('lst','residual'), sums/32):
            stats[key] = {'mean': float(mean), 'std': float(np.sqrt(max(second-mean*mean,1e-12)))}
        destination = out/'smoke'
    else:
        stats = json.loads((out/'normalization.json').read_text())
        assert json.loads((out/'smoke/status.json').read_text())['phase'] == 'smoke_passed'
        assert json.loads((out/'preflight.json').read_text())['status'] == 'passed'
        destination = out/'formal'
    return run('/data', destination, stats, smoke=smoke, micro=4, workers=8, commit=runs_volume.commit)


@app.function(image=image, volumes=mounts, cpu=1, memory=4096, timeout=86400)
def pipeline(resume: bool = False):
    from fit_stats import merge
    from train import save_json
    out = Path('/runs')/RUN
    out.mkdir(parents=True, exist_ok=True)
    try:
        if resume:
            import hashlib
            assert json.loads((out/'preflight.json').read_text())['status'] == 'passed'
            smoke = json.loads((out/'smoke/status.json').read_text())
            assert smoke['phase'] == 'smoke_passed'
            evidence = json.loads((out/'smoke/provenance.json').read_text())
            for relative, digest in evidence['code_sha256'].items():
                assert hashlib.sha256((Path('/code')/relative).read_bytes()).hexdigest() == digest
            stats = json.loads((out/'normalization.json').read_text())
        else:
            save_json(out/'pipeline.json', {'phase':'preflight'})
            runs_volume.commit()
            preflight.remote()
            runs_volume.reload()
            save_json(out/'pipeline.json', {'phase':'training_statistics_and_gpu_smoke'})
            runs_volume.commit()
            calls = [stats_shard.spawn(i, 8) for i in range(8)]
            smoke = train_gpu.remote(True)
            print('GPU smoke result:', json.dumps(smoke), flush=True)
            assert smoke['phase'] == 'smoke_passed'
            stats = merge([call.get() for call in calls])
            runs_volume.reload()
            save_json(out/'normalization.json', stats)
        save_json(out/'pipeline.json', {'phase':'formal_training', 'smoke':smoke})
        runs_volume.commit()
        while True:
            result = train_gpu.remote(False)
            print('Formal training result:', json.dumps(result), flush=True)
            if result['phase'] == 'training_complete':
                break
        runs_volume.reload()
        save_json(out/'pipeline.json', result)
        runs_volume.commit()
        return result
    except Exception as exc:
        runs_volume.reload()
        save_json(out/'pipeline.json', {'phase':'failed', 'error':f'{type(exc).__name__}: {exc}'})
        runs_volume.commit()
        raise


@app.local_entrypoint()
def main(mode: str = 'pipeline'):
    if mode == 'preflight':
        result = preflight.remote()
    elif mode == 'smoke':
        result = train_gpu.remote(True)
    elif mode == 'resume':
        result = pipeline.remote(True)
    else:
        result = pipeline.remote()
    print(json.dumps(result, indent=2))
