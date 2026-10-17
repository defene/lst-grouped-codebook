"""Upload prepared HLP/HLC releases only to the explicitly selected account.

Run with the Python environment containing huggingface_hub. Visibility must be
explicit, and an existing repository's visibility is never silently changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/huggingface_upload_20261005'
STAGES = {'HLP': OUT / 'HLS-LST-Pairs',
          'HLC': OUT / 'HLS-LST-CrossSensor'}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--visibility', choices=['private', 'public'], required=True)
    parser.add_argument('--dataset', choices=['HLP', 'HLC', 'both'], default='both')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    api = HfApi()
    who = api.whoami()
    if who['name'] != args.owner:
        raise ValueError(f'Logged in as {who["name"]}, expected {args.owner}')
    private = args.visibility == 'private'
    selected = {k: v for k, v in STAGES.items() if args.dataset in [k, 'both']}
    releases = {}
    for dataset, stage in selected.items():
        release = json.loads((stage / 'release.json').read_text(encoding='utf-8'))
        if release['status'] != 'packaged':
            raise ValueError(f'{dataset} is not completely packaged')
        for item in release['files']:
            if args.verify_only and dataset == 'HLC' and item['path'].startswith('data/'):
                continue  # Temporary local shards may be removed after remote verification.
            p = stage / item['path']
            if p.stat().st_size != item['size'] or p.stat().st_mtime_ns != item['mtime_ns']:
                raise ValueError(f'Staged release changed: {p}')
        releases[dataset] = release
    # A free private account cannot accommodate even the HLP archive.
    if private and not who.get('isPro') and sum(x['total_bytes'] for x in releases.values()) > 100_000_000_000:
        raise ValueError('Release exceeds the free-account private quota (100 GB); check storage plan before uploading')
    for dataset, stage in selected.items():
        api = HfApi()  # Re-check identity before each repository.
        if api.whoami()['name'] != args.owner:
            raise ValueError('Active account changed')
        repo = args.owner + '/' + stage.name
        state_path = OUT / f'{dataset.lower()}_upload_status.json'
        release = releases[dataset]
        try:
            info = api.repo_info(repo, repo_type='dataset')
        except RepositoryNotFoundError:
            if args.verify_only:
                raise
            api.create_repo(repo, repo_type='dataset', private=private, exist_ok=False)
            info = api.repo_info(repo, repo_type='dataset')
        if info.private != private:
            raise ValueError(f'{repo} visibility differs; refusing to change it')
        if not args.verify_only:
            save(state_path, {'phase': 'uploading', 'repo_id': repo, 'private': private, 'started_unix': time.time()})
            # Ship useful documentation first. Full files are selected by the release manifest.
            api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=stage,
                              allow_patterns=['README.md', 'DATASHEET.md', '.gitattributes'],
                              commit_message='Add dataset description and lossless release layout')
            api.upload_large_folder(repo_id=repo, repo_type='dataset', folder_path=stage,
                                    allow_patterns=[x['path'] for x in release['files']] + ['release.json'],
                                    num_workers=args.workers, print_report=True, print_report_every=60)
        save(state_path, {'phase': 'verifying', 'repo_id': repo, 'updated_unix': time.time()})
        revision = api.repo_info(repo, repo_type='dataset').sha
        remote = {x.path: x for x in api.list_repo_tree(repo, repo_type='dataset', revision=revision, recursive=True) if hasattr(x, 'size')}
        expected = release['files'] + [{'path': 'release.json', 'size': (stage / 'release.json').stat().st_size,
                                       'sha256': sha(stage / 'release.json')}]
        verified = 0
        for item in expected:
            entry = remote.get(item['path'])
            if entry is None or entry.size != item['size']:
                raise ValueError(f'Remote missing/wrong size: {repo}/{item["path"]}')
            if entry.lfs:
                actual = entry.lfs.sha256
            else:
                path = hf_hub_download(repo, item['path'], repo_type='dataset', revision=revision)
                actual = sha(Path(path))
            if actual != item['sha256']:
                raise ValueError(f'Remote hash mismatch: {repo}/{item["path"]}')
            verified += 1
        save(state_path, {'phase': 'complete', 'repo_id': repo, 'private': private, 'revision': revision,
                          'files_verified': verified, 'bytes': release['total_bytes'],
                          'verification': 'Pinned Hub revision: all file sizes and SHA-256 (LFS metadata or downloaded Git blobs)',
                          'completed_unix': time.time()})
        print(f'Uploaded and verified https://huggingface.co/datasets/{repo} @ {revision}', flush=True)


if __name__ == '__main__':
    main()
