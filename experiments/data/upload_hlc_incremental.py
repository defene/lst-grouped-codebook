"""Upload completed HLC shards while local packing continues, then verify release."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from upload_huggingface import OUT, STAGES, save, sha


def main():
    stage = STAGES['HLC']
    repo = 'dsaaf/HLS-LST-CrossSensor'
    state = OUT / 'hlc_incremental_status.json'
    receipt_dir = OUT / 'hlc_uploaded_shards'
    api = HfApi()
    if api.whoami()['name'] != 'dsaaf' or api.repo_info(repo, repo_type='dataset').private:
        raise ValueError('Expected the authorized dsaaf account and public HLC repository')
    while True:
        pending = []
        receipts = sorted((OUT / 'shards').glob('*.json'))
        for receipt in receipts:
            row = json.loads(receipt.read_text())
            uploaded = receipt_dir / receipt.name
            if uploaded.exists():
                if json.loads(uploaded.read_text())['sha256'] != row['sha256']:
                    raise ValueError('Completed shard changed')
                local_tar = stage / row['path']
                local_members = stage / row['members']
                if local_tar.exists() and local_members.exists():
                    entries = api.get_paths_info(repo, [row['path']], repo_type='dataset')
                    if len(entries) != 1 or not entries[0].lfs or entries[0].lfs.sha256 != row['sha256']:
                        raise ValueError('Previously uploaded shard is no longer present')
                    if local_tar.resolve().parent != (stage / 'data').resolve():
                        raise ValueError('Unexpected temporary shard path')
                    local_tar.unlink()
                continue
            if not (stage / row['path']).exists() or not (stage / row['members']).exists():
                continue  # This shard is still being rebuilt on the new staging volume.
            if (stage / row['path']).stat().st_size != row['size'] or sha(stage / row['members']) != row['members_sha256']:
                raise ValueError('Local shard differs from receipt')
            pending.append((receipt, row))
            if len(pending) == 1:
                break
        if pending:
            if api.whoami()['name'] != 'dsaaf':
                raise ValueError('Active account changed')
            patterns = [r[k] for _, r in pending for k in ('path', 'members')]
            save(state, {'phase': 'uploading_shards', 'repo_id': repo,
                         'current': patterns, 'completed_shards': len(list(receipt_dir.glob('*.json'))),
                         'updated_unix': time.time()})
            api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=stage,
                              allow_patterns=patterns,
                              commit_message=f'Add verified HLC shards {pending[0][0].stem}-{pending[-1][0].stem}')
            for receipt, row in pending:
                entries = {p.path: p for p in api.get_paths_info(repo, [row['path'], row['members']], repo_type='dataset')}
                for key, expected in [('path', row['sha256']), ('members', row['members_sha256'])]:
                    entry = entries[row[key]]
                    if not entry.lfs or entry.lfs.sha256 != expected:
                        raise ValueError(f'Remote shard hash mismatch: {row[key]}')
                save(receipt_dir / receipt.name, {'sha256': row['sha256'], 'path': row['path'], 'uploaded_unix': time.time()})
                local_tar = stage / row['path']
                if local_tar.resolve().parent != (stage / 'data').resolve():
                    raise ValueError('Unexpected temporary shard path')
                local_tar.unlink()
            print(f'Committed and verified {len(list(receipt_dir.glob("*.json")))} HLC shards', flush=True)
            continue
        release_path = stage / 'release.json'
        if release_path.exists():
            release = json.loads(release_path.read_text())
            if len(receipts) != len(release['shards']):
                raise ValueError('Release/shard count mismatch')
            patterns = [r['path'] for r in release['files'] if not r['path'].startswith('data/')] + ['release.json']
            api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=stage, allow_patterns=patterns,
                              commit_message='Complete HLC metadata, frozen indices and release checksums')
            subprocess.run([sys.executable, str(Path(__file__).with_name('upload_huggingface.py')),
                            '--owner', 'dsaaf', '--visibility', 'public', '--dataset', 'HLC', '--verify-only'], check=True)
            save(state, {'phase': 'complete', 'repo_id': repo, 'completed_shards': len(receipts), 'completed_unix': time.time()})
            return
        save(state, {'phase': 'waiting_for_local_shards', 'repo_id': repo,
                     'completed_shards': len(list(receipt_dir.glob('*.json'))), 'updated_unix': time.time()})
        time.sleep(20)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(OUT / 'hlc_incremental_failure.json', {'type': type(exc).__name__, 'message': str(exc), 'time': time.time()})
        raise
