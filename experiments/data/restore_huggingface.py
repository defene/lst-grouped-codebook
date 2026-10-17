"""Restore a versioned HLP/HLC Hub snapshot without changing frozen indices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside snapshot: {relative}")
    return path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def restore(root):
    root = root.resolve()
    release = json.loads((root / 'release.json').read_text(encoding='utf-8'))
    # Validate manifests and reader/indices before trusting their restoration instructions.
    for item in release['files']:
        path = safe_path(root, item['path'])
        if not path.is_file() or path.stat().st_size != item['size']:
            raise ValueError(f"Missing or truncated file: {path}")
        if item['path'].startswith('data/'):
            continue  # Every extracted byte is verified below against the member manifest.
        if digest(path) != item['sha256']:
            raise ValueError(f"Checksum mismatch: {path}")
    if release['dataset'] == 'HLP':
        item = next(x for x in release['files'] if x['path'].endswith('.zip'))
        path = safe_path(root, item['path'])
        os.utime(path, ns=(path.stat().st_atime_ns, item['mtime_ns']))
        if path.stat().st_mtime_ns != item['mtime_ns']:
            raise ValueError('Filesystem cannot preserve required nanosecond timestamp')
        print('HLP archive verified; frozen timestamp restored. Use HLP v2 prepared explicitly.')
        return
    restored = 0
    for shard in release['shards']:
        records = {}
        with safe_path(root, shard['members']).open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['path'] in records:
                    raise ValueError('Duplicate member')
                records[row['path']] = row
        seen = set()
        with tarfile.open(safe_path(root, shard['path']), 'r|') as archive:
            for member in archive:
                if not member.isfile() or member.name not in records or member.name in seen:
                    raise ValueError(f'Unexpected archive member: {member.name}')
                seen.add(member.name)
                row = records[member.name]
                if member.size != row['size']:
                    raise ValueError(f'Wrong member size: {member.name}')
                target = safe_path(root, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                # Existing restored files are checked, never blindly overwritten.
                if target.exists():
                    if target.stat().st_size != row['size'] or digest(target) != row['sha256']:
                        raise ValueError(f'Existing file differs: {target}')
                else:
                    temporary = target.with_name(target.name + '.hf-restore-partial')
                    h = hashlib.sha256()
                    with archive.extractfile(member) as source, temporary.open('wb') as dest:
                        for block in iter(lambda: source.read(8 * 1024**2), b''):
                            dest.write(block)
                            h.update(block)
                    if h.hexdigest() != row['sha256']:
                        raise ValueError(f'Member checksum mismatch: {target}')
                    temporary.replace(target)
                os.utime(target, ns=(target.stat().st_atime_ns, row['mtime_ns']))
                if target.stat().st_mtime_ns != row['mtime_ns']:
                    raise ValueError(f'Filesystem timestamp precision mismatch: {target}')
                restored += 1
        if seen != records.keys():
            raise ValueError(f'Missing members: {shard["path"]}')
        print(f'Restored and verified {restored} files', flush=True)
    print('HLC restored; use prepared="processed_data/v1".')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    restore(parser.parse_args().root)
