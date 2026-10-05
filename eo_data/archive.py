"""ZIP64 container around seekable EOZ frames; no second lossy encoding."""
from pathlib import Path
import hashlib
import json
import os
import struct
import time
import zipfile

from .core import VERSION, json_write, sha256_file

PREFIX = 'processed_data/v1/prithvi/'
CONTENTS = 'ARCHIVE_CONTENTS.json'


class ArchiveFrames:
    def __init__(self, root, storage):
        self.path = (Path(root)/storage['archive']).resolve()
        stat = self.path.stat()
        if stat.st_size != storage['size'] or stat.st_mtime_ns != storage['mtime_ns']:
            raise ValueError('Archive changed or moved without reattaching; run attach-prithvi')
        self.stream = self.path.open('rb')
        with zipfile.ZipFile(self.path) as z:
            entries = z.infolist()
            if len(entries) != len({i.filename for i in entries}):
                raise ValueError('Duplicate archive members')
            self.entries = {i.filename: i for i in entries}
        self.offsets = {}

    def read(self, loc):
        member = PREFIX+'packs/'+loc['shard']
        info = self.entries[member]
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
            raise ValueError('EOZ members must be stored and unencrypted for direct seeking')
        if member not in self.offsets:
            self.stream.seek(info.header_offset)
            header = self.stream.read(30)
            if header[:4] != b'PK\x03\x04':
                raise ValueError('Invalid ZIP local header')
            name_len, extra_len = struct.unpack_from('<HH', header, 26)
            name = self.stream.read(name_len).decode('utf-8')
            if name != member:
                raise ValueError('ZIP local/central member mismatch')
            self.offsets[member] = info.header_offset+30+name_len+extra_len
        offset, length = loc['offset'], loc['length']
        if offset < 0 or length <= 0 or offset+length > info.file_size:
            raise ValueError('Frame exceeds stored shard')
        self.stream.seek(self.offsets[member]+offset)
        frame = self.stream.read(length)
        if len(frame) != length:
            raise ValueError('Truncated archive frame')
        return frame

    def close(self):
        self.stream.close()


def verify_archive(path):
    path = Path(path).resolve()
    started = time.time()
    with zipfile.ZipFile(path) as z:
        members = z.infolist()
        manifest = json.loads(z.read(CONTENTS))
        names = [i.filename for i in members]
        if len(names) != len(set(names)) or set(names) != set(manifest['files']) | {CONTENTS}:
            raise ValueError('Archive member inventory mismatch')
        for n, (member, expected) in enumerate(manifest['files'].items(), 1):
            info = z.getinfo(member)
            if info.file_size != expected['size'] or info.compress_type != expected['compression']:
                raise ValueError(f'Archive member schema mismatch: {member}')
            h = hashlib.sha256()
            with z.open(member) as source:
                for block in iter(lambda: source.read(8*1024*1024), b''):
                    h.update(block)
            if h.hexdigest() != expected['sha256']:
                raise ValueError(f'Archive content hash mismatch: {member}')
            if n % 64 == 0:
                print(json.dumps({'phase':'verify_archive','files':n,'total':len(manifest['files'])}), flush=True)
        bundle = json.loads(z.read(PREFIX+'bundle.json'))
        for shard in bundle['shards']:
            assert manifest['files'][PREFIX+'packs/'+shard['name']]['sha256'] == shard['sha256']
    print('Archive members verified; computing whole-file SHA-256', flush=True)
    digest = sha256_file(path)
    report = {'status':'passed','path':str(path),'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,
              'sha256':digest,'members':len(members),'records':bundle['records'],
              'all_member_sha256_verified':True,'seconds':time.time()-started}
    json_write(path.with_suffix(path.suffix+'.verification.json'), report)
    path.with_suffix(path.suffix+'.sha256').write_text(f'{digest}  {path.name}\n', encoding='utf-8')
    return report


def create_archive(root, destination, prepared=None):
    root = Path(root).resolve()
    directory = Path(prepared or root/'processed_data/v1')/'prithvi'
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(destination.suffix+'.partial')
    if temporary.exists():
        raise FileExistsError(f'Previous partial archive needs inspection: {temporary}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = json.loads((directory/'bundle.json').read_text(encoding='utf-8'))
    if bundle['status'] != 'ready' or not bundle['all_packed_records_byte_verified']:
        raise ValueError('A completed and byte-verified Prithvi build is required')
    expected = {PREFIX+'packs/'+s['name']:s['sha256'] for s in bundle['shards']}
    expected.update({PREFIX+k:v for k,v in bundle['artifacts'].items()})
    expected.update({PREFIX+v['index']:v['sha256'] for v in bundle['samples'].values()})
    expected.update({PREFIX+'original_aux/'+k:v['sha256'] for k,v in bundle['auxiliary'].items()})
    files = [(directory/'packs'/s['name'], PREFIX+'packs/'+s['name'], zipfile.ZIP_STORED) for s in bundle['shards']]
    excluded = {'storage.json','.prepare.lock'}
    for p in sorted(directory.rglob('*')):
        if p.is_file() and p.parent != directory/'packs' and p.name not in excluded and not p.name.endswith(('-wal','-shm','.tmp')):
            files.append((p, PREFIX+p.relative_to(directory).as_posix(), zipfile.ZIP_DEFLATED))
    package = Path(__file__).parent
    for p in sorted(package.glob('*.py')) + [package/'README.md',package/'requirements.txt']:
        if p.exists(): files.append((p,'eo_data/'+p.name,zipfile.ZIP_DEFLATED))
    manifest = {'format':'eoz-in-zip64-v1','dataset':'prithvi','data_version':VERSION,'files':{}}
    with zipfile.ZipFile(temporary,'w',allowZip64=True,compresslevel=3) as z:
        for n,(source,member,compression) in enumerate(files,1):
            info = zipfile.ZipInfo(member, time.localtime(source.stat().st_mtime)[:6])
            info.compress_type = compression
            info.external_attr = 0o100644 << 16
            h = hashlib.sha256(); size = 0
            with source.open('rb') as src, z.open(info,'w',force_zip64=True) as dst:
                for block in iter(lambda: src.read(8*1024*1024), b''):
                    h.update(block); dst.write(block); size += len(block)
            digest = h.hexdigest()
            if member in expected and digest != expected[member]:
                raise ValueError(f'Frozen source changed while archiving: {source}')
            manifest['files'][member] = {'size':size,'sha256':digest,'compression':compression}
            if n % 32 == 0 or n == len(files):
                print(json.dumps({'phase':'create_archive','files':n,'total':len(files),'GiB_written':temporary.stat().st_size/2**30}),flush=True)
        z.writestr(CONTENTS,json.dumps(manifest,indent=2).encode(),compress_type=zipfile.ZIP_DEFLATED)
    temporary.replace(destination)
    return verify_archive(destination)


def attach_archive(root, archive, prepared=None, verified_report=None):
    root = Path(root).resolve(); archive = Path(archive).resolve()
    directory = Path(prepared or root/'processed_data/v1')/'prithvi'
    directory.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read(CONTENTS))
        bundle = json.loads(z.read(PREFIX+'bundle.json'))
        if bundle['version'] != VERSION or bundle['dataset'] != 'prithvi':
            raise ValueError('Unsupported dataset archive')
        frozen = set(bundle['artifacts']) | {'bundle.json'} | {v['index'] for v in bundle['samples'].values()}
        frozen |= {'original_aux/'+k for k in bundle['auxiliary']}
        for member, expected in manifest['files'].items():
            if not member.startswith(PREFIX) or member.startswith(PREFIX+'packs/'):
                continue
            relative = member[len(PREFIX):]
            target = (directory/relative).resolve()
            if not target.is_relative_to(directory.resolve()):
                raise ValueError('Unsafe archive path')
            payload = z.read(member)
            if hashlib.sha256(payload).hexdigest() != expected['sha256']:
                raise ValueError(f'Archive metadata hash mismatch: {member}')
            if target.exists():
                if relative in frozen and sha256_file(target) != expected['sha256']:
                    raise ValueError(f'Existing frozen metadata differs: {target}')
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(payload)
    stat = archive.stat()
    storage = {'backend':'zip64_eoz','archive':archive.relative_to(root).as_posix() if archive.is_relative_to(root) else str(archive),
               'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'archive_sha256':None}
    if verified_report:
        if (verified_report['status']!='passed' or verified_report['size']!=stat.st_size
                or Path(verified_report['path']).resolve()!=archive
                or verified_report['mtime_ns']!=stat.st_mtime_ns):
            raise ValueError('Archive verification report mismatch')
        storage['archive_sha256'] = verified_report['sha256']
    json_write(directory/'storage.json',storage)
    return storage
