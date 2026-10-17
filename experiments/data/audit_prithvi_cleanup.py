"""Check that no original files changed or appeared since lossless packing."""
from pathlib import Path
import json
import os
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from eo_data.codec import FILENAMES
from eo_data.core import json_write


def main():
    started=time.time()
    source=ROOT/'prithvi_data'
    directory=ROOT/'processed_data/v1/prithvi'
    actual={}; stack=[source]
    while stack:
        folder=stack.pop()
        with os.scandir(folder) as entries:
            for e in entries:
                if e.is_symlink(): raise ValueError(f'Unexpected source link: {e.path}')
                if e.is_dir(follow_symlinks=False): stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    s=e.stat(follow_symlinks=False)
                    actual[Path(e.path).relative_to(source).as_posix()]=(s.st_size,s.st_mtime_ns)
                else: raise ValueError(f'Unexpected source entry: {e.path}')
    count=len(actual); total=sum(s[0] for s in actual.values())
    print(json.dumps({'phase':'original_inventory','files':count,'GiB':total/2**30}),flush=True)
    db=sqlite3.connect((directory/'index.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True)
    checked=0
    for (loc_text,) in db.execute('SELECT loc FROM records ORDER BY id'):
        loc=json.loads(loc_text)
        for name,sig in zip(FILENAMES,loc['source_signatures']):
            relative=loc['path']+'/'+name
            if actual.pop(relative,None)!=(sig['size'],sig['mtime_ns']):
                raise ValueError(f'Original changed or disappeared after packing: {relative}')
            checked+=1
    db.close()
    bundle=json.loads((directory/'bundle.json').read_text())
    for relative,sig in bundle['auxiliary'].items():
        if actual.pop(relative,None)!=(sig['size'],sig['mtime_ns']):
            raise ValueError(f'Original auxiliary changed: {relative}')
        checked+=1
    if actual: raise ValueError(f'Unarchived source files: {list(actual)[:10]}')
    old_zip=ROOT/'prithvi_data.zip'
    if old_zip.stat().st_size!=147517786920: raise ValueError('Old ZIP identity changed')
    report={'status':'passed','original_files':count,'checked_files':checked,
            'original_bytes':total,'old_zip_bytes':old_zip.stat().st_size,
            'source_changed':False,'seconds':time.time()-started}
    json_write(ROOT/'experiments/data/original_prithvi_cleanup_audit.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
