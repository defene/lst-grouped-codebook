from pathlib import Path
import io
import json
import shutil
import sqlite3
import time
import numpy as np
from . import codec
from .core import canonical, dataset_name, json_write, sha256_file
from .dataset import open_dataset

def verify_all(root,output=None,dataset='all',samples=16,allow_development=False):
    root=Path(root).resolve();output=Path(output or root/'processed_data/v1')
    names=['prithvi','lstsr_tb'] if dataset=='all' else [dataset_name(dataset)]
    report={}
    for name in names:
        directory=output/name
        bundle=json.loads((directory/'bundle.json').read_text(encoding='utf-8'))
        checked={};started=time.perf_counter()
        for key,entry in bundle['samples'].items():
            split,mod=key.split('/')
            ds=open_dataset(name,mod,split,root=root,prepared=output,allow_development=allow_development)
            indices=np.unique(np.linspace(0,len(ds)-1,min(samples,len(ds)),dtype=int))
            for index in indices:
                item=ds[int(index)]
                image=item['image'];mask=item['reference_mask']
                assert image.shape==((6,256,256) if mod=='hls' else (1,256,256))
                assert image.dtype==np.float32 and mask.dtype==bool
                assert np.isfinite(image).all() and np.array_equal(image,item['target'])
                assert not np.shares_memory(image,item['target'])
                assert (image[~np.broadcast_to(mask,image.shape)]==0).all()
                _,_,slot,_,row=ds._record(int(index));loc=json.loads(row[3])
                if name=='prithvi':
                    parts=ds._parts(loc,row[4])
                    raw=np.load(io.BytesIO(parts[0 if mod=='hls' else 2]),allow_pickle=False)
                    f=np.load(io.BytesIO(parts[1]),allow_pickle=False) if mod=='hls' else None
                    reference,rmask=canonical(raw,mod,name,fmask=f)
                    original=ds.source/loc['path']
                    if original.exists():
                        for base,payload in zip(codec.FILENAMES,parts):
                            if (original/base).read_bytes()!=payload:raise ValueError('Original/packed byte mismatch')
                elif mod=='hls':
                    raw,f,v=ds._npz(loc);reference,rmask=canonical(raw,mod,name,fmask=f,source_valid=v)
                else:
                    raw,v=ds._npz(loc);reference,rmask=canonical(raw[slot],mod,name,source_valid=v[slot])
                assert np.array_equal(mask,rmask)
                valid=np.broadcast_to(mask,reference.shape)
                tolerance=2e-6 if mod=='hls' else 0.005
                if not np.allclose(ds.denormalize(image)[valid],reference[valid],rtol=0,atol=tolerance):
                    raise ValueError('Normalization round trip failed')
            checked[key]={'samples':len(ds),'checked':len(indices),'roundtrip_passed':True}
            ds.close()
        report[name]={'status':'passed','scope':bundle['scope'],'profiles':checked,
                      'all_compressed_records_verified_during_preparation':bundle['all_packed_records_byte_verified'],
                      'seconds':time.perf_counter()-started}
        json_write(directory/'verification.json',report[name])
    return report

def restore_prithvi(root,prepared,destination,limit=None):
    root=Path(root).resolve();directory=Path(prepared or root/'processed_data/v1')/'prithvi'
    destination=Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Restore destination must be absent or empty')
    destination.mkdir(parents=True,exist_ok=True)
    bundle=json.loads((directory/'bundle.json').read_text(encoding='utf-8'))
    reader=open_dataset('prithvi','hls',root=root,prepared=directory.parent,allow_development=True)
    db=reader._connect()
    n=0
    for key,loc_text,digest in db.execute('SELECT key,loc,digest FROM records ORDER BY id'):
        loc=json.loads(loc_text)
        parts=reader._parts(loc,digest)
        folder=(destination/key).resolve()
        if not folder.is_relative_to(destination):raise ValueError('Unsafe restored path')
        folder.mkdir(parents=True,exist_ok=True)
        for base,payload in zip(codec.FILENAMES,parts):(folder/base).write_bytes(payload)
        n+=1
        if limit and n>=limit:break
    reader.close()
    for rel,info in bundle['auxiliary'].items():
        src=directory/'original_aux'/rel;dst=(destination/rel).resolve()
        if not dst.is_relative_to(destination):raise ValueError('Unsafe auxiliary path')
        if sha256_file(src)!=info['sha256']:raise ValueError('Auxiliary hash mismatch')
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    return {'restored_records':n,'full_dataset':n==bundle['records'],'destination':str(destination)}
