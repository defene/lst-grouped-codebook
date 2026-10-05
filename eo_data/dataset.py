"""One public dataset interface for packed Prithvi and original LSTSR NPZ."""
from collections import OrderedDict
from pathlib import Path
import hashlib
import io
import json
import os
import sqlite3
import numpy as np
from . import codec
from .core import VERSION, canonical, dataset_name, denormalize, modality_name, normalize, sha256_file

PROJECT = Path(__file__).resolve().parents[1]
_VERIFIED = {}

class EODataset:
    """NumPy map-style dataset; compatible with PyTorch's default collator.

    Connections/caches are opened per process; images and targets never alias.
    Index may be int or (epoch,int). No augmentation/noise is enabled in v1.
    """
    def __init__(self, dataset, modality, split='train', root=None, prepared=None,
                 verify=True, allow_development=False):
        self.root = Path(root or PROJECT).resolve()
        self.name = dataset_name(dataset)
        self.modality = modality_name(modality)
        self.split = split
        self.directory = Path(prepared or self.root/'processed_data/v1').resolve()/self.name
        manifest = self.directory/'bundle.json'
        if not manifest.exists():
            raise FileNotFoundError(f'Preprocessing not complete: {manifest}. Run python -m eo_data prepare --dataset {self.name}')
        self.bundle = json.loads(manifest.read_text(encoding='utf-8'))
        if self.bundle['status'] != 'ready' or self.bundle['version'] != VERSION:
            raise ValueError('Unsupported/incomplete prepared dataset')
        for name in ('core.py','codec.py'):
            if sha256_file(Path(__file__).parent/name) != self.bundle['identity']['pipeline_sha256'][name]:
                raise ValueError(f'Processing code differs from the frozen data: {name}')
        if self.bundle['scope'] != 'full' and not allow_development:
            raise ValueError('Development subset cannot be used as a full training dataset')
        key = f'{split}/{self.modality}'
        if key not in self.bundle['samples']:
            raise ValueError(f'No {key} split in {self.name}; available: {list(self.bundle["samples"])}')
        entry = self.bundle['samples'][key]
        if verify:
            checks = dict(self.bundle['artifacts'])
            checks[entry['index']] = entry['sha256']
            for relative, expected in checks.items():
                path = self.directory/relative
                st = path.stat()
                cache_key = (str(path),st.st_size,st.st_mtime_ns,expected)
                if cache_key not in _VERIFIED:
                    if sha256_file(path) != expected:
                        raise ValueError(f'Frozen artifact changed: {path}')
                    _VERIFIED[cache_key] = True
        self.index_path = self.directory/entry['index']
        self.indices = np.load(self.index_path,mmap_mode='r',allow_pickle=False)
        if self.indices.shape != (entry['samples'],3):raise ValueError('Invalid sample index')
        self.stats = json.loads((self.directory/self.bundle['normalization']).read_text(encoding='utf-8'))[self.modality]
        self.source = self.root/self.bundle['source_root']
        storage_file = self.directory/'storage.json'
        self.storage = json.loads(storage_file.read_text(encoding='utf-8')) if storage_file.exists() else None
        self._archive_reader = None
        self._pid = None;self._db = None;self._pack_files = OrderedDict();self._npz_cache = OrderedDict()
        self.fingerprint = {'dataset':self.name,'modality':self.modality,'split':self.split,
                            'protocol':self.bundle['identity']['protocol_sha256'],
                            'sample_index':entry['sha256'],
                            'normalization':self.bundle['artifacts']['normalization.json'],
                            'source_index':self.bundle['identity']['source_index_sha256'],
                            'record_index':self.bundle['artifacts']['index.sqlite'],
                            'reader_code':sha256_file(Path(__file__)),
                            'processing_code':self.bundle['identity']['pipeline_sha256']}

    def __len__(self):return len(self.indices)

    def _connect(self):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if self._db is None:
            uri=(self.directory/'index.sqlite').as_uri()+'?mode=ro&immutable=1'
            self._db=sqlite3.connect(uri,uri=True)
        return self._db

    def _record(self, index):
        if isinstance(index, tuple):_,index=index
        index=int(index)
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        rid,slot,count=map(int,self.indices[index])
        row=self._connect().execute('SELECT key,tile,meta,loc,digest FROM records WHERE id=?',(rid,)).fetchone()
        if row is None:raise ValueError('Record missing from frozen index')
        return index,rid,slot,count,row

    def _parts(self, loc, digest=None):
        if self.storage is not None:
            from .archive import ArchiveFrames
            if self._archive_reader is None:
                self._archive_reader = ArchiveFrames(self.root,self.storage)
            return codec.decode(self._archive_reader.read(loc),digest)
        shard=loc['shard']
        if shard not in self._pack_files:
            self._pack_files[shard]=(self.directory/'packs'/shard).open('rb')
            if len(self._pack_files)>8:self._pack_files.popitem(last=False)[1].close()
        self._pack_files.move_to_end(shard)
        stream=self._pack_files[shard]
        stream.seek(loc['offset'])
        frame=stream.read(loc['length'])
        if len(frame)!=loc['length']:raise ValueError('Truncated compressed frame')
        return codec.decode(frame,digest)

    def _npz(self, loc):
        kind='hls' if self.modality=='hls' else 'lst'
        relative=loc[kind]
        path=(self.source/relative).resolve()
        if not path.is_relative_to(self.source.resolve()):raise ValueError('Source path escapes root')
        expected=loc['source_signatures'][kind]
        stat=path.stat()
        if stat.st_size!=expected['size'] or stat.st_mtime_ns!=expected['mtime_ns']:
            raise ValueError(f'Source file changed since preprocessing: {path}. Revalidate/rebuild before training.')
        if relative not in self._npz_cache:
            with np.load(path,allow_pickle=False) as z:
                if kind=='hls':value=(z['hls_reflectance_i16'],z['hls_fmask'],z['hls_valid'])
                else:value=(z['hr_lst'],z['hr_valid'])
            self._npz_cache[relative]=value
            if len(self._npz_cache)>2:self._npz_cache.popitem(last=False)
        self._npz_cache.move_to_end(relative)
        return self._npz_cache[relative]

    def __getitem__(self, index):
        _,rid,slot,count,row=self._record(index)
        key,tile,_,loc_text,digest=row
        loc=json.loads(loc_text)
        if self.name=='prithvi':
            parts=self._parts(loc)
            raw=np.load(io.BytesIO(parts[0 if self.modality=='hls' else 2]),allow_pickle=False)
            f=np.load(io.BytesIO(parts[1]),allow_pickle=False) if self.modality=='hls' else None
            image,mask=canonical(raw,self.modality,self.name,fmask=f)
        elif self.modality=='hls':
            raw,f,valid=self._npz(loc)
            image,mask=canonical(raw,self.modality,self.name,fmask=f,source_valid=valid)
        else:
            raw,valid=self._npz(loc)
            image,mask=canonical(raw[slot],self.modality,self.name,source_valid=valid[slot])
        if int(mask.sum())!=count:raise ValueError(f'Validity policy/source drift at {key}')
        image=normalize(image,mask,self.stats)
        sample_id=f'{self.name}:{key}:{self.modality}'+(f':m{slot+1:02d}' if slot>=0 else '')
        return {'image':image,'target':image.copy(),'reference_mask':mask.copy(),'input_mask':mask.copy(),
                'corruption_mask':np.zeros_like(mask),'sample_id':sample_id,'dataset_id':self.name,
                'modality':self.modality,'split':self.split,'tile_id':tile,'month':slot+1 if slot>=0 else 0,
                'valid_pixels':count}

    def get_metadata(self,index):
        *_,row=self._record(index)
        return json.loads(row[2])

    def denormalize(self, image):return denormalize(image,self.stats)

    def epoch_indices(self, epoch, seed=20260909):
        digest=hashlib.sha256(f'{seed}:{epoch}:{self.name}:{self.modality}:{self.split}'.encode()).digest()
        rng=np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16],'little')))
        return rng.permutation(len(self))

    def close(self):
        if getattr(self,'_archive_reader',None) is not None:
            self._archive_reader.close();self._archive_reader=None
        if getattr(self,'_db',None) is not None:self._db.close();self._db=None
        for stream in getattr(self,'_pack_files',{}).values():stream.close()
        self._pack_files=OrderedDict();self._npz_cache=OrderedDict()

    def __getstate__(self):
        state=dict(self.__dict__)
        state.update(_db=None,_pid=None,_pack_files=OrderedDict(),_npz_cache=OrderedDict(),indices=None,_archive_reader=None)
        return state

    def __setstate__(self,state):
        self.__dict__.update(state)
        self.indices=np.load(self.index_path,mmap_mode='r',allow_pickle=False)

    def __del__(self):
        try:self.close()
        except Exception:pass

def open_dataset(dataset, modality, split='train', **kwargs):
    return EODataset(dataset,modality,split,**kwargs)

def open_datasets(split='train', **kwargs):
    return {name:{mod:open_dataset(name,mod,split,**kwargs) for mod in ('hls','lst')}
            for name in ('prithvi','lstsr_tb')}

def make_dataloader(dataset, modality, split='train', batch_size=16, num_workers=0, seed=20260909, **dataset_kwargs):
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise ImportError('PyTorch is optional for NumPy reads; install it in your training environment to use make_dataloader.') from e
    ds=open_dataset(dataset,modality,split,**dataset_kwargs)
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(ds,batch_size=batch_size,shuffle=split=='train',num_workers=num_workers,
                      generator=generator,persistent_workers=num_workers>0,drop_last=False)
