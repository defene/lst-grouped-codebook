"""Fixed-checkpoint, resumable HLC test comparison. Never trains or selects models."""
import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from center_data import HLCPairs as CenterPairs, check_stats
from epoch1_train import save_json
import full_test_local as unet
from raw_mmdit_model import build

RUN='hlc-unet-mmdit-fulltest-20261015-v1'
MM='hlc-raw-mmdit-v-b256-fullval-epoch2-20261014-v2'
OLD='hlc-synthetic-full-epoch1-ddp2-io16-20260916-v3'
VARIANTS={'unet_x0':15,'unet_epsilon':50,'mmdit_5':5,'mmdit_50':50}
BATCH=32
CHUNK=512

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class Pairs(CenterPairs):
    def __init__(self,root,split,stats,metadata):
        super().__init__(root,split,stats,augment=False)
        meta=Path(metadata);a=json.loads((meta/'audit.json').read_text())
        assert a['source_hashes']['index.sqlite']==self.lst.fingerprint['record_index']
        assert a['splits'][split]['sample_index_sha256']==self.lst.fingerprint['sample_index']
        assert sha(meta/'latlon_by_record.npy')==a['artifact_sha256']
        self.geo=np.load(meta/'latlon_by_record.npy',mmap_mode='r')
        assert np.isfinite(self.geo[self.lst.indices[:,0]]).all()
        self.fingerprint['test_geo_sidecar']=a['artifact_sha256']
    def __getitem__(self,index):
        r=super().__getitem__(index)
        hs=self.hls[int(self.hls_positions[index])]
        valid=hs['reference_mask'][:,16:240,16:240].astype(bool)
        r['HLS_raw']=np.where(valid,self.hls.denormalize(hs['image'])[:,16:240,16:240],0).astype(np.float32)
        r['HLS_valid']=valid
        r['geo_latlon']=self.geo[r['record_id']].astype(np.float32)
        return r

def aggregate(parts):
    out={}
    for name in ['bicubic',*VARIANTS]:
        n=sum(x['records'] for x in parts);pixels=sum(x['sums'][name]['pixels'] for x in parts)
        sse=sum(x['sums'][name]['sse'] for x in parts);sae=sum(x['sums'][name]['sae'] for x in parts)
        out[name]=dict(records=n,valid_pixels=pixels,rmse_c=math.sqrt(sse/pixels),mae_c=sae/pixels,
                       squared_error_sum=sse,absolute_error_sum=sae,nfe=VARIANTS.get(name,0))
    return out

def run(scope,commit):
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    out=Path('/runs')/RUN/scope;out.mkdir(parents=True,exist_ok=True)
    chunks=out/'chunks';chunks.mkdir(exist_ok=True)
    unet.MODELS=Path('/runs')/OLD/'local_export_epoch1_v1'
    manifest=json.loads((unet.MODELS/'manifest.json').read_text())
    for name,info in manifest['files'].items():
        p=unet.MODELS/name
        assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256']
    source=Path('/runs')/MM/'train/v'
    state=torch.load(source/'best_ema.pt',map_location='cpu',weights_only=False)
    stats=state['stats']
    oldstate=torch.load(unet.MODELS/'x0_epoch1_ema.pt',map_location='cpu',weights_only=False,mmap=True)
    assert oldstate['normalization']==stats, 'Normalization differs: explicit per-model SR conversion required'
    del oldstate
    check_stats(stats,CenterPairs('/data',stats=stats,load_hls=False))
    split='val' if scope=='smoke' else 'test'
    ds=Pairs('/data',split,stats,'/test_metadata')
    count=44 if scope=='smoke' else len(ds)
    assert scope=='smoke' or count==141524
    ids=set(map(int,ds.lst.indices[:,0]))
    if scope!='smoke':
        for s in ['train','val']:
            other=np.load(f'/data/processed_data/v1/lstsr_tb/indices/{s}.lst_hr.npy',mmap_mode='r')
            assert not ids.intersection(map(int,other[:,0]))
        db=ds.lst._connect()
        tiles={s:{r[0] for r in db.execute('SELECT DISTINCT tile FROM records WHERE split=?',(s,))} for s in ['train','val','test']}
        assert len(tiles['test'])==123 and not tiles['test']&(tiles['train']|tiles['val'])
    plan=json.loads((Path(__file__).parent/'raw_mmdit_b256_plan.json').read_text())
    protocol=dict(run=RUN,split=split,records=count,batch=BATCH,chunk=CHUNK,nfe=VARIANTS,
        checkpoint_selection='Frozen UNet epoch1 EMA; MMDiT full-val50 best epoch2; no test selection',
        seed_rule='20261015 + ordered batch start; reset for every model and NFE',
        crop=[16,240,16,240],degradation='synthetic Wald32; common fine target/mask/coarse',
        normalization=stats,fingerprint=ds.fingerprint,unet_weights=manifest,
        mmdit_weights_sha256=sha(source/'best_ema.pt'),architecture=plan['architecture'],
        eval_code_sha256=sha(__file__),torch=torch.__version__,precision='BF16 model, FP32 Celsius, FP64 sums',
        training_performed=False,test_used_for_selection=False,
        caveats=['Historical UNet has different training crop/loss/budget/HLS preprocessing.',
                 'This is synthetic degradation test, not observed coarse LST restoration.',
                 'New test noise seed, fixed batch32; old partial test scores are not reused.'])
    fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:save_json(out/'protocol.json',protocol)
    parts=[]
    for start in range(0,count,CHUNK):
        p=chunks/f'{start:07d}.json'
        if not p.exists():break
        part=json.loads(p.read_text());assert part['protocol']==fingerprint and part['start']==start and part['end']==min(start+CHUNK,count)
        assert sha(chunks/part['csv'])==part['csv_sha256']
        parts.append(part)
    offset=parts[-1]['end'] if parts else 0
    if offset==count:
        done=dict(phase='complete',split=split,records=count,metrics=aggregate(parts),protocol=fingerprint,test_used_for_selection=False)
        save_json(out/'complete.json',done);commit();return done
    mm=build('v',device='cuda',architecture=plan['architecture']).eval().requires_grad_(False)
    mm.denoise_fn.load_state_dict(state['ema'],strict=True);del state
    nets={v:unet.build_net(v,stats).requires_grad_(False) for v in ['x0','epsilon']}
    encoder=torch.jit.load(str(unet.MODELS/'prithvi_encoder.pt'),map_location='cuda').eval()
    loader=DataLoader(ds,sampler=range(offset,count),batch_size=BATCH,num_workers=12,pin_memory=True,
                      generator=torch.Generator().manual_seed(20261015))
    captures=set(np.linspace(0,count-1,16,dtype=int).tolist())
    keys=['index','record_id','month','target_c','condition_c','synthetic_coarse_c','mask','HLS_raw','HLS_valid']
    stored={k:[] for k in keys+list(VARIANTS)}
    rows=[];sums={k:dict(pixels=0,sse=0.,sae=0.) for k in ['bicubic',*VARIANTS]}
    tick=time.time();start=offset;processed=offset;lastcommit=0;timings={k:0. for k in VARIANTS}
    with torch.inference_mode():
        for raw in loader:
            n=len(raw['index']);np.testing.assert_array_equal(raw['index'].numpy(),np.arange(processed,processed+n))
            b={k:raw[k].cuda(non_blocking=True) for k in ['SR','HLS','HLS_raw','HLS_valid','month','geo_latlon']}
            with torch.autocast('cuda',dtype=torch.bfloat16):context=encoder(b['HLS'])
            target=raw['target_c'].numpy();condition=raw['condition_c'].numpy();mask=raw['mask'].numpy()
            errors={'bicubic':unet.error_sums(condition,target,mask)};preds={}
            for tag,nfe in VARIANTS.items():
                torch.manual_seed(20261015+processed);torch.cuda.manual_seed_all(20261015+processed)
                torch.cuda.synchronize();t=time.time()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    if tag.startswith('unet_'):
                        v=tag.removeprefix('unet_');net=nets[v];net.prithvi_encoder.context=context
                        residual=net.super_resolution({'SR':b['SR'],'HLS':b['HLS']},verbose=False,ddim_steps=50 if v=='epsilon' else None)
                    else:
                        mb=dict(b,HLS=b['HLS_raw']);noise=torch.randn(raw['HR'].shape,device='cuda')
                        residual=mm.sample(mb,nfe,noise)
                torch.cuda.synchronize();timings[tag]+=time.time()-t
                pred=condition+residual.float().cpu().numpy()*(3*stats['residual']['std'])+stats['residual']['mean']
                preds[tag]=pred;errors[tag]=unet.error_sums(pred,target,mask)
            for i in range(n):
                row={'index':processed+i,'record_id':int(raw['record_id'][i]),'month':int(raw['month'][i])}
                for tag,err in errors.items():
                    pixels,sse,sae=err[i];sums[tag]['pixels']+=pixels;sums[tag]['sse']+=sse;sums[tag]['sae']+=sae
                    row.update({tag+'_pixels':pixels,tag+'_sse':sse,tag+'_sae':sae})
                rows.append(row)
                if processed+i in captures:
                    for k in keys:stored[k].append(raw[k][i].numpy())
                    for tag in VARIANTS:stored[tag].append(preds[tag][i])
            processed+=n
            save_json(out/'progress.json',dict(phase='evaluating',split=split,records_done=processed,records_total=count,
                committed_records=start,batch=BATCH,seconds=time.time()-tick,gpu=torch.cuda.get_device_name(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),unix=time.time()))
            if processed%CHUNK==0 or processed==count:
                path=chunks/f'{start:07d}.csv'
                with path.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                if stored['index']:np.savez_compressed(chunks/f'{start:07d}.npz',**{k:np.asarray(v) for k,v in stored.items()})
                part=dict(start=start,end=processed,records=processed-start,sums=sums,protocol=fingerprint,csv=path.name,csv_sha256=sha(path),timings=timings)
                save_json(chunks/f'{start:07d}.json',part);parts.append(part);commit()
                print(json.dumps(dict(scope=scope,records=processed,total=count,elapsed=time.time()-tick)),flush=True)
                start=processed;rows=[];sums={k:dict(pixels=0,sse=0.,sae=0.) for k in ['bicubic',*VARIANTS]}
                stored={k:[] for k in stored};timings={k:0. for k in VARIANTS}
                if time.time()-tick>5400 and processed<count:return dict(phase='checkpointed',records=processed)
            elif time.time()-lastcommit>60:commit();lastcommit=time.time()
    assert processed==count
    done=dict(phase='complete',split=split,records=count,metrics=aggregate(parts),protocol=fingerprint,test_used_for_selection=False)
    assert all(x['records']==count and math.isfinite(x['rmse_c']) and math.isfinite(x['mae_c']) for x in done['metrics'].values())
    assert len({x['valid_pixels'] for x in done['metrics'].values()})==1
    save_json(out/'complete.json',done);commit();return done
