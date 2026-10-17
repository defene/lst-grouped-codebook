"""Single-GPU, equal-membership staged comparison with a bounded deadline.

Historical trainers/configurations remain unchanged. Stage boundaries are not epochs.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from data import HERE, UPSTREAM, HLCPairs, EpochSampler, check_stats
from pilot_data import HLCPairs as ValidationPairs
from epoch1_train import MaskedLoss, move, apply_update, evaluate, save_json
from model.factory import define_G, init_weights

PLAN = HERE/'width32_12h_plan.json'


def run(output, variant, records, smoke=False, stop_after=None):
    import modal
    volume=modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    plan=json.loads(PLAN.read_text())
    cfg=json.loads((HERE/'configs'/f'width32_single_{variant}.json').read_text())
    marker=output/f'stage_{records}.json'
    if marker.exists():
        return json.loads(marker.read_text())
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark=True
    random.seed(20260915)
    np.random.seed(20260915)
    torch.manual_seed(20260915)
    torch.cuda.manual_seed_all(20260915)
    stats=json.loads(Path(plan['normalization_path']).read_text())
    train=HLCPairs('/data',stats=stats,augment=True)
    val=ValidationPairs('/data',split='val',stats=stats)
    check_stats(stats,train)
    assert len(train)==2608772 and len(val)==512 and records<=len(train)
    assert (records==131) if smoke else (records in plan['stage_records'])
    assert not set(map(int,train.lst.indices[:,0])) & set(map(int,val.lst.indices[val.members,0]))
    stats_hash=hashlib.sha256(json.dumps(stats,sort_keys=True).encode()).hexdigest()
    paths=[Path(__file__),PLAN,HERE/'configs'/f'width32_single_{variant}.json',
           HERE/'data.py',HERE/'pilot_data.py',HERE/'epoch1_train.py',HERE/'pilot_membership/val.npy']
    paths+=sorted((UPSTREAM/'model').glob('*.py'))+sorted((UPSTREAM/'util').glob('*.py'))
    hashes={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    opt=copy.deepcopy(cfg)
    opt['phase']='test'
    net=define_G(opt).cuda().train()
    checkpoint=output/'latest.pt'
    if not checkpoint.exists():
        init_weights(net.denoise_fn,init_type='orthogonal')
    net.set_new_noise_schedule(opt['model']['beta_schedule']['train'],'cuda')
    net.loss_func=MaskedLoss(net.loss_type)
    assert sum(p.numel() for p in net.denoise_fn.parameters())==66900097
    assert not any(p.requires_grad for p in net.prithvi_encoder.parameters())
    optimizer=torch.optim.Adam(net.denoise_fn.parameters(),lr=cfg['train']['optimizer']['lr'])
    ema=copy.deepcopy(net.denoise_fn).eval().requires_grad_(False)
    offset=step=0
    previous_seconds=0.
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['hashes']==hashes and state['stats_hash']==stats_hash
        assert state['fingerprint']==train.fingerprint and state['variant']==variant and state['smoke']==smoke
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        ema.load_state_dict(state['ema'])
        offset,step,previous_seconds=state['offset'],state['step'],state['elapsed_seconds']
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        del state
    assert offset<=records and step==math.ceil(offset/64)
    full_order=EpochSampler(train,0).order
    member_hashes={str(n):hashlib.sha256(full_order[:n].astype('<i4').tobytes()).hexdigest()
                   for n in ([131] if smoke else plan['stage_records'])}
    del full_order
    provenance={'config':cfg,'plan':plan,'hashes':hashes,'stats_hash':stats_hash,
        'stage_members_sha256':member_hashes,
        'normalization':stats,'fingerprint':train.fingerprint,'variant':variant,'smoke':smoke,
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'selection':'fixed stage EMA; same 512 pilot validation members; no test',
        'epoch_definition':2608772,'unet_parameters':66900097}
    save_json(output/'provenance.json',provenance)
    for p in paths:
        dest=output/'source'/p.relative_to(HERE.parent)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest)
    save_json(output/'normalization.json',stats)
    started=time.time()

    def save():
        payload={'model':net.state_dict(),'optimizer':optimizer.state_dict(),'ema':ema.state_dict(),
            'offset':offset,'step':step,'elapsed_seconds':previous_seconds+time.time()-started,
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
            'hashes':hashes,'stats_hash':stats_hash,'fingerprint':train.fingerprint,
            'variant':variant,'smoke':smoke}
        temp=output/'latest.tmp'
        torch.save(payload,temp)
        temp.replace(checkpoint)
        volume.commit()

    if offset<records:
        loader=DataLoader(train,sampler=EpochSampler(train,0,start=offset,limit=records),
            batch_size=64,num_workers=16,prefetch_factor=4,pin_memory=True,
            generator=torch.Generator().manual_seed(20260915))
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        for raw in loader:
            b=move(raw)
            n=len(raw['index'])
            net.loss_func.mask=b['mask']
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=net(b)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            (loss*n).backward()
            base_lr=cfg['train']['optimizer']['lr']
            factor=max(step,1)/5000 if step<5000 else max(
                .5*(1+math.cos(math.pi*(step-5000)/(40763-5000))),1e-6/base_lr)
            for group in optimizer.param_groups:
                group['lr']=base_lr*factor
            grad=apply_update(net,optimizer,n)
            step+=1
            offset+=n
            with torch.no_grad():
                decay=.9999 if step>=5000 else 0.
                for e,p in zip(ema.parameters(),net.denoise_fn.parameters()):
                    e.lerp_(p,1-decay)
            status={'phase':'training','variant':variant,'step':step,'presented':offset,
                'stage_target':records,'epoch_fraction':offset/len(train),'update_records':n,
                'loss':loss.item(),'grad_norm':grad,'lr':optimizer.param_groups[0]['lr'],
                'elapsed_seconds':previous_seconds+time.time()-started,
                'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9,
                'peak_reserved_GB':torch.cuda.max_memory_reserved()/1e9}
            if step%50==0 or smoke or offset==records:
                print(json.dumps(status),flush=True)
                save_json(output/'status.json',status)
                with (output/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(status)+'\n')
            deadline_stop=not smoke and time.time()>=plan['deadline_unix']-1800
            segment_stop=time.time()-started>=5400 or (stop_after is not None and step>=stop_after)
            if step%500==0 or offset==records or deadline_stop or segment_stop:
                save()
            if offset<records and (deadline_stop or segment_stop):
                result={'phase':'deadline_stop' if deadline_stop else 'checkpointed',
                    'variant':variant,'step':step,'presented':offset,'stage_target':records}
                save_json(output/'segment_result.json',result)
                volume.commit()
                return result
        del loader,b,raw,loss
    assert offset==records and step==math.ceil(records/64)
    live=net.denoise_fn
    net.denoise_fn=ema
    metrics=evaluate(net,val,stats,16,8,limit=4 if smoke else 512,
        capture_path=None if smoke else output/f'predictions_{records}.npz')
    net.denoise_fn=live
    metrics.update(split='val',training_records=offset,updates=step,epoch_fraction=offset/len(train),
        complete_epochs=int(offset==len(train)),smoke=smoke,checkpoint='fixed stage EMA')
    save_json(output/f'validation_{records}.json',metrics)
    torch.save({'ema':ema.state_dict(),'stats':stats,'variant':variant,'config':cfg,
                'training_records':offset,'metrics':metrics},output/f'ema_{records}.pt')
    result={'phase':'smoke_passed' if smoke else 'stage_complete','variant':variant,
        'training_records':offset,'updates':step,'epoch_fraction':offset/len(train),
        'complete_epochs':int(offset==len(train)),'metrics':metrics,
        'elapsed_seconds':previous_seconds+time.time()-started,'test_used':False}
    save_json(marker,result)
    save_json(output/'status.json',result)
    volume.commit()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--variant',choices=['x0','epsilon'],required=True)
    parser.add_argument('--records',type=int,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    result=run(args.output,args.variant,args.records,args.smoke,args.stop_after)
    save_json(Path(args.output)/'segment_result.json',result)
    import modal
    modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main').commit()
