"""Bounded real-data single-H200 throughput probe for the 12-hour comparison."""
import copy
import gc
import json
import time
from pathlib import Path


def run():
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from data import HERE, HLCPairs, EpochSampler, check_stats
    from epoch1_train import MaskedLoss, apply_update, move
    from model.factory import define_G
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    stats=json.loads(Path('/runs/hlc-synthetic-full-epoch1-ddp2-io16-20260916-v3/normalization.json').read_text())
    ds=HLCPairs('/data',stats=stats,augment=True)
    check_stats(stats,ds)
    results=[]
    for variant in ('x0','epsilon'):
        torch.manual_seed(20260915)
        cfg=yaml.safe_load((HERE/'configs'/f'epoch1_{variant}.yaml').read_text())
        cfg['model']['unet']['inner_channel']=32
        cfg['phase']='test'
        net=define_G(cfg).cuda().train()
        net.set_new_noise_schedule(cfg['model']['beta_schedule']['train'],'cuda')
        net.loss_func=MaskedLoss(net.loss_type)
        ema=copy.deepcopy(net.denoise_fn).eval().requires_grad_(False)
        optimizer=torch.optim.Adam(net.denoise_fn.parameters(),lr=cfg['train']['optimizer']['lr'])
        loader=DataLoader(ds,sampler=EpochSampler(ds,0,limit=64*60),batch_size=64,
            num_workers=16,prefetch_factor=4,pin_memory=True,
            generator=torch.Generator().manual_seed(20260915))
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        tick=time.perf_counter()
        times=[]
        for raw in loader:
            b=move(raw)
            net.loss_func.mask=b['mask']
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=net(b)
            assert torch.isfinite(loss).item()
            (loss*64).backward()
            apply_update(net,optimizer,64)
            with torch.no_grad():
                for e,p in zip(ema.parameters(),net.denoise_fn.parameters()):
                    e.lerp_(p,.0001)
            torch.cuda.synchronize()
            now=time.perf_counter()
            times.append(now-tick)
            tick=now
        row={'variant':variant,'inner_channel':32,'micro_batch':64,'effective_batch':64,
            'world_size':1,'gpu':torch.cuda.get_device_name(),'warmup_updates':10,
            'measured_updates':50,'seconds_per_update':float(np.mean(times[10:])),
            'p90_seconds_per_update':float(np.percentile(times[10:],90)),
            'update_seconds':times,'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9,
            'peak_reserved_GB':torch.cuda.max_memory_reserved()/1e9,
            'unet_parameters':sum(p.numel() for p in net.denoise_fn.parameters()),
            'scope':'real HLC train inputs; full-train statistics; no experiment checkpoint'}
        results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='update_seconds'}),flush=True)
        del net,ema,optimizer,loader,b,raw,loss
        gc.collect()
        torch.cuda.empty_cache()
    return results
