"""Real-data GPU benchmark with synchronized start for multiple training processes."""
from pathlib import Path
import argparse
import json
import sys
import time
from make_configs import config,ROOT
import torch
from torch.utils.data import DataLoader
from eo_denoise.engine import seed_everything,device_for,memory_stats,move_batch,inputs_only,amp_context
from eo_denoise.data import PairedDataset
from eo_denoise.models import CrossViewModel
from eo_denoise.losses import objective
from eo_data.core import json_write

def run(args):
    c=config('hls',args.backbone,'noisy','pilot')
    c['train']['memory_limit_gib']=args.cap
    seed_everything(17);torch.set_num_threads(4);device=device_for(c)
    ds=PairedDataset(c,'train')
    cpu=next(iter(DataLoader(ds,batch_size=args.batch)))
    batch=move_batch(cpu,device,['hls'])
    model=CrossViewModel(c).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.01)
    def step():
        optimizer.zero_grad(set_to_none=True)
        with amp_context(c,device):pred=model(inputs_only(batch,model.codecs))
        loss,_=objective(pred,batch,c);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        return float(loss.detach())
    for _ in range(3):step()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.ready').write_text('ready')
    if args.barrier:
        deadline=time.time()+90
        while not args.barrier.exists():
            if time.time()>deadline:raise TimeoutError('Benchmark barrier not released')
            time.sleep(.1)
    samples=[];started=time.perf_counter()
    while time.perf_counter()-started<args.seconds:
        before=time.perf_counter();loss=step()
        samples.append(dict(seconds=time.perf_counter()-before,loss=loss))
    elapsed=time.perf_counter()-started
    result=dict(backbone=args.backbone,batch=args.batch,parameters=sum(p.numel() for p in model.parameters()),
        elapsed_seconds=elapsed,steps=len(samples),samples_per_second=args.batch*len(samples)/elapsed,
        memory=memory_stats(device),last_loss=loss,pretrained_weights=False,
        quantizer=c['model']['quantizer'],samples=samples)
    json_write(args.output,result);print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)
    ds.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['conv','prithvi'],required=True)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--cap',type=float,default=26)
    p.add_argument('--seconds',type=float,default=12);p.add_argument('--barrier',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    run(args)
