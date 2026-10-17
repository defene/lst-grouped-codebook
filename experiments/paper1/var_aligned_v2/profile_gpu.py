"""Profile the actual 109M codecs on frozen real HLS samples, one GPU process at a time."""
from pathlib import Path
import argparse
import copy
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import DataLoader
from eo_denoise.config import DEFAULT
from eo_denoise.data import PairedDataset
from eo_denoise.engine import seed_everything,device_for,memory_stats,move_batch,inputs_only,amp_context
from eo_denoise.models import CrossViewModel
from eo_denoise.losses import objective
from eo_data.core import json_write


def config(backbone):
    c=copy.deepcopy(DEFAULT)
    c['data'].update(modalities=['hls'],train_limit=16,val_limit=16,train_views=1)
    c['noise'].update(coverages=[.1],clean_probability=0)
    c['loss'].update(stability=0,cross_view=0)
    c['model'].update(backbone=backbone,quantizer='var_vq',codebook_size=4096,
        eo_dim=768,eo_depth=11,eo_heads=12,eo_decoder_dim=512,eo_decoder_depth=9,eo_decoder_heads=8)
    c['train'].update(device='cuda',workers=0)
    return c


def run(args):
    c=config(args.backbone)
    c['model']['activation_checkpointing']=args.checkpoint
    seed_everything(c['seed']);torch.set_num_threads(4)
    device=device_for(c)
    ds=PairedDataset(c,'train')
    batch=next(iter(DataLoader(ds,batch_size=args.batch)))
    model=CrossViewModel(c).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.01)
    samples=[]
    for step in range(4):
        torch.cuda.synchronize();start=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        moved=move_batch(batch,device,model.codecs)
        with amp_context(c,device):pred=model(inputs_only(moved,model.codecs))
        loss,terms=objective(pred,moved,c)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        opt.step();torch.cuda.synchronize()
        samples.append(dict(step=step+1,seconds=time.perf_counter()-start,loss=float(loss.detach()),
            gradient_norm=float(norm),memory=memory_stats(device)))
        print(json.dumps(samples[-1]),flush=True)
    result=dict(backbone=args.backbone,batch=args.batch,activation_checkpointing=args.checkpoint,
        parameters=sum(p.numel() for p in model.parameters()),samples=samples,
        steady_samples_per_second=args.batch/(sum(s['seconds'] for s in samples[1:])/3))
    json_write(args.output,result)
    ds.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone',required=True,choices=['var_conv','var_prithvi'])
    p.add_argument('--batch',type=int,default=4);p.add_argument('--checkpoint',action='store_true')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:run(args)
    except Exception as e:
        json_write(args.output,dict(status='failed',error=repr(e),memory=memory_stats(torch.device('cuda'))))
        raise
