"""Bounded, real-data single-H200 hardware check; never a quality/epoch result."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import torch
from dit_model import build
from vscore_train import tensor_hash


def run(variant, checkpointing, data_path, result_path):
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    plan=json.loads((Path(__file__).parent/'dit_plan.json').read_text())
    arch=dict(plan['architecture'],gradient_checkpointing=checkpointing)
    torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
    batches=torch.load(data_path,map_location='cpu',weights_only=True)
    net=build(variant,architecture=arch).train()
    initial_hash=tensor_hash(net.denoise_fn.state_dict())
    encoder_hash=tensor_hash(net.prithvi_encoder.state_dict())
    ema=copy.deepcopy(net.denoise_fn).eval().requires_grad_(False)
    optimizer=torch.optim.Adam(net.denoise_fn.parameters(),lr=plan['learning_rate'])
    torch.manual_seed(plan['seed']+1);torch.cuda.manual_seed_all(plan['seed']+1)
    torch.cuda.reset_peak_memory_stats()
    timings=[];losses=[];gradients=[]
    for i in range(16):
        # 5 warmup + 10 full-batch measured updates, then a genuine four-record tail.
        raw=batches[i%2] if i<15 else batches[2]
        torch.cuda.synchronize();start=time.perf_counter()
        batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in raw.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=net(batch)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(net.denoise_fn.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for e,p in zip(ema.parameters(),net.denoise_fn.parameters()):e.lerp_(p,.0001)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if 5<=i<15:timings.append(seconds)
        losses.append(loss.item());gradients.append(grad.item())
        print(json.dumps(dict(variant=variant,checkpointing=checkpointing,update=i+1,
                              records=len(batch['HR']),seconds=seconds,loss=loss.item(),grad=grad.item())),flush=True)
        if i==15:assert len(batch['HR'])==4
    allocated=torch.cuda.max_memory_allocated()/1e9
    reserved=torch.cuda.max_memory_reserved()/1e9
    cross_grad=net.denoise_fn.blocks[0].cross_attention.kv.weight.grad
    assert cross_grad is not None and torch.isfinite(cross_grad).all() and cross_grad.abs().sum()>0
    net.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        prediction=net.sample(batch,5,torch.randn_like(batch['HR']))
    assert prediction.shape==batch['HR'].shape and torch.isfinite(prediction).all()
    # Exercise actual model/Adam/EMA/RNG serialization without creating a second GPU copy.
    check=Path(result_path).with_suffix('.checkpoint.pt')
    torch.save(dict(model=net.denoise_fn.state_dict(),optimizer=optimizer.state_dict(),
                    ema=ema.state_dict(),rng=torch.cuda.get_rng_state()),check)
    saved=torch.load(check,map_location='cpu',weights_only=True)
    assert tensor_hash(saved['model'])==tensor_hash(net.denoise_fn.state_dict())
    assert len(saved['optimizer']['state'])>0
    net.denoise_fn.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
    ema.load_state_dict(saved['ema']);torch.cuda.set_rng_state(saved['rng'])
    del saved
    check.unlink()  # This probe's temporary checkpoint only; no experiment weights.
    result=dict(phase='passed',variant=variant,checkpointing=checkpointing,gpu=torch.cuda.get_device_name(),
                batch_size=64,unique_train_records=132,record_presentations=964,updates=16,tail_records=4,
                warmup_updates=5,measured_updates=10,mean_update_seconds=sum(timings)/len(timings),
                median_update_seconds=sorted(timings)[len(timings)//2],update_seconds=timings,
                peak_train_allocated_GB=allocated,peak_train_reserved_GB=reserved,
                peak_all_allocated_GB=torch.cuda.max_memory_allocated()/1e9,
                losses=losses,grad_norms_before_clip=gradients,finite_backward=True,
                sampling_5_steps_finite=True,checkpoint_reload=True,
                initial_dit_sha256=initial_hash,frozen_encoder_sha256=encoder_hash,
                parameters=sum(p.numel() for p in net.denoise_fn.parameters()),
                scope='Technical feasibility only. Cached real train batches; excludes dataset I/O, validation, queues. Not an epoch or quality comparison.',
                nominal_epoch_compute_hours=(sum(timings)/len(timings))*40763/3600)
    Path(result_path).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['flow','epsilon'],required=True)
    p.add_argument('--checkpointing',type=int,choices=[0,1],required=True)
    p.add_argument('--data',required=True);p.add_argument('--result',required=True)
    a=p.parse_args()
    try:run(a.variant,bool(a.checkpointing),a.data,a.result)
    except torch.cuda.OutOfMemoryError as exc:
        Path(a.result).write_text(json.dumps(dict(phase='oom',variant=a.variant,checkpointing=bool(a.checkpointing),error=str(exc)))+'\n')
