"""Finite memory-only comparison; no experiment checkpoint or dataset mutation."""
import argparse
import contextlib
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def run(variant, width, output):
    import torch
    import torch.distributed as dist
    import yaml
    from torch.nn.parallel import DistributedDataParallel as DDP
    from epoch1_train import MaskedLoss, apply_update
    from model.factory import define_G

    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    torch.set_num_threads(8)
    torch.manual_seed(20260923)
    torch.backends.cudnn.benchmark = True
    here = Path(__file__).resolve().parent
    cfg = yaml.safe_load((here/'configs'/f'deadline_fast_{variant}.yaml').read_text())
    cfg['model']['unet']['inner_channel'] = width
    cfg['phase'] = 'test'  # Default initialization suffices for shape/memory profiling.
    net = define_G(cfg).cuda().train()
    net.set_new_noise_schedule(cfg['model']['beta_schedule']['train'], 'cuda')
    net.loss_func = MaskedLoss(net.loss_type)
    trainable = sum(p.numel() for p in net.denoise_fn.parameters())
    frozen = sum(p.numel() for p in net.prithvi_encoder.parameters())
    assert not any(p.requires_grad for p in net.prithvi_encoder.parameters())
    ema = copy.deepcopy(net.denoise_fn).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(net.denoise_fn.parameters(), lr=cfg['train']['optimizer']['lr'])
    net.denoise_fn = DDP(net.denoise_fn, device_ids=[rank], output_device=rank,
        broadcast_buffers=False, gradient_as_bucket_view=True)
    torch.manual_seed(20260923 + rank)
    # Dense shape-equivalent random inputs, not a quality or throughput experiment.
    batch = {'HR': torch.rand(16, 1, 224, 224, device=rank)*2-1,
             'SR': torch.rand(16, 1, 224, 224, device=rank)*2-1,
             'HLS': torch.rand(16, 6, 224, 224, device=rank)}
    net.loss_func.mask = torch.ones(16, 1, 224, 224, device=rank, dtype=torch.bool)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    started = time.perf_counter()
    cold = None
    for step in range(5):
        if step == 2:
            cold = {'allocated_bytes': torch.cuda.max_memory_allocated(),
                    'reserved_bytes': torch.cuda.max_memory_reserved()}
            torch.cuda.reset_peak_memory_stats()
        for micro in range(2):
            sync = net.denoise_fn.no_sync() if micro == 0 else contextlib.nullcontext()
            with sync:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = net(batch)
                assert torch.isfinite(loss).item()
                (loss*16).backward()
        grad = apply_update(net, optimizer, 32)
        with torch.no_grad():
            for e, p in zip(ema.parameters(), net.denoise_fn.module.parameters()):
                e.lerp_(p, .0001)
        torch.cuda.synchronize()
        rows.append({'step': step+1, 'loss': loss.item(), 'grad_norm': grad,
                     'allocated_bytes': torch.cuda.max_memory_allocated(),
                     'reserved_bytes': torch.cuda.max_memory_reserved()})
    result = {'variant': variant, 'inner_channel': width, 'rank': rank,
        'channels': [width*m for m in [1,2,4,8,16]],
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'unet_parameters': trainable,
        'encoder_parameters': frozen, 'world_size': 2, 'micro_batch':16,
        'gradient_accumulation':2, 'effective_batch':64, 'precision':'bf16 autocast; fp32 weights',
        'ema':'fp32 GPU copy', 'optimizer':'Adam, fp32 moments',
        'cold_peak': cold, 'steady_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'steady_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'seconds':time.perf_counter()-started, 'steps':rows,
        'scope':'memory only; dense random shape-equivalent inputs; 2 warmup + 3 measured updates; no checkpoints',
        'config':cfg,
        'source_sha256': {str(p.relative_to(here.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), here/'epoch1_train.py', here.parent/'efdiff/upstream/model/unet.py',
                      here.parent/'efdiff/upstream/model/diffusion.py']}}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out/f'{variant}_{width}_rank{rank}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('config','source_sha256','steps')}), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['x0','epsilon'], required=True)
    parser.add_argument('--width', type=int, choices=[32,64], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.variant, args.width, args.output)
