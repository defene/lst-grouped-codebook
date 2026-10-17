"""Two-GPU exact-epoch HLC runner, versioned for the 24-hour requirement.

Global batches preserve the original order; two equal rank batches (including
the four-record tail) make DDP's mean reduction equal to a global record mean.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

from data import HERE, UPSTREAM, HLCPairs, EpochSampler, check_stats, PROTOCOL
from pilot_data import HLCPairs as ValidationPairs
from epoch1_train import build_model, move, apply_update, evaluate, save_json


def config(variant):
    return yaml.safe_load((HERE/'configs'/f'deadline_fast_{variant}.yaml').read_text())


class RankSampler(Sampler):
    def __init__(self, dataset, rank, world=2, start=0, limit=None, effective=64):
        self.order = EpochSampler(dataset, 0, start=start, limit=limit).order
        # EpochSampler.order is the COMPLETE permutation, independent of start.
        end = len(dataset) if limit is None else limit
        assert world == 2 and effective == 64 and start % effective == 0
        assert (end-start) % world == 0
        parts = []
        for a in range(start, end, effective):
            b = min(a+effective, end)
            local = (b-a)//world
            parts.append(self.order[a+rank*local:a+(rank+1)*local])
        self.indices = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def __iter__(self):
        return ((0, int(i)) for i in self.indices)

    def __len__(self):
        return len(self.indices)


def run(args):
    rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    assert world == 2
    torch.cuda.set_device(rank)
    from datetime import timedelta
    dist.init_process_group('nccl', timeout=timedelta(minutes=40))
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    random.seed(20260915)
    np.random.seed(20260915)
    torch.manual_seed(20260915)
    torch.cuda.manual_seed_all(20260915)
    if args.smoke:
        # Verify NCCL/DDP averaging on the exact 2+2-record final-batch shape.
        toy = torch.nn.Linear(1,1,bias=False).cuda()
        toy.weight.data.fill_(.1)
        wrapped = DDP(toy, device_ids=[rank])
        all_x = torch.tensor([[.1],[.2],[.3],[.4]], device=rank)
        local_x = all_x[rank*2:(rank+1)*2]
        wrapped(local_x).square().mean().backward()
        reference = (2*.1*all_x.square().mean()).reshape_as(toy.weight)
        torch.testing.assert_close(toy.weight.grad, reference)
        del wrapped, toy, all_x, local_x, reference
    opt = config(args.variant)
    micro = opt['train']['micro_batch']
    assert micro == 16 and opt['train']['effective_batch'] == 64
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stats = json.loads(Path(args.stats).read_text())
    ds = HLCPairs('/data', stats=stats, augment=True)
    val = ValidationPairs('/data', split='val', stats=stats)
    if not args.smoke and not args.probe:
        check_stats(stats, ds)
    per_epoch = opt['smoke']['records'] if args.smoke else len(ds)
    assert per_epoch % world == 0
    total_steps = math.ceil(per_epoch/64)
    budget = json.loads((HERE/'deadline_fast_budget.json').read_text())
    assert len(ds) == budget['train_records'] == 2608772
    assert budget['epochs'] == opt['train']['epochs'] == 1
    assert math.ceil(len(ds)/64) == budget['updates'] == 40763
    sources = [HERE/p for p in ('data.py', 'pilot_data.py', 'epoch1_train.py',
        'epoch1_stats.py', 'fit_stats.py', 'deadline_fast_train.py', 'deadline_fast_budget.json',
        f'configs/deadline_fast_{args.variant}.yaml', f'configs/epoch1_{args.variant}.yaml',
        'pilot_membership/summary.json', 'pilot_membership/val.npy')]
    sources += sorted((UPSTREAM/'model').glob('*.py')) + sorted((UPSTREAM/'util').glob('*.py'))
    hashes = {str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    stats_hash = hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()
    net, base_opt = build_model(args.variant)
    assert opt['model'] == base_opt['model']
    net.train()
    optimizer = torch.optim.Adam(net.denoise_fn.parameters(), lr=opt['train']['optimizer']['lr'])
    ema = copy.deepcopy(net.denoise_fn).eval().requires_grad_(False)
    checkpoint = output/'latest.pt'
    offset = step = epoch_done = 0
    if args.probe:
        offset, step = 64000, 1000
    previous_seconds = 0.0
    best = math.inf
    state = None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['code_sha256'] == hashes and state['stats_sha256'] == stats_hash
        assert state['data_fingerprint'] == ds.fingerprint and state['world_size'] == world
        assert state['variant'] == args.variant and state['smoke'] == args.smoke
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        ema.load_state_dict(state['ema'])
        offset, step, epoch_done, best = (state[k] for k in ('offset','step','epoch','best'))
        previous_seconds = state['accumulated_seconds']
    net.denoise_fn = DDP(net.denoise_fn, device_ids=[rank], output_device=rank,
                          broadcast_buffers=False, gradient_as_bucket_view=True)
    if state is not None:
        torch.set_rng_state(state['rank_rng'][rank]['cpu'])
        torch.cuda.set_rng_state(state['rank_rng'][rank]['cuda'], device=rank)
        del state
    else:
        # Independent rank noise streams; shared initialization and sample order.
        torch.manual_seed(20260915+rank)
        torch.cuda.manual_seed(20260915+rank)
    if rank == 0:
        import modal
        volume = modal.Volume.from_name('efdiff-hlc-synthetic-runs', environment_name='main')
        provenance = {'protocol':PROTOCOL, 'variant':args.variant, 'smoke':args.smoke,
            'world_size':world, 'gpu':torch.cuda.get_device_name(), 'model_config':opt,
            'records_per_epoch':per_epoch, 'epochs':1, 'updates':total_steps,
            'effective_batch':64, 'micro_batch_per_gpu':micro,
            'code_sha256':hashes, 'stats_sha256':stats_hash, 'normalization':stats,
            'data_fingerprint':ds.fingerprint, 'torch':torch.__version__, 'cuda':torch.version.cuda,
            'performance_probe':args.probe,
            'noise_rule':'seed 20260915 + rank; differs from earlier single-GPU pilot',
            'selection':'fixed epoch1 EMA, identical fixed 512 validation members, no test'}
        save_json(output/'provenance.json', provenance)
        for p in sources:
            dst = output/'source'/p.relative_to(HERE.parent)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
        volume.commit()
    dist.barrier()
    started = time.time()

    def save(epoch, position):
        rng = {'cpu':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state(rank).cpu()}
        collected = [None]*world if rank == 0 else None
        dist.gather_object(rng, collected, dst=0)
        if rank == 0:
            weights = {k.replace('denoise_fn.module.','denoise_fn.',1):v for k,v in net.state_dict().items()}
            state = {'model':weights, 'optimizer':optimizer.state_dict(), 'ema':ema.state_dict(),
                'epoch':epoch, 'offset':position, 'step':step, 'presented':per_epoch if epoch else position,
                'best':best, 'rank_rng':collected, 'world_size':world, 'variant':args.variant,
                'smoke':args.smoke, 'code_sha256':hashes, 'stats_sha256':stats_hash,
                'data_fingerprint':ds.fingerprint,
                'accumulated_seconds':previous_seconds+time.time()-started}
            temp = output/'latest.tmp'
            torch.save(state, temp)
            temp.replace(checkpoint)
            volume.commit()
        dist.barrier()

    if not epoch_done and offset < per_epoch:
        sampler = RankSampler(ds, rank, world, start=offset, limit=per_epoch)
        loader = DataLoader(ds, sampler=sampler, batch_size=micro, num_workers=opt['train']['workers'], prefetch_factor=4,
            pin_memory=True, generator=torch.Generator().manual_seed(20260915+rank))
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        loss_sum = 0.0
        torch.cuda.reset_peak_memory_stats()
        for raw in loader:
            b = move(raw)
            n = len(raw['index'])
            expected = min(64, per_epoch-offset)//world
            final_micro = pending+n == expected
            assert pending+n <= expected
            context = contextlib.nullcontext() if final_micro else net.denoise_fn.no_sync()
            with context:
                net.loss_func.mask = b['mask']
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = net(b)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                (loss*n).backward()
            loss_sum += float(loss.detach())*n
            pending += n
            if not final_micro:
                continue
            warmup = opt['train']['warmup_steps']
            base_lr = opt['train']['optimizer']['lr']
            factor = max(step,1)/warmup if step < warmup else max(
                .5*(1+math.cos(math.pi*(step-warmup)/max(total_steps-warmup,1))), 1e-6/base_lr)
            for group in optimizer.param_groups:
                group['lr'] = base_lr*factor
            # DDP averages sums across ranks; dividing by equal local count
            # yields the exact global sample mean, including the 2+2 tail.
            grad = apply_update(net, optimizer, pending)
            step += 1
            offset += pending*world
            with torch.no_grad():
                decay = opt['train']['ema_decay'] if step >= opt['train']['ema_start_step'] else 0.0
                for e,p in zip(ema.parameters(), net.denoise_fn.module.parameters()):
                    e.lerp_(p, 1-decay)
            sums = torch.tensor([loss_sum], device=rank, dtype=torch.float64)
            dist.all_reduce(sums)
            record = {'epoch':1, 'step':step, 'epoch_records':offset, 'presented':offset,
                'update_records':pending*world, 'loss':sums.item()/(pending*world), 'grad_norm':grad,
                'elapsed_seconds':previous_seconds+time.time()-started,
                'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9,
                'peak_reserved_gb':torch.cuda.max_memory_reserved()/1e9, 'world_size':world}
            if rank == 0 and (step % (10 if args.probe else opt['train']['log_every']) == 0 or args.smoke or offset == per_epoch):
                print(json.dumps(record), flush=True)
                with (output/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                save_json(output/'status.json', dict(record, phase='training'))
            pending = 0
            loss_sum = 0.0
            stop = torch.tensor([int(rank == 0 and (
                (args.stop_after and step >= args.stop_after) or
                (not args.smoke and time.time()-started > opt['train']['segment_seconds'])))], device=rank)
            dist.broadcast(stop, src=0)
            if step % opt['train']['checkpoint_every'] == 0 or stop.item():
                save(0, offset)
            if stop.item():
                if rank == 0:
                    save_json(output/'segment_result.json', {'phase':'checkpointed','updates':step,
                        'record_presentations':offset})
                    volume.commit()
                dist.barrier()
                dist.destroy_process_group()
                return
        assert pending == 0 and offset == per_epoch
        save(0, offset)
    assert step == total_steps
    if rank == 0 and not epoch_done:
        # Evaluation is single-rank with EMA, no distributed collectives.
        live = net.denoise_fn
        net.denoise_fn = ema
        metrics = evaluate(net, val, stats, micro, 8,
            limit=opt['smoke']['validation_records'] if args.smoke else 512,
            capture_path=output/'predictions_epoch1.npz' if not args.smoke else None)
        net.denoise_fn = live
        metrics.update(epoch=1, step=step, split='val', smoke=args.smoke)
        best = metrics['rmse_c']
        save_json(output/'validation_epoch_1.json', metrics)
        torch.save({'ema':ema.state_dict(), 'epoch':1, 'metrics':metrics,
            'stats':stats, 'variant':args.variant}, output/'best_ema.pt')
    dist.barrier()
    save(1, 0)
    if rank == 0:
        result = {'phase':'smoke_passed' if args.smoke else 'training_complete',
            'updates':step, 'record_presentations':per_epoch, 'epochs':1,
            'best_val_rmse_c':best, 'world_size':world,
            'elapsed_seconds':previous_seconds+time.time()-started, 'test_used':False}
        save_json(output/'status.json', result)
        save_json(output/'segment_result.json', result)
        volume.commit()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['x0','epsilon'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stats', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--stop-after', type=int, default=0)
    run(parser.parse_args())
