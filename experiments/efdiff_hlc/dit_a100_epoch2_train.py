"""Fixed-center pixel DiT flow/epsilon MSE trainer; independent of frozen UNet runs."""
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
from data import HERE, UPSTREAM
from center_data import HLCPairs, EpochSampler, check_stats
from center_data import HLCPairs as ValidationPairs
from dit_full_validation import evaluate, EvaluationPending
from epoch1_train import save_json, move, apply_update
from dit_model import build

PLAN = HERE / 'dit_a100_epoch2_plan.json'


def tensor_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def milestones(records):
    # Align intermediate validation to batch boundaries, preserve the final tail.
    return sorted({min(records, math.ceil(records * f / 64) * 64)
                   for f in (.25, .5, .75, 1.)})


def load_plan(smoke=False):
    plan = json.loads(PLAN.read_text())
    if not smoke:
        assert plan['budget_confirmed'] and plan['epochs'] == 2, 'Budget not confirmed'
    return plan



def run(output, variant, smoke=False, stop_after=None, root='/data', commit=lambda: None):
    plan = load_plan(smoke)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    done = output / 'complete.json'
    if done.exists():
        return json.loads(done.read_text())
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    random.seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.manual_seed(plan['seed'])
    torch.cuda.manual_seed_all(plan['seed'])
    stats = json.loads(Path(plan['normalization_path']).read_text())
    ds = HLCPairs(root, stats=stats, augment=False, seed=plan['seed'])
    val = ValidationPairs(root, split='val', stats=stats, augment=False, seed=plan['seed'])
    check_stats(stats, ds)
    assert len(ds) == plan['full_train_records'] and len(val) == plan['validation_records']
    assert not set(map(int, ds.lst.indices[:, 0])) & set(map(int, val.lst.indices[:, 0]))
    sources = [PLAN, Path(__file__), HERE/'dit_model.py', HERE/'dit_objectives.py', HERE/'center_data.py', HERE/'center_crop_protocol.json', HERE/'vscore_diffusion.py', HERE/'data.py', HERE/'pilot_data.py',
               HERE/'epoch1_train.py', HERE/'dit_full_validation.py']
    sources += sorted((UPSTREAM/'model').glob('*.py')) + sorted((UPSTREAM/'util').glob('*.py'))
    hashes = {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    stats_hash = hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()
    epochs, per_epoch = (1, 132) if smoke else (plan['epochs'], len(ds))
    total_steps = epochs * math.ceil(per_epoch / 128)
    checkpoint = output / 'latest.pt'
    net = build(variant, initialize=not checkpoint.exists(), architecture=plan['architecture']).train()
    optimizer = torch.optim.Adam(net.denoise_fn.parameters(), lr=plan['learning_rate'])
    ema = copy.deepcopy(net.denoise_fn).eval().requires_grad_(False)
    epoch = offset = step = presented = 0
    best = None
    previous_seconds = 0.
    provenance_path = output / 'provenance.json'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['hashes'] == hashes and state['stats_hash'] == stats_hash
        assert state['variant'] == variant and state['smoke'] == smoke and state['fingerprint'] == ds.fingerprint
        net.denoise_fn.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        ema.load_state_dict(state['ema'])
        epoch, offset, step, presented = (state[k] for k in ('epoch','offset','step','presented'))
        best, previous_seconds = state['best'], state['elapsed_seconds']
        total_steps = state['schedule_total_steps']
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        del state
    else:
        order_hashes = {}
        for ep in range(epochs):
            order = EpochSampler(ds, ep, limit=per_epoch).order.astype('<i4')
            assert len(np.unique(order)) == per_epoch
            order_hashes[str(ep)] = hashlib.sha256(order.tobytes()).hexdigest()
        provenance = dict(plan=plan, variant=variant, smoke=smoke, hashes=hashes,
                          stats_hash=stats_hash, normalization=stats, fingerprint=ds.fingerprint,
                          initial_dit_sha256=tensor_hash(net.denoise_fn.state_dict()),
                          frozen_encoder_sha256=tensor_hash(net.prithvi_encoder.state_dict()),
                          train_order_sha256=order_hashes, validation_records=len(val),
                          dit_parameters=sum(p.numel() for p in net.denoise_fn.parameters()),
                          gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda)
        save_json(provenance_path, provenance)
        for p in sources:
            dest = output / 'source' / p.relative_to(HERE.parent)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    # Use an identical dedicated training stream after initialization/loading.
    if not checkpoint.exists():
        torch.manual_seed(plan['seed'] + 1)
        torch.cuda.manual_seed_all(plan['seed'] + 1)
    start = time.time()
    torch.cuda.reset_peak_memory_stats()

    def save():
        payload = dict(model=net.denoise_fn.state_dict(), optimizer=optimizer.state_dict(), ema=ema.state_dict(),
                       epoch=epoch, offset=offset, step=step, presented=presented, best=best,
                       cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                       hashes=hashes, stats_hash=stats_hash, fingerprint=ds.fingerprint,
                       schedule_total_steps=total_steps, variant=variant, smoke=smoke, elapsed_seconds=previous_seconds+time.time()-start)
        tmp = output / 'latest.tmp'
        torch.save(payload, tmp)
        tmp.replace(checkpoint)
        commit()

    def validate_stage():
        nonlocal best
        path = output / f'validation_{presented}.json'
        if path.exists():
            return
        # Save current training state before validation; restart can repeat it safely.
        save()
        live = net.denoise_fn
        net.denoise_fn = ema
        metrics = evaluate(net, val, stats, 50, limit=len(val), capture_path=output/f'validation_{presented}.npz',
                           progress_path=output/f'validation_{presented}.progress.pt', deadline=start+plan['segment_seconds'], commit=commit)
        net.denoise_fn = live
        metrics.update(presented=presented, step=step, epoch_fraction=presented/per_epoch)
        if best is None or metrics['rmse_c'] < best['rmse_c']:
            best = dict(metrics)
            tmp = output / 'best_ema.tmp'
            torch.save(dict(ema=ema.state_dict(), metrics=metrics, variant=variant, stats=stats), tmp)
            tmp.replace(output / 'best_ema.pt')
        # Commit checkpoint BEFORE marker, so a restart cannot skip best-state update.
        save()
        save_json(path, metrics)
        commit()

    while epoch < epochs:
        stage_offsets = [per_epoch]
        if offset in stage_offsets:
            validate_stage()
        if offset < per_epoch:
            loader = DataLoader(ds, sampler=EpochSampler(ds, epoch, start=offset, limit=per_epoch),
                                batch_size=128, num_workers=16, prefetch_factor=2, pin_memory=True,
                                generator=torch.Generator().manual_seed(plan['seed'] + epoch))
            optimizer.zero_grad(set_to_none=True)
            last_end = time.perf_counter()
            for raw in loader:
                data_wait = time.perf_counter() - last_end
                b = move(raw)
                n = len(raw['index'])
                torch.cuda.synchronize()
                update_start = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = net(b)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                (loss * n).backward()
                warmup = plan['warmup_records']
                progress = presented+n
                factor = progress/warmup if progress < warmup else max(
                    .5*(1+math.cos(math.pi*(progress-warmup)/max(epochs*per_epoch-warmup,1))),
                    plan['min_learning_rate']/plan['learning_rate'])
                for group in optimizer.param_groups:
                    group['lr'] = plan['learning_rate'] * factor
                grad = apply_update(net, optimizer, n)
                step += 1
                offset += n
                presented += n
                with torch.no_grad():
                    decay = plan['ema_decay']**(n/128) if presented >= plan['ema_start_records'] else 0.
                    for e, p in zip(ema.parameters(), net.denoise_fn.parameters()):
                        e.lerp_(p, 1-decay)
                torch.cuda.synchronize()
                update_seconds = time.perf_counter()-update_start
                status = dict(phase='training', variant=variant, epoch=epoch+1, offset=offset,
                              step=step, presented=presented, target_records=epochs*per_epoch,
                              update_records=n, loss=loss.item(), grad_norm=grad,
                              lr=optimizer.param_groups[0]['lr'], data_wait_seconds=data_wait,
                              update_seconds=update_seconds, elapsed_seconds=previous_seconds+time.time()-start,
                              peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9,
                              peak_reserved_GB=torch.cuda.max_memory_reserved()/1e9)
                if step % 50 == 0 or smoke or offset == per_epoch:
                    print(json.dumps(status), flush=True)
                    save_json(output/'status.json',status)
                    with (output/'metrics.jsonl').open('a') as f:
                        f.write(json.dumps(status)+'\n')
                    commit()
                segment_stop = time.time()-start >= plan['segment_seconds'] or (stop_after is not None and step >= stop_after)
                if step % plan['checkpoint_every'] == 0 or offset in stage_offsets or segment_stop:
                    save()
                if segment_stop and offset < per_epoch:
                    result = dict(status,phase='checkpointed')
                    save_json(output/'status.json',result)
                    commit()
                    return result
                if offset in stage_offsets:
                    validate_stage()
                last_end = time.perf_counter()
            del loader, b, raw, loss
        assert offset == per_epoch
        epoch += 1
        offset = 0
        save()
    assert presented == epochs*per_epoch and step == total_steps
    torch.save(dict(ema=ema.state_dict(),variant=variant,stats=stats,presented=presented,step=step), output/'final_ema.pt')
    sweep = {}
    live = net.denoise_fn
    net.denoise_fn = ema
    for selection in ('final', 'best'):
        if selection == 'best':
            selected = torch.load(output/'best_ema.pt',map_location='cpu',weights_only=False)
            ema.load_state_dict(selected['ema'])
            selected_records = selected['metrics']['presented']
            del selected
        else:
            selected_records = presented
        for nfe in plan['evaluation_nfe']:
            path = output/f'{selection}_nfe{nfe}.json'
            if path.exists():
                metrics = json.loads(path.read_text())
            else:
                metrics = evaluate(net,val,stats,nfe,limit=len(val), capture_path=output/f'{selection}_nfe{nfe}.npz',
                                   progress_path=(output/f'validation_{selected_records}.progress.pt' if nfe==50 else output/f'eval_{selected_records}_nfe{nfe}.progress.pt'),
                                   deadline=start+plan['segment_seconds'], commit=commit)
                metrics.update(selection=selection,selected_training_records=selected_records)
                save_json(path, metrics)
                commit()
            sweep[f'{selection}_{nfe}'] = metrics
    net.denoise_fn = live
    result = dict(phase='smoke_passed' if smoke else 'complete',variant=variant,epochs=epochs,
                  training_records=presented,updates=step,tail_records=per_epoch%128,
                  best=best,sweep=sweep,test_used=False,elapsed_seconds=previous_seconds+time.time()-start)
    save_json(done,result)
    save_json(output/'status.json',result)
    commit()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--variant', choices=['flow','epsilon'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    import modal
    volume=modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
    try:
        result=run(args.output,args.variant,args.smoke,args.stop_after,commit=volume.commit)
    except EvaluationPending as exc:
        result=dict(phase='checkpointed',variant=args.variant,reason='full_validation_segment',progress=str(exc))
    save_json(Path(args.output)/'segment_result.json',result)
    volume.commit()
