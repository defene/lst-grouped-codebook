"""Bounded performance probe; never contributes checkpoints to the experiment."""
import modal
from modal_app import image, mounts, runs_volume

app = modal.App('efdiff-hlc-deadline-benchmark-20260916')


def run_benchmark(micros=(16, 32, 64)):
    import copy
    import json
    import time
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    from data import HLCPairs, EpochSampler
    from epoch1_train import build_model, move, apply_update, save_json

    torch.set_num_threads(8)
    torch.manual_seed(20260915)
    torch.backends.cudnn.benchmark = True
    stats = json.loads(Path('/runs/hlc-synthetic-pilot-16384-512-v1/normalization.json').read_text())
    ds = HLCPairs('/data', stats=stats, augment=True)
    net, opt = build_model('epsilon')
    net.train()
    optimizer = torch.optim.Adam(net.denoise_fn.parameters(), lr=2e-5)
    ema = copy.deepcopy(net.denoise_fn).eval().requires_grad_(False)
    results = []
    for micro in micros:
        loader = DataLoader(ds, sampler=EpochSampler(ds, 0, limit=64*16),
                            batch_size=micro, num_workers=8, pin_memory=True,
                            generator=torch.Generator().manual_seed(20260915))
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pending = step = 0
        times = []
        torch.cuda.synchronize()
        tick = time.perf_counter()
        for raw in loader:
            b = move(raw)
            n = len(raw['index'])
            net.loss_func.mask = b['mask']
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = net(b)
            assert torch.isfinite(loss)
            (loss*n).backward()
            pending += n
            if pending == 64:
                apply_update(net, optimizer, pending)
                with torch.no_grad():
                    for e, p in zip(ema.parameters(), net.denoise_fn.parameters()):
                        e.lerp_(p, .0001)
                torch.cuda.synchronize()
                now = time.perf_counter()
                times.append(now-tick)
                tick = now
                pending = 0
                step += 1
        measured = times[4:]
        mean = sum(measured)/len(measured)
        result = {'gpu': torch.cuda.get_device_name(), 'micro_batch': micro,
                  'effective_batch': 64, 'variant': 'epsilon',
                  'warmup_updates': 4, 'measured_updates': len(measured),
                  'seconds_per_update': mean, 'update_seconds': measured,
                  'projected_train_hours': mean*40763/3600,
                  'peak_allocated_GB': torch.cuda.max_memory_allocated()/1e9,
                  'peak_reserved_GB': torch.cuda.max_memory_reserved()/1e9,
                  'scope': 'performance only; pilot train statistics; no experiment checkpoint'}
        results.append(result)
        print(json.dumps(result), flush=True)
        del loader, b, loss
    out = Path('/runs/hlc-synthetic-full-epoch1-20260916-v1/deadline')
    out.mkdir(exist_ok=True)
    name = 'b200' if 'B200' in torch.cuda.get_device_name() else 'h200'
    save_json(out/f'{name}_microbatch_benchmark.json', results)
    runs_volume.commit()
    return results


@app.function(image=image, volumes=mounts, gpu='H200', cpu=12, memory=65536,
              timeout=1800, max_containers=1, single_use_containers=True)
def benchmark():
    return run_benchmark()


@app.function(image=image, volumes=mounts, gpu='B200', cpu=12, memory=65536,
              timeout=1800, max_containers=1, single_use_containers=True)
def benchmark_b200():
    return run_benchmark((32, 64))


@app.local_entrypoint()
def main(gpu: str = 'H200'):
    import json
    from pathlib import Path
    assert gpu in ('H200', 'B200')
    result = (benchmark if gpu == 'H200' else benchmark_b200).remote()
    out = Path('output/efdiff_hlc_full_epoch1_20260916')
    out.mkdir(parents=True, exist_ok=True)
    (out/f'{gpu.lower()}_microbatch_benchmark.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
