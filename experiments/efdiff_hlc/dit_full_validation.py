"""Resumable full-validation reduction, with a fixed dedicated inference RNG."""
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from epoch1_train import move, save_json


class EvaluationPending(Exception):
    pass


@torch.no_grad()
def evaluate(net, val, stats, steps, limit, capture_path, progress_path, deadline, commit):
    progress_path = Path(progress_path)
    keys = ('target_c', 'condition_c', 'mask', 'prediction_c', 'index', 'record_id', 'month')
    rng = torch.Generator(device='cuda').manual_seed(20260927)
    state = dict(records=0, pixels=0, squared=0., absolute=0., seconds=0., captures={k: [] for k in keys})
    if progress_path.exists():
        state = torch.load(progress_path, map_location='cpu', weights_only=False)
        assert state['nfe'] == steps and state['limit'] == limit and state['fingerprint'] == val.fingerprint
        rng.set_state(state['rng'])
    capture_ids = set(np.linspace(0, limit-1, min(16, limit), dtype=int).tolist())
    loader = DataLoader(val, sampler=range(state['records'], limit), batch_size=128,
                        num_workers=16, pin_memory=True,
                        generator=torch.Generator().manual_seed(20260927))
    net.eval()
    with torch.random.fork_rng(devices=[0]):
        for raw in loader:
            b = move(raw)
            noise = torch.randn(b['HR'].shape, device='cuda', generator=rng)
            torch.cuda.synchronize(); start = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                residual = net.sample(b, steps, noise)
            torch.cuda.synchronize(); state['seconds'] += time.perf_counter()-start
            pred = b['condition_c'] + residual.float()*(3*stats['residual']['std']) + stats['residual']['mean']
            errors = (pred-b['target_c'])[b['mask'].bool()].double()
            assert errors.numel() and torch.isfinite(errors).all()
            state['squared'] += errors.square().sum().item()
            state['absolute'] += errors.abs().sum().item()
            state['pixels'] += errors.numel()
            for i in range(len(raw['index'])):
                if state['records']+i in capture_ids:
                    for key in keys:
                        value = pred[i] if key == 'prediction_c' else raw[key][i]
                        state['captures'][key].append(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
            state['records'] += len(raw['index'])
            if state['records'] % 2048 == 0 or state['records'] == limit or time.time() >= deadline:
                state.update(rng=rng.get_state(), nfe=steps, limit=limit, fingerprint=val.fingerprint)
                tmp = progress_path.with_suffix('.tmp'); torch.save(state, tmp); tmp.replace(progress_path)
                save_json(progress_path.with_suffix('.json'), dict(phase='validation', records=state['records'],
                          target_records=limit, nfe=steps, unix=time.time(), progress_file=str(progress_path)))
                commit()
                if time.time() >= deadline and state['records'] < limit:
                    raise EvaluationPending(str(progress_path))
    assert state['records'] == limit and state['pixels'] > 0
    result = dict(records=limit, valid_pixels=state['pixels'], rmse_c=math.sqrt(state['squared']/state['pixels']),
                  mae_c=state['absolute']/state['pixels'], squared_error_sum=state['squared'],
                  absolute_error_sum=state['absolute'], nfe=steps,
                  sampling_seconds_including_encoder=state['seconds'], split='val', test_used=False,
                  validation_fingerprint=val.fingerprint, inference_batch_size=128)
    assert math.isfinite(result['rmse_c']) and math.isfinite(result['mae_c'])
    if capture_path:
        np.savez_compressed(capture_path, **{k: np.asarray(v) for k,v in state['captures'].items()})
    net.train()
    return result
