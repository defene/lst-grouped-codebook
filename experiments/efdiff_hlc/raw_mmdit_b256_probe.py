"""Three disposable real-data batch256 updates; never alter resumable model state."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from raw_mmdit_model import build
from raw_mmdit_data import HLCPairs, EpochSampler
from epoch1_train import move, apply_update, save_json


def probe(output, assets):
    output,assets=Path(output),Path(assets)
    plan=json.loads((Path(__file__).parent/'raw_mmdit_b256_plan.json').read_text())
    torch.set_num_threads(8)
    state=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
    net=build('v',device='cuda',architecture=plan['architecture']).train()
    net.denoise_fn.load_state_dict(state['model'])
    optimizer=torch.optim.Adam(net.denoise_fn.parameters(),lr=plan['learning_rate'])
    optimizer.load_state_dict(state['optimizer'])
    # Include the same resident EMA footprint as formal training.
    import copy
    ema=copy.deepcopy(net.denoise_fn).eval().requires_grad_(False)
    ema.load_state_dict(state['ema'])
    torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
    del state
    ds=HLCPairs('/data',stats=json.loads((assets/'normalization.json').read_text()),augment=False,
                seed=plan['seed'],metadata_dir=assets/'metadata')
    loader=DataLoader(ds,sampler=EpochSampler(ds,1,limit=768),batch_size=256,num_workers=8,pin_memory=True)
    torch.cuda.reset_peak_memory_stats()
    rows=[]
    for raw in loader:
        b=move(raw); optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=net(b)
        assert torch.isfinite(loss) and len(raw['index'])==256
        (loss*256).backward()
        grad=apply_update(net,optimizer,256)
        with torch.no_grad():
            for e,p in zip(ema.parameters(),net.denoise_fn.parameters()):
                e.lerp_(p,1-plan['ema_decay']**2)
        torch.cuda.synchronize()
        rows.append(dict(loss=loss.item(),grad_norm=grad,seconds=time.perf_counter()-start))
    assert len(rows)==3
    result=dict(phase='passed',batch_size=256,updates=3,disposable_probe=True,counts_toward_training=False,
        checkpoint_unchanged=True,gpu=torch.cuda.get_device_name(),
        peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9,
        peak_reserved_GB=torch.cuda.max_memory_reserved()/1e9,updates_detail=rows)
    save_json(output/'batch256_probe.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); parser.add_argument('--assets',required=True)
    args=parser.parse_args(); probe(args.output,args.assets)
