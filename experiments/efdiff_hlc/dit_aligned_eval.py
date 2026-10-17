"""Read-only checkpoint evaluation on the frozen historical UNet validation cohort."""
import gc
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from center_data import HLCPairs
from data import HERE
from dit_model import build
from dit_train import evaluate, tensor_hash

RUN = 'hlc-dit-aligned-val512-20261002-v1'
SOURCE = 'hlc-dit-a100-b128-fullval-epoch2-20261001-v1'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()

def run(commit):
    out = Path('/runs') / RUN
    out.mkdir(exist_ok=True)
    def save(name, obj):
        (out/name).write_text(json.dumps(obj, indent=2))
        commit()
    if (out/'complete.json').exists(): return json.loads((out/'complete.json').read_text())
    torch.set_num_threads(8)
    source = Path('/runs') / SOURCE / 'train'
    provenance = json.loads((source/'flow/provenance.json').read_text())
    for name, digest in provenance['hashes'].items():
        assert sha(HERE.parent/name) == digest, name
    stats = provenance['normalization']
    memberfile = HERE/'pilot_membership/val.npy'
    summary = json.loads((HERE/'pilot_membership/summary.json').read_text())['splits']['val']
    assert sha(memberfile) == summary['membership_sha256']
    assert sha('/data/processed_data/v1/lstsr_tb/indices/val.lst_hr.npy') == summary['source_index_sha256']
    members = np.load(memberfile)
    assert len(members) == len(np.unique(members)) == 512
    full = HLCPairs('/data', split='val', stats=stats, augment=False)
    assert len(full) == 141711
    val = Subset(full, members.tolist())
    reference = np.load('/code/alignment_reference.npz')
    capture_positions = np.linspace(0, 511, 16, dtype=int).tolist()
    baseline_sse = baseline_sae = pixels = count = 0
    for batch in DataLoader(val, batch_size=16, num_workers=8):
        errors = (batch['condition_c']-batch['target_c'])[batch['mask'].bool()].double()
        baseline_sse += errors.square().sum().item()
        baseline_sae += errors.abs().sum().item()
        pixels += errors.numel()
        for i in range(len(batch['index'])):
            if count+i in capture_positions:
                j = capture_positions.index(count+i)
                for key in ('index','record_id','month','target_c','condition_c','mask'):
                    assert np.array_equal(batch[key][i].numpy(), reference[key][j], equal_nan=True), (key,j)
        count += len(batch['index'])
    assert count == 512 and pixels == 25549027
    assert math.isclose(baseline_sse, 69653407.97920328, rel_tol=1e-8)
    audit = dict(run=RUN, source_run=SOURCE, records=count, valid_pixels=pixels,
                 membership_sha256=sha(memberfile), reference_sha256=sha('/code/alignment_reference.npz'),
                 exact_saved_reference_match=True, baseline_rmse_c=math.sqrt(baseline_sse/pixels),
                 baseline_mae_c=baseline_sae/pixels, gpu=torch.cuda.get_device_name(),
                 gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), inference_batch=16, seed=20260927,
                 training_performed=False, test_used=False, checkpoints={}, results={})
    save('alignment_audit.json', audit)
    for variant in ('flow','epsilon'):
        net = build(variant, architecture=provenance['plan']['architecture']).eval().requires_grad_(False)
        vp = json.loads((source/variant/'provenance.json').read_text())
        assert tensor_hash(net.prithvi_encoder.state_dict()) == vp['frozen_encoder_sha256']
        seen = {}
        for weight in ('best','final'):
            path = source/variant/f'{weight}_ema.pt'
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert state['stats'] == stats and state['variant'] == variant
            digest = tensor_hash(state['ema'])
            key = f'{variant}_{weight}'
            meta = dict(file_sha256=sha(path), weights_sha256=digest,
                        presented=state.get('presented',state.get('metrics',{}).get('presented')),
                        source=str(path), selection='Original full-val 50 NFE best; no reselection on 512')
            audit['checkpoints'][key] = meta
            net.denoise_fn.load_state_dict(state['ema'])
            del state
            for nfe in (50,20,10,5):
                tag = f'{key}_nfe{nfe}'
                save('status.json', dict(stage='evaluating', tag=tag, time=time.time(), gpu_uuid=audit['gpu_uuid']))
                if digest in seen:
                    previous = f'{seen[digest]}_nfe{nfe}'
                    result = dict(audit['results'][previous], reused_from=previous)
                    shutil.copyfile(out/f'{previous}.npz', out/f'{tag}.npz')
                else:
                    result = evaluate(net, val, stats, nfe, capture_path=out/f'{tag}.npz')
                result.update(variant=variant, weights=weight, presented=meta['presented'], source_checkpoint_sha256=meta['file_sha256'])
                assert result['valid_pixels'] == pixels and result['records'] == 512
                saved = np.load(out/f'{tag}.npz')
                for field in ('index','record_id','month','target_c','condition_c','mask'):
                    assert np.array_equal(saved[field], reference[field], equal_nan=True), (tag,field)
                audit['results'][tag] = result
                save(f'{tag}.json', result)
                save('alignment_audit.json', audit)
                print(tag, result['rmse_c'], result['mae_c'], flush=True)
            seen[digest] = key
        del net
        gc.collect()
        torch.cuda.empty_cache()
    save('complete.json', dict(status='complete', **audit))
    return dict(status='complete', run=RUN, evaluations=len(audit['results']))
