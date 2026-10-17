"""Explicit audited migration of the completed batch128 epoch to batch256."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import torch
from epoch1_train import save_json
from raw_mmdit_data import HLCPairs, EpochSampler, check_stats


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def state_sha(value):
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            h.update(str((v.dtype, tuple(v.shape))).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); add(v[k])
        elif isinstance(v, (list, tuple)):
            for item in v: add(item)
        else:
            h.update(repr(v).encode())
    add(value)
    return h.hexdigest()


def migrate(root):
    here = Path(__file__).resolve().parent
    plan = json.loads((here/'raw_mmdit_b256_plan.json').read_text())
    source = Path('/runs')/plan['parent_run']
    old = source/'train/v'
    out = root/'train/v'
    assert not out.exists(), 'Never overwrite an existing migration'
    state = torch.load(old/'latest.pt', map_location='cpu', weights_only=False)
    assert (state['epoch'], state['offset'], state['step'], state['presented']) == (0,2608772,20382,2608772)
    assert not state['smoke'] and state['variant']=='v'
    old_hashes = state['hashes']
    for relative, expected in old_hashes.items():
        assert sha(here.parent/relative)==expected, relative
    preserved = {k:state_sha(state[k]) for k in ('model','optimizer','ema','cpu_rng','cuda_rng')}
    source_sha = sha(old/'latest.pt')
    provenance=json.loads((old/'provenance.json').read_text())
    preflight=json.loads((source/'preflight.json').read_text())
    shutil.copytree(source/'assets', root/'assets')
    stats=json.loads((root/'assets/normalization.json').read_text())
    for split in ('train','val'):
        ds=HLCPairs('/data',split=split,stats=stats,augment=False,seed=plan['seed'],metadata_dir=root/'assets/metadata')
        assert ds.fingerprint==preflight['fingerprints'][split]
        if split=='train':
            check_stats(stats,ds)
            for ep in range(2):
                order=EpochSampler(ds,ep).order.astype('<i4')
                assert hashlib.sha256(order.tobytes()).hexdigest()==provenance['train_order_sha256'][str(ep)]
    assert state['stats_hash']==hashlib.sha256(json.dumps(stats,sort_keys=True).encode()).hexdigest()
    new_hashes=dict(old_hashes)
    for old_name,new_name in [('raw_mmdit_train.py','raw_mmdit_b256_train.py'),('raw_mmdit_train_plan.json','raw_mmdit_b256_plan.json')]:
        del new_hashes['efdiff_hlc/'+old_name]
        new_hashes['efdiff_hlc/'+new_name]=sha(here/new_name)
    state['hashes']=new_hashes
    state['schedule_total_steps']=state['step']+math.ceil(2608772/256)
    assert state['schedule_total_steps']==plan['updates_per_model']==30573
    out.mkdir(parents=True)
    # Copy completed validation or its accumulator and RNG without changing inference batch128.
    copied=[]
    for file in old.iterdir():
        if file.is_file() and (file.name.startswith(('validation_','best','epoch_')) or file.name=='metrics.jsonl'):
            shutil.copyfile(file,out/file.name)
            assert sha(file)==sha(out/file.name)
            copied.append(dict(name=file.name,sha256=sha(out/file.name),bytes=file.stat().st_size))
    torch.save(state,out/'latest.pt')
    shutil.copyfile(out/'latest.pt',out/'last.pt')
    reloaded=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
    assert preserved=={k:state_sha(reloaded[k]) for k in preserved}
    del reloaded
    migration=dict(parent_run=plan['parent_run'],run_id=plan['run_id'],source_checkpoint_sha256=source_sha,
        migrated_checkpoint_sha256=sha(out/'latest.pt'),preserved_state_sha256=preserved,
        inherited_batch=128,inherited_records=state['presented'],inherited_updates=state['step'],
        batch_size=256,remaining_records=2608772,schedule_total_steps=30573,total_records=5217544,
        epoch_tails=[4,132],copied_validation=copied,member_order_verified=True,
        learning_rate='unchanged sample-based schedule',ema='0.9998000100000001 ** (actual_batch/128)',unix=time.time())
    save_json(root/'resume_migration.json',migration)
    save_json(out/'raw_mmdit_run.json',dict(run_id=plan['run_id'],variant='v',smoke=False))
    save_json(out/'parent_provenance.json',provenance)
    provenance.update(plan=plan,hashes=new_hashes,resume_migration=migration)
    save_json(out/'provenance.json',provenance)
    for relative in new_hashes:
        dest=out/'source'/relative; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(here.parent/relative,dest)
    preflight.update(plan=plan,migration=migration)
    save_json(root/'preflight.json',preflight)
    return migration
