"""Audited, one-time checkpoint migration; preserve all training tensors and RNG."""
import hashlib
import json
import math
from pathlib import Path


def source_hashes(here, upstream, variant, continuation):
    trainer = 'width32_epoch1_train.py' if continuation else 'width32_deadline_train.py'
    plan = 'width32_epoch1_plan.json' if continuation else 'width32_12h_plan.json'
    config = f'width32_epoch1_{variant}.json' if continuation else f'width32_single_{variant}.json'
    paths = [here/trainer, here/plan, here/'configs'/config, here/'data.py',
             here/'pilot_data.py', here/'epoch1_train.py', here/'pilot_membership/val.npy']
    paths += sorted((upstream/'model').glob('*.py')) + sorted((upstream/'util').glob('*.py'))
    return {str(p.relative_to(here.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare_run(destination):
    import torch
    from data import HERE, UPSTREAM, HLCPairs, EpochSampler, check_stats
    from epoch1_train import save_json
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    done = destination/'resume_migration.json'
    if done.exists():
        return json.loads(done.read_text())
    plan = json.loads((HERE/'width32_epoch1_plan.json').read_text())
    source = Path('/runs')/plan['source_run']
    stats = json.loads(Path(plan['normalization_path']).read_text())
    train = HLCPairs('/data', stats=stats, augment=True)
    check_stats(stats, train)
    assert len(train) == plan['full_epoch_records'] == 2608772
    order = EpochSampler(train, 0).order
    import numpy as np
    assert len(order) == len(train) and len(np.unique(order)) == len(train)
    assert int(order.min()) == 0 and int(order.max()) == len(train)-1
    prefix_hash = hashlib.sha256(order[:819200].astype('<i4').tobytes()).hexdigest()
    full_hash = hashlib.sha256(order.astype('<i4').tobytes()).hexdigest()
    remaining = list(range(819200, len(train), 64))
    assert len(remaining) == 27963 and len(train)-remaining[-1] == 4
    assert math.ceil(len(train)/64) == 40763
    stat_hash = hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()
    migrated = {}
    for variant in ('x0', 'epsilon'):
        old_cfg = json.loads((HERE/'configs'/f'width32_single_{variant}.json').read_text())
        new_cfg = json.loads((HERE/'configs'/f'width32_epoch1_{variant}.json').read_text())
        for field in ('model', 'data', 'normalization', 'evaluation', 'seed'):
            assert old_cfg[field] == new_cfg[field], field
        changed_train = {k for k in old_cfg['train'] if old_cfg['train'][k] != new_cfg['train'][k]}
        assert changed_train == {'epochs', 'tail_records', 'total_updates', 'record_presentations', 'stage_records'}
        original = source/'train'/variant
        marker = json.loads((original/'stage_819200.json').read_text())
        assert marker['phase'] == 'stage_complete' and marker['training_records'] == 819200
        provenance = json.loads((original/'provenance.json').read_text())
        assert provenance['stage_members_sha256']['819200'] == prefix_hash
        expected = source_hashes(HERE, UPSTREAM, variant, False)
        new_hashes = source_hashes(HERE, UPSTREAM, variant, True)
        state = torch.load(original/'latest.pt', map_location='cpu', weights_only=False)
        assert state['hashes'] == expected, 'Frozen predecessor source mismatch'
        assert state['offset'] == 819200 and state['step'] == 12800
        assert state['variant'] == variant and not state['smoke']
        assert state['fingerprint'] == train.fingerprint and state['stats_hash'] == stat_hash
        assert state['optimizer']['state'] and state['ema'] and state['model']
        target = destination/'train'/variant
        target.mkdir(parents=True, exist_ok=True)
        # Update only provenance hashes after explicitly validating compatibility.
        # Model, optimizer, EMA, elapsed time, offset and all RNG tensors are retained.
        state['hashes'] = new_hashes
        temp = target/'migration.tmp'
        torch.save(state, temp)
        temp.replace(target/'latest.pt')
        migrated[variant] = {'source_checkpoint': str(original/'latest.pt'),
            'resume_records': state['offset'], 'resume_step': state['step'],
            'original_elapsed_seconds': state['elapsed_seconds'],
            'old_hashes': expected, 'new_hashes': new_hashes,
            'preserved': ['model', 'optimizer', 'ema', 'cpu_rng', 'cuda_rng', 'offset', 'step']}
        del state
    result = {'phase': 'resume_prepared', 'source_run': plan['source_run'],
        'run_id': plan['run_id'], 'models': migrated, 'stats_hash': stat_hash,
        'fingerprint': train.fingerprint, 'prefix_819200_sha256': prefix_hash,
        'full_epoch_members_sha256': full_hash, 'remaining_updates_per_model': 27963,
        'last_batch_records': 4, 'authorization': plan['authorization'],
        'previous_deadline_superseded': True, 'historical_run_unchanged': True}
    save_json(destination/'plan.json', plan)
    save_json(done, result)
    save_json(destination/'pipeline.json', {'phase': 'resume_prepared', 'gpu_count': 1})
    return result
