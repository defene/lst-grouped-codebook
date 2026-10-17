"""Read-only verification of completed A02 runs and saved per-sample metrics."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eo_data.core import json_write, sha256_file
from eo_denoise.config import load_config
from eo_denoise.data import PairedDataset
from eo_denoise.engine import source_hashes


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit():
    output = ROOT / 'runs/paper1/var_vs_prithvi_ae_full'
    config_dir = Path(__file__).parent / 'configs/full'
    status = read(output / 'status.json')
    assert status['status'] == 'training_and_evaluation_complete'
    assert len(status['completed']) == 8 and not status['active'] and not status['pending']
    manifest = read(output / 'manifest.json')
    assert source_hashes() == manifest['source_hashes']
    assert {p.name: sha256_file(p) for p in config_dir.glob('*.json')} == manifest['config_hashes']
    report = read(output / 'report/results.json')
    assert len(report['rows']) == 16
    assert all(sha256_file(output / p) == digest for p, digest in report['sources'].items())
    records = {}
    sample_order_hashes = set()
    validation_memberships = set()
    previews = {}
    for name in sorted(status['completed']):
        run = output / name
        modality = name.split('_')[1]
        config = load_config(config_dir / (name + '.json'))
        assert config == read(run / 'config.json')
        train_ds = PairedDataset(config, 'train')
        assert train_ds.fingerprint == read(run / 'data_fingerprint.json')
        expected = set(train_ds.record_ids.tolist())
        assert expected == set(np.load(run / 'train_record_ids.npy').tolist())
        assert len(expected) == 230659
        train_ds.close()
        logs = [json.loads(s) for s in (run / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['step'] for r in logs] == list(range(1, 10814))
        assert read(run / 'status.json')['status'] == 'complete'
        assert logs[-1]['epoch'] == 3 and logs[-1]['consumed'] == 40
        counts = Counter(int(s.rsplit(':', 1)[1]) for r in logs for s in r['sample_ids'])
        assert set(counts) == expected and Counter(counts.values()) == {3: 230619, 4: 40}
        assert sum(r['batch_samples'] for r in logs) == sum(counts.values()) == 692017
        assert all(math.isfinite(r[k]) for r in logs for k in ('reconstruction', 'gradient', 'bottleneck', 'kl', 'total', 'gradient_norm'))
        assert all(r['kl'] == r['cross_view'] == r['stability'] == 0 for r in logs)
        if 'prithvi' in name:
            assert config['model']['backbone'] == 'prithvi_eo2_continuous'
            assert config['model']['quantizer'] == 'ae'
            assert all(r['bottleneck'] == 0 for r in logs)
        order_hash = hashlib.sha256(json.dumps([r['sample_ids'] for r in logs]).encode()).hexdigest()
        sample_order_hashes.add(order_hash)
        before = float(np.mean([r['reconstruction'] for r in logs[-2000:-1000]]))
        after = float(np.mean([r['reconstruction'] for r in logs[-1000:]]))
        record = dict(training_records=len(expected), sample_presentations=sum(counts.values()),
                      steps=logs[-1]['step'], sample_order_sha256=order_hash,
                      reconstruction_previous_1000_mean=before, reconstruction_final_1000_mean=after,
                      final_window_reduction_percent=(1-after/before)*100,
                      all_losses_finite=True, kl_zero=True,
                      prithvi_bottleneck_zero=True if 'prithvi' in name else None,
                      diagnostics={}, evaluations={})
        for folder in sorted(run.glob('val_step_*')):
            metric = read(folder / 'metrics.json')
            record['diagnostics'][folder.name] = dict(
                samples=metric['evaluated_pairs'], coverage=metric['coverage'],
                all_rmse=metric['regions'][modality + '/all']['rmse'],
                selection_score=metric['selection_score'])
        checkpoint_sha = sha256_file(run / 'last.pt')
        val_config = copy.deepcopy(config)
        val_config['data']['val_limit'] = 11757
        val_ds = PairedDataset(val_config, 'val')
        expected_val = {'prithvi:val:' + str(i) for i in val_ds.record_ids.tolist()}
        assert len(expected_val) == 11757
        assert not expected.intersection(val_ds.record_ids.tolist())
        for condition, coverage in [('clean', 0), ('noisy10', .1)]:
            folder = run / ('eval_' + condition)
            metric = read(folder / 'metrics.json')
            assert metric['step'] == 10813 and metric['evaluated_pairs'] == 11757
            assert metric['checkpoint_sha256'] == checkpoint_sha and metric['coverage'] == coverage
            assert metric['data_fingerprint'] == val_ds.fingerprint
            by_region = defaultdict(lambda: defaultdict(list))
            all_views = Counter()
            with (folder / 'samples.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    assert row['modality'] == modality and row['sample_id'] in expected_val
                    assert math.isfinite(row['mse_physical']) and row['mse_physical'] >= 0
                    by_region[row['region']][row['sample_id']].append(row['mse_physical'])
                    if row['region'] == 'all':
                        all_views[(row['sample_id'], row['view'])] += 1
            assert all_views == Counter({(sid, view): 1 for sid in expected_val for view in (0, 1)})
            validation_memberships.add(hashlib.sha256('\n'.join(sorted(expected_val)).encode()).hexdigest())
            recomputed = {}
            for region, samples in by_region.items():
                rmse = math.sqrt(float(np.mean([np.mean(v) for v in samples.values()])))
                assert math.isclose(rmse, metric['regions'][modality + '/' + region]['rmse'], rel_tol=1e-9)
                recomputed[region] = rmse
            with np.load(folder / 'preview.npz') as source:
                arrays = {k: source[k] for k in source.files}
            key = (modality, condition)
            fields = [modality + suffix for suffix in ('_target', '_reference_mask', '_image', '_input_mask')]
            if key in previews:
                assert all(np.array_equal(arrays[f], previews[key][f]) for f in fields)
            else:
                previews[key] = arrays
            record['evaluations'][condition] = dict(pairs=11757, views_per_pair=2,
                checkpoint_sha256=checkpoint_sha, recomputed_rmse=recomputed,
                saved_metric_rmse_matches_raw_samples=True, data_fingerprint_matches=True)
        val_ds.close()
        records[name] = record
        print(json.dumps(dict(name=name, final_window_reduction_percent=record['final_window_reduction_percent'],
              diagnostics=record['diagnostics'], evaluations={k:v['recomputed_rmse'] for k,v in record['evaluations'].items()})), flush=True)
    assert len(sample_order_hashes) == len(validation_memberships) == 1
    gpu = [json.loads(s) for s in (output / 'gpu.jsonl').read_text().splitlines()]
    result = dict(status='passed', verified_utc=datetime.now(timezone.utc).isoformat(),
                  runs=records, frozen_source_and_config_match=True, same_training_order_all_runs=True,
                  same_validation_membership_all_runs=True, same_preview_inputs_within_modality_condition=True,
                  raw_sample_metrics_recomputed=True, global_gpu_peak_gib=max(r['used_gib'] for r in gpu),
                  minimum_sampled_device_free_gib=min(r['total_gib']-r['used_gib'] for r in gpu),
                  audit_script_sha256=sha256_file(__file__))
    json_write(output / 'report/completion_audit.json', result)
    print('COMPLETION AUDIT PASSED', flush=True)


if __name__ == '__main__':
    audit()
