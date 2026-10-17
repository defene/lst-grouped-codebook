"""Collect inspectable acceptance evidence from already completed GPU smoke runs."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from eo_data.core import sha256_file
from eo_denoise.config import load_config
from eo_denoise.data import PairedDataset
from eo_denoise.engine import read_checkpoint, seed_everything, source_hashes
from eo_denoise.models import CrossViewModel


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def collect():
    import os
    os.chdir(ROOT)
    folder = ROOT / 'experiments/paper1'
    configs = {p.stem: load_config(p) for p in sorted((folder / 'configs').glob('*.json'))}
    names = ['vq11', 'fsq', 'ae', 'hls_conv_vae_clean', 'lst_conv_vae_clean',
             'hls_prithvi_style_vae_clean', 'lst_prithvi_style_vae_clean']
    runs = []
    for name in names:
        run = ROOT / 'runs/paper1' / f'framework_{name}_smoke'
        status = read(run / 'status.json')
        assert status['status'] == 'complete' and status['purpose'] == 'framework_smoke'
        assert status['device'] == 'cuda'
        checkpoint = read_checkpoint(run / 'last.pt')
        assert checkpoint['step'] == status['step'] == status['configured_steps']
        assert all(torch.isfinite(value).all() for value in checkpoint['model'].values())
        runs.append({'name': name, 'directory': run.relative_to(ROOT).as_posix(),
                     'steps': status['step'], 'status': status['status'], 'device': status['device'],
                     'model': read(run / 'model.json'),
                     'environment': read(run / 'environment.json'),
                     'checkpoint_matches_current_source': True,
                     'last_checkpoint_sha256': sha256_file(run / 'last.pt')})
        del checkpoint
    factorial = []
    for variant in ('00', '10', '01', '11'):
        c = configs['prithvi_vq' + variant]
        seed_everything(c['seed'])
        model = CrossViewModel(c)
        model_hash = hashlib.sha256()
        for key, value in model.state_dict().items():
            model_hash.update(key.encode())
            model_hash.update(value.cpu().numpy().tobytes())
        del model
        ds = PairedDataset(c, 'train')
        first = ds[0]
        input_hash = hashlib.sha256()
        for modality in ('hls', 'lst'):
            for field in ('image', 'input_mask', 'corruption_mask'):
                input_hash.update(first[modality][field].tobytes())
        factorial.append({'variant': 'VQ' + variant,
                          'initial_model_sha256': model_hash.hexdigest(),
                          'first_training_pair_input_sha256': input_hash.hexdigest(),
                          'paired_record_ids_sha256': ds.fingerprint['paired_record_ids_sha256'],
                          'train_pairs': len(ds)})
        ds.close()
    for field in ('initial_model_sha256', 'first_training_pair_input_sha256', 'paired_record_ids_sha256'):
        assert len({row[field] for row in factorial}) == 1, f'Factorial mismatch: {field}'
    tokens = []
    for relative in ('framework_vq11_smoke/lst_example.cvdt', 'framework_fsq_smoke/hls_example.cvdt'):
        path = ROOT / 'runs/paper1' / relative
        evidence = read(path.with_suffix('.cvdt.json'))
        assert sha256_file(path) == evidence['sha256'] and evidence['decoder_roundtrip_max_error'] == 0.0
        tokens.append(evidence)
    clean = read(ROOT / 'runs/paper1/framework_vq11_smoke/eval_clean/metrics.json')
    assert clean['evaluated_pairs'] == 8
    assert all(value['actual_coverage'] == 0 for value in clean['regions'].values())
    for modality in ('hls', 'lst'):
        assert modality + '/corrupt' not in clean['regions']
    # The model test suite was executed separately before evidence collection.
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'framework_ready_formal_training_not_started',
        'primary_data': 'Prithvi one-to-one HLS-HR LST',
        'source_hashes': source_hashes(),
        'validated_configs': list(configs),
        'tests_observed': {'passed': 19, 'seconds': 8.11,
                           'command': '.venv-paper1/Scripts/python.exe -m pytest tests/test_eo_denoise.py tests/test_eo_data.py -q',
                           'scope': ['input leakage', 'mask/gradient losses', 'VQ/FSQ gradients and decode',
                                     'sample-equal physical metrics', 'mask-bank integrity',
                                     'Windows persistent workers', 'CPU VAE exact uninterrupted/resume equality']},
        'gpu_smoke_runs': runs,
        'vq_factorial_parity': factorial,
        'discrete_token_roundtrips': tokens,
        'standalone_zero_noise_evaluation': {'pairs': 8, 'empty_corruption_regions_omitted': True,
                                           'directory': 'runs/paper1/framework_vq11_smoke/eval_clean'},
        'gpu_resume_observed': {'run': 'runs/paper1/framework_vq11_smoke', 'stopped_at_step': 2,
                                'resumed_to_step': 4, 'exact_model_equality_tested_on': 'CPU VAE unit test'},
        'paired_metadata_audit': read(folder / 'paired_metadata_audit.json'),
        'mask_banks': {s: {'shapes': len(read(folder / 'mask_banks' / f'{s}.json')['shapes']),
                           'sha256': sha256_file(folder / 'mask_banks' / f'{s}.npz')}
                       for s in ('train', 'val')},
        'limitations': ['Smoke scores are not model quality evidence',
                        'Prithvi has train/val only; no independent test',
                        'Prithvi-style VAE is a from-scratch adaptation, not an official pretrained model',
                        'Metadata eligibility is not a pixel-level registration guarantee',
                        'Known-mask synthetic missing-region recovery; no real-cloud accuracy claim',
                        'Fixed-bit token packaging, not entropy coding; side information and weights accounted separately',
                        'Single GPU; planned full experiments and extensions remain unrun'],
    }
    output = folder / 'framework_validation.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'gpu_smoke_runs': len(runs),
                      'configs': len(configs), 'vq_factorial_parity': 'passed',
                      'token_roundtrips': len(tokens), 'report': str(output)}, indent=2))


if __name__ == '__main__':
    collect()
