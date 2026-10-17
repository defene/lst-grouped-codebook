"""Generate disabled, freshly identified configs for all 39 comparisons."""
import argparse
import copy
import json
from pathlib import Path
import uuid
from . import VERSION
from .config import file_digest, validate, write_json
from .epoch_validation import default_protocol
from .runtime import source_hashes


def prepare(bundle, output):
    bundle, output = Path(bundle), Path(output)
    if output.exists():
        raise ValueError('Use a new destination; never overwrite a frozen config set')
    manifest = json.loads((bundle / 'MODEL_MANIFEST.json').read_text(encoding='utf-8'))
    jobs = []
    for row in manifest['models']:
        original = bundle / row['config']
        config = copy.deepcopy(json.loads(original.read_text(encoding='utf-8')))
        config['version'] = VERSION
        config['name'] += '_lossfix_20261014'
        config['diagnostics'] = {'gradient_every_updates': 100}
        config['representation']['colour_inverse'] = ('trilinear_v1' if
            config['representation']['kind'] in ('C', 'C_power', 'C_asinh') else 'nearest_segment')
        if config['representation']['kind'] in ('C', 'C_power', 'C_asinh'):
            config['representation']['inverse_gradient_policy'] = 'saturated_softpath_v1'
        if config['model']['kind'] == 'vq':
            config['model']['vq_geometry'] = 'unit_rms_train_init_v1'
        if config['model']['kind'] == 'vq_drifting':
            config['model']['drift_reconstruction_gradient'] = 'encoder_and_codes_v1'
        config['execution'] = {'enabled': False, 'preflight_report': None, 'alignment_report': None}
        config['train']['selection'] = 'best_validation_rmse'
        config['epoch_validation'] = default_protocol()
        config['evaluation'].update(stage='selection', splits=['val'])
        config['tracking'].update(enabled=False, run_id=uuid.uuid4().hex[:12],
            name=row['name'] + '-lossfix-20261014', group='lossfix_20261014')
        result = validate(config)
        if not result['ok']:
            raise ValueError((row['name'], result['errors']))
        jobs.append((row['name'], config, file_digest(original)))
    for name, config, _ in jobs:
        write_json(output / (name + '.json'), config)
    write_json(output / 'manifest.json', {'status': 'prepared_not_launched', 'version': VERSION,
        'source_hashes': source_hashes(), 'models': [
            {'name': name, 'config': name + '.json', 'historical_config_sha256': sha,
             'new_run_id': config['tracking']['run_id']} for name, config, sha in jobs],
        'historical_checkpoints_are_not_resume_sources': True,
        'requires_new_preflight_and_explicit_launch': True})
    return len(jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', default='outputs/torch_models_20261013')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(f'Prepared {prepare(args.bundle, args.output)} disabled configs; no experiments submitted')
