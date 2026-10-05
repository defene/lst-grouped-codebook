"""A03: derive four scratch VQ-VAE configurations, changing only vocabulary size."""
from pathlib import Path
import copy
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from eo_data.core import sha256_file
from eo_denoise.config import validate
from experiments.paper1.var_vs_prithvi_ae.run_suite import json_write

BASELINE = ROOT / 'experiments/paper1/var_vs_prithvi_ae'


def configurations(phase):
    return sorted((HERE / 'configs' / phase).glob('*.json'))


def check_configs(phase):
    paths = configurations(phase)
    if len(paths) != 4:
        raise ValueError('A03 requires exactly four convolutional configurations')
    for path in paths:
        c = json.loads(path.read_text(encoding='utf-8'))
        original = json.loads((BASELINE / 'configs' / phase / path.name).read_text(encoding='utf-8'))
        expected = copy.deepcopy(original)
        expected['model']['codebook_size'] = 1024
        if c != expected or c['model']['backbone'] != 'var_conv':
            raise ValueError(f'{path.name}: changes beyond codebook size are forbidden')
        validate(c)
    return paths


def main():
    provenance = {}
    for phase in ('pilot', 'full'):
        for path in sorted((BASELINE / 'configs' / phase).glob('*_conv_*.json')):
            c = json.loads(path.read_text(encoding='utf-8'))
            c['model']['codebook_size'] = 1024
            target = HERE / 'configs' / phase / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and json.loads(target.read_text(encoding='utf-8')) != c:
                raise ValueError(f'Refusing to overwrite a different configuration: {target}')
            json_write(target, c)
            provenance[str(target.relative_to(HERE))] = dict(
                baseline=str(path.relative_to(ROOT)), baseline_sha256=sha256_file(path),
                config_sha256=sha256_file(target), changes={'model.codebook_size': [4096, 1024]})
        check_configs(phase)
    json_write(HERE / 'config_derivation.json', provenance)
    print('Eight configurations verified: four pilot + four full; only codebook_size differs.')


if __name__ == '__main__':
    main()
