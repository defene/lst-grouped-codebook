"""Record local verification and refresh the unlaunched source/config manifest."""
import ast
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.paper1.lst_restoration_epochval_20261014.config import file_digest, validate, write_json
from experiments.paper1.lst_restoration_epochval_20261014.runtime import source_hashes


def main():
    package = ROOT / 'experiments/paper1/lst_restoration_epochval_20261014'
    origins = json.loads((package / 'SOURCE_ORIGINS.json').read_text(encoding='utf-8'))
    for origin in origins.values():
        assert file_digest(ROOT / origin['source']) == origin['sha256'], origin
    for path in package.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    directory = ROOT / 'runs/paper1/lst_restoration_epochval_20261014/configs'
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    for row in manifest['models']:
        config = json.loads((directory / row['config']).read_text(encoding='utf-8'))
        assert validate(config)['ok'], row['name']
        assert not config['execution']['enabled'] and not config['tracking']['enabled']
    evidence_path = ROOT / 'outputs/epochval_20261014_tests/preflight_cpu/epoch_validation.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    assert evidence['status'] == 'passed' and evidence['source_hashes'] == source_hashes()
    suites = list(ET.parse(evidence['junit_xml']).getroot().iter('testsuite'))
    assert all(int(s.get(key, 0)) == 0 for s in suites for key in ('failures', 'errors', 'skipped'))
    inventory = json.loads((ROOT / 'outputs/torch_experiments_20261013/snapshot/DOWNLOAD_MANIFEST.json').read_text())
    prefixes = ('output/stage1_20260930_attempt02/runs/', 'output/stage2_vq_20261005_attempt02/runs/',
                'output/stage3_grouped_20261005_attempt01/runs/')
    checkpoints = [r for r in inventory['files'] + inventory['excluded']
                   if r['path'].startswith(prefixes) and r['path'].endswith(('.pt', '.pth', '.ckpt'))]
    manifest['source_hashes'] = source_hashes()
    write_json(directory / 'manifest.json', manifest)
    report = {'status': 'local_code_verified_not_launched', 'models_prepared': len(manifest['models']),
              'unit_tests_passed': sum(int(s.get('tests', 0)) for s in suites),
              'cpu_regression_evidence': str(evidence_path.relative_to(ROOT)),
              'historical_source_files_unchanged': len(origins),
              'historical_checkpoint_inventory': {'files': len(checkpoints),
                  'names': sorted({Path(r['path']).name for r in checkpoints}),
                  'scope': 'Downloaded results manifest; not a fresh server scan'},
              'real_data_used': False, 'gpu_used': False, 'jobs_submitted': False,
              'source_hashes': source_hashes(),
              'limits': ['No full-model GPU preflight', 'No new real-data validation scores',
                         'No historical intermediate checkpoints recovered']}
    write_json(ROOT / 'runs/paper1/lst_restoration_epochval_20261014/LOCAL_VERIFICATION.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'source_hashes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
