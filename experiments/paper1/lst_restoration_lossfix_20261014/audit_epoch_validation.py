"""Explicit CPU-only validation/resume regression gate; never launches a job."""
from pathlib import Path
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET


def run_checks(config, output):
    from .config import write_json, validate
    from .runtime import protocol_digest, source_hashes
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    source_before = source_hashes()
    xml = output.parent / ('epoch_validation_' + uuid.uuid4().hex + '.xml')
    temporary = output.parent / ('cpu_test_' + uuid.uuid4().hex)
    command = [sys.executable, '-m', 'pytest', 'tests/test_lst_lossfix_epoch_validation_20261014.py',
               '-q', '--tb=short', '--basetemp', str(temporary), '--junitxml', str(xml)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, errors='replace')
    log = xml.with_suffix('.log')
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    suites = list(ET.parse(xml).getroot().iter('testsuite')) if xml.exists() else []
    passed = (result.returncode == 0 and bool(suites) and validate(config)['ok']
              and source_before == source_hashes()
              and sum(int(s.get('tests', 0)) for s in suites) >= 17
              and all(int(s.get(k, 0)) == 0 for s in suites for k in ('failures', 'errors', 'skipped')))
    report = {'status': 'passed' if passed else 'failed', 'gates': {'epoch_validation': passed},
              'protocol_config_sha256': protocol_digest(config), 'source_hashes': source_before,
              'junit_xml': str(xml), 'log': str(log), 'device': 'cpu',
              'real_data_used': False, 'formal_training': False, 'authorizes_training': False}
    write_json(output, report)
    if not passed:
        raise RuntimeError(f'Epoch validation regression failed: {log}')
    return report
