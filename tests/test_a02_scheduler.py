"""Scheduler resilience tests; model and dataset source remain frozen."""
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/paper1/var_vs_prithvi_ae/run_suite.py'
spec = importlib.util.spec_from_file_location('a02_scheduler', SCRIPT)
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)


@pytest.mark.parametrize('blocked_attempts', [1, 2])
def test_status_recovers_from_transient_replace_conflict(tmp_path, monkeypatch, blocked_attempts):
    target = tmp_path / 'status.json'
    target.write_text('{"status":"old"}', encoding='utf-8')
    original_replace = Path.replace
    calls = []
    sleeps = []

    def replace(path, destination):
        calls.append(path)
        if len(calls) <= blocked_attempts:
            assert scheduler.read(target) == {'status': 'old'}
            raise PermissionError(13, 'Sharing conflict')
        return original_replace(path, destination)

    monkeypatch.setattr(Path, 'replace', replace)
    monkeypatch.setattr(scheduler.time, 'sleep', sleeps.append)
    scheduler.json_write(target, {'status': 'running'})
    assert scheduler.read(target) == {'status': 'running'}
    assert len(calls) == blocked_attempts + 1
    assert len(sleeps) == blocked_attempts
    assert not target.with_suffix('.json.tmp').exists()


@pytest.mark.parametrize('exception,attempts', [(PermissionError(13, 'Denied'), 8), (OSError(28, 'Disk full'), 1)])
def test_status_surfaces_persistent_or_unrelated_errors(tmp_path, monkeypatch, exception, attempts):
    calls = []
    sleeps = []

    def fail(path, value):
        calls.append(path)
        raise exception

    monkeypatch.setattr(scheduler, 'atomic_json_write', fail)
    monkeypatch.setattr(scheduler.time, 'sleep', sleeps.append)
    with pytest.raises(type(exception)) as caught:
        scheduler.json_write(tmp_path / 'status.json', {'status': 'running'})
    assert caught.value is exception
    assert len(calls) == attempts
    assert len(sleeps) == attempts - 1
    assert sum(sleeps) <= 5.5
