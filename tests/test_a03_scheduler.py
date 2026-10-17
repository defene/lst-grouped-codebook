"""Prevent unsafe full launches and preserve completed status during postprocessing."""
import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from experiments.paper1.codebook1024 import run_experiment as scheduler
from experiments.paper1.codebook1024 import make_configs


def test_full_launch_rejects_failed_pilot(tmp_path):
    (tmp_path / 'pilot').mkdir()
    scheduler.json_write(tmp_path / 'pilot/gate.json', {'status': 'failed', 'source_hashes': {}})
    with pytest.raises(ValueError, match='successful current-source'):
        scheduler.run_phase(tmp_path, 'full', {})
    assert not (tmp_path / 'full/manifest.json').exists()


def test_config_rejects_unrelated_training_change(tmp_path, monkeypatch):
    original = make_configs.configurations('full')
    paths = []
    for path in original:
        c = scheduler.read(path)
        if not paths:
            c['train']['batch_size'] = 4
        destination = tmp_path / path.name
        scheduler.json_write(destination, c)
        paths.append(destination)
    monkeypatch.setattr(make_configs, 'configurations', lambda phase: paths)
    with pytest.raises(ValueError, match='beyond codebook size'):
        make_configs.check_configs('full')


def test_postprocess_can_read_completed_phase(tmp_path, monkeypatch):
    phase = tmp_path / 'full'
    phase.mkdir()
    scheduler.json_write(phase / 'status.json', {'status': 'training_and_evaluation_complete'})
    monkeypatch.setattr(scheduler, 'source_hashes', lambda: {})
    monkeypatch.setattr(scheduler, 'check_configs', lambda phase: [])
    monkeypatch.setattr(scheduler, 'gpu_snapshot', lambda: {'used_gib': 4, 'total_gib': 32})

    class Process:
        pid = 1
        def poll(self):
            assert scheduler.read(phase / 'status.json')['status'] == 'training_and_evaluation_complete'
            return 0

    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: Process())
    scheduler.execute(['audit'], tmp_path, phase, {'source_hashes': {}, 'config_hashes': {}},
        {'phase': 'full'}, phase, 'audit', phase / 'audit.log')
    assert scheduler.read(phase / 'status.json')['status'] == 'training_and_evaluation_complete'


def test_stop_marker_prevents_process_launch(tmp_path, monkeypatch):
    phase = tmp_path / 'pilot'
    phase.mkdir()
    (tmp_path / 'STOP').touch()
    def forbidden(*args, **kwargs):
        raise AssertionError('A STOP marker must prevent spawning')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    with pytest.raises(scheduler.Stopped):
        scheduler.execute(['train'], tmp_path, phase, {}, {'phase': 'pilot'}, phase, 'train', phase / 'log')


def test_worker_failure_requests_training_checkpoint(tmp_path, monkeypatch):
    phase = tmp_path / 'pilot'
    phase.mkdir()
    destination = phase / 'run'
    destination.mkdir()
    monkeypatch.setattr(scheduler, 'source_hashes', lambda: {})
    monkeypatch.setattr(scheduler, 'check_configs', lambda phase: [])
    readings = iter([{'used_gib': 4, 'total_gib': 32}, {'used_gib': 31.5, 'total_gib': 32}])
    monkeypatch.setattr(scheduler, 'gpu_snapshot', lambda: next(readings))

    class Process:
        pid = 2
        def poll(self):
            return 0 if (destination / 'STOP').exists() else None

    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: Process())
    with pytest.raises(RuntimeError, match='free VRAM below'):
        scheduler.execute(['train'], tmp_path, phase, {'source_hashes': {}, 'config_hashes': {}},
            {'phase': 'pilot'}, destination, 'train', phase / 'log')
    assert (destination / 'STOP').exists()
