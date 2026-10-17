from unittest.mock import Mock
import pytest
from experiments.paper1.lst_v2_5epoch import supervise


def test_retries_sharing_failure_without_changing_payload(monkeypatch):
    writer = Mock(side_effect=[PermissionError('shared'), PermissionError('shared'), None])
    monkeypatch.setattr(supervise, '_json_write', writer)
    monkeypatch.setattr(supervise.time, 'sleep', lambda _: None)
    payload = {'step': 10472}
    supervise.json_write('status.json', payload)
    assert writer.call_count == 3
    assert all(c.args == ('status.json', payload) for c in writer.call_args_list)


def test_persistent_write_failure_is_not_hidden(monkeypatch):
    writer = Mock(side_effect=PermissionError('permanent'))
    monkeypatch.setattr(supervise, '_json_write', writer)
    monkeypatch.setattr(supervise.time, 'sleep', lambda _: None)
    with pytest.raises(PermissionError):
        supervise.json_write('status.json', {})
    assert writer.call_count == 12
