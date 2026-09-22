from types import SimpleNamespace
import pytest
from soarm101_lab.real2sim import launcher


class Bus:
    CONNECTING_STATES = {'CONNECTING', 'BUILDING_ENVIRONMENT'}
    alive = False
    data = {'session': 'old', 'state': 'IDLE'}

    def __init__(self): self.sent = []
    def owner_alive(self): return self.alive
    def status(self): return self.data
    def responsive(self, status): return status.get('state') == 'IDLE'
    def send(self, name, payload=None):
        self.sent.append((name, payload))
        return 'command-id'


def setup(tmp_path, monkeypatch):
    bus = Bus(); process = SimpleNamespace(code=None, poll=lambda: process.code)
    calls = []
    def spawn(*args, **kwargs):
        calls.append((args, kwargs)); return process
    monkeypatch.setattr(launcher.subprocess, 'Popen', spawn)
    python = tmp_path / 'python'; python.touch()
    return launcher.RuntimeLauncher(bus, tmp_path, tmp_path), bus, process, calls, str(python)


def test_launch_wait_and_connect_once(tmp_path, monkeypatch):
    service, bus, process, calls, python = setup(tmp_path, monkeypatch)
    payload = {'preview_only': True}
    assert service.connect(payload, python) is None
    assert len(calls) == 1 and calls[0][0][0][0] == python
    assert service.poll() is None and not bus.sent
    with pytest.raises(RuntimeError, match='시작 중'): service.connect(payload, python)
    bus.alive = True  # Old status must not receive the queued command.
    assert service.poll() is None
    bus.data = {'session': 'new', 'state': 'IDLE'}
    assert service.poll() == 'command-id'
    assert service.poll() is None
    assert bus.sent == [('connect', payload)]


def test_existing_runtime_reused(tmp_path, monkeypatch):
    service, bus, _, calls, python = setup(tmp_path, monkeypatch)
    bus.alive = True
    assert service.connect({}, python) == 'command-id'
    assert not calls


def test_failure_and_cancel_do_not_connect(tmp_path, monkeypatch):
    service, bus, process, calls, python = setup(tmp_path, monkeypatch)
    service.connect({}, python); process.code = 1
    with pytest.raises(RuntimeError, match='exit 1'): service.poll()
    assert service.pending is None and not bus.sent
    process.code = None; service.process = None
    service.connect({}, python); service.cancel_connect()
    bus.alive = True; bus.data = {'session': 'new', 'state': 'IDLE'}
    assert service.poll() is None and not bus.sent


def test_timeout_cancels_only_pending_connect(tmp_path, monkeypatch):
    service, bus, process, calls, python = setup(tmp_path, monkeypatch)
    service.connect({}, python)
    monkeypatch.setattr(launcher.time, 'monotonic', lambda: service.started + 301)
    with pytest.raises(RuntimeError, match='5분'): service.poll()
    assert not bus.sent and service.pending is None
    with pytest.raises(RuntimeError, match='아직 준비'): service.connect({}, python)
    assert len(calls) == 1


def test_modes_and_disconnect(tmp_path, monkeypatch):
    service, bus, process, calls, python = setup(tmp_path, monkeypatch)
    service.connect({}, python)
    assert '--headless' in calls[0][0][0]
    stopped = []
    process.terminate = lambda: stopped.append(True)
    assert service.disconnect() is None
    assert stopped == [True] and service.pending is None and not bus.sent
    process.code = 0
    service.connect({}, python, headless=False)
    assert '--headless' not in calls[1][0][0]
    service.cancel_connect(); bus.alive = True
    assert service.disconnect() == 'command-id'
    assert bus.sent == [('disconnect', None)]
