"""Local IPC/liveness regressions; no Isaac renderer or hardware connections."""
import time
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from soarm101_lab.real2sim.storage import Bus
from soarm101_lab.real2sim.runtime import Runtime


@pytest.mark.parametrize('state', ['CONNECTING', 'BUILDING_ENVIRONMENT'])
def test_busy_ui_requires_live_owner(tmp_path, state):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.real2sim.ui import Window

    app = QApplication.instance() or QApplication([])
    owner = Bus(tmp_path / 'ipc', owner=True)
    w = Window(tmp_path)
    try:
        owner.publish(state=state, message='Isaac 환경 생성 중', timestamp=time.time() - 600)
        w.refresh()
        assert state in w.status.text()
        assert '연결 끊김' not in w.status.text()
        with pytest.raises(RuntimeError, match='생성 중'):
            w.bus.send('connect')
        # Simulate process exit without CLOSED publication: OS releases flock.
        owner.lock.close()
        owner.lock = None
        w.refresh()
        assert '연결 끊김' in w.status.text()
    finally:
        owner.close()
        w.close()
        app.processEvents()


def test_normal_stale_and_closed_are_disconnected(tmp_path):
    owner = Bus(tmp_path, owner=True)
    client = Bus(tmp_path)
    try:
        owner.publish(state='CONNECTED', timestamp=time.time() - 10)
        assert not client.responsive(client.status())
        with pytest.raises(RuntimeError, match='not responding'):
            client.send('connect')
        owner.publish(state='CONNECTED')
        assert client.responsive(client.status())
        owner.close()
        assert not client.responsive(client.status())
    finally:
        owner.close()


@pytest.mark.parametrize('fail', [False, True])
def test_connect_completion_published_before_tick_and_ipc_reply(tmp_path, fail):
    app = Mock()
    app.is_running.side_effect = [True, False]
    r = Runtime(SimpleNamespace(workspace=str(tmp_path)), app)
    client = Bus(tmp_path / 'ipc')
    identifier = client.send('connect')

    def connect(payload):
        r.state = 'BUILDING_ENVIRONMENT'
        r.publish()
        assert client.responsive(client.status())
        if fail:
            raise ValueError('environment failed')
        r.state = 'PREVIEW'
        r.message = 'preview complete'

    r.connect = connect

    def tick():
        assert client.status()['state'] == ('ERROR' if fail else 'PREVIEW')
        if fail:
            with pytest.raises(RuntimeError, match='environment failed'):
                client.wait(identifier, timeout=1)
        else:
            assert client.wait(identifier, timeout=1) == 'Connected; follower not commanded'
        assert client.responsive(client.status())

    r.tick = tick
    r.run()
    assert client.status()['state'] == 'CLOSED'
    assert not list(client.root.glob('cmd_*.json'))


def test_connect_publishes_before_profile_load(tmp_path, monkeypatch):
    r = Runtime(SimpleNamespace(workspace=str(tmp_path)), Mock())
    def load(path):
        assert r.bus.status()['state'] == 'CONNECTING'
        raise ValueError('bad profile')
    monkeypatch.setattr('soarm101_lab.real2sim.runtime.load', load)
    try:
        with pytest.raises(ValueError, match='bad profile'):
            r.connect({'profile': 'unused'})
    finally:
        r.writer.shutdown()
        r.bus.close()


def test_building_status_precedes_environment_work(tmp_path, monkeypatch):
    r = Runtime(SimpleNamespace(workspace=str(tmp_path), device='cpu'), Mock())
    old_env = Mock()
    r.env = old_env
    manager = Mock()
    monkeypatch.setitem(sys.modules, 'isaaclab.envs', SimpleNamespace(ManagerBasedEnv=manager))
    monkeypatch.setitem(
        sys.modules, 'soarm101_lab.tasks.manager_based.soarm101_lab',
        SimpleNamespace(SO101TeleopEnvCfg=Mock(return_value=Mock())),
    )
    monkeypatch.setattr('soarm101_lab.real2sim.runtime.load', lambda path: {})

    def build(*args):
        assert r.bus.status()['state'] == 'BUILDING_ENVIRONMENT'
        old_env.close.assert_called_once()
        assert r.env is None
        manager.assert_not_called()
        raise ValueError('stop before Isaac construction')

    monkeypatch.setattr('soarm101_lab.real2sim.runtime.build_usd', build)
    try:
        with pytest.raises(ValueError, match='stop before Isaac'):
            r.connect({'profile': 'unused', 'hardware': {}, 'preview_only': True})
    finally:
        r.writer.shutdown()
        r.bus.close()


def test_disconnect_replies_then_cleans_up_without_tick(tmp_path):
    app = Mock(); app.is_running.return_value = True
    r = Runtime(SimpleNamespace(workspace=str(tmp_path)), app)
    r.stop = Mock(); r.disconnect_hardware = Mock(); r.tick = Mock()
    r.env = Mock()
    r.bus.commands = Mock(return_value=iter([{'id': 'disconnect-test', 'command': 'disconnect', 'payload': {}}]))
    r.run()
    assert r.shutdown_requested
    r.tick.assert_not_called()
    r.env.close.assert_called_once()
    assert r.bus.status()['state'] == 'CLOSED'
    import json
    reply = json.loads((r.bus.root / 'reply_disconnect-test.json').read_text())
    assert reply['ok'] and 'shutting down' in reply['result']
