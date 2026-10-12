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
        assert '이전' in w.status.text()
        assert w.live_backend.isEnabled() and w.capture_start.isEnabled()
        assert not w.runtime_enabled
        assert not (tmp_path/'ipc/ui_heartbeat.json').exists()
        with pytest.raises(RuntimeError, match='생성 중'):
            w.bus.send('connect')
        # Simulate process exit without CLOSED publication: OS releases flock.
        owner.lock.close()
        owner.lock = None
        w.refresh()
        assert '대기' in w.status.text()
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


def test_mujoco_startup_ignores_previous_isaac_status(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.real2sim.ui import Window
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    app = QApplication.instance() or QApplication([])
    registry = Registry(tmp_path / 'registry')
    monkeypatch.setattr(ui, 'Registry', lambda: registry)
    w = Window(tmp_path / 'live')
    try:
        w.live_backend.setCurrentIndex(w.live_backend.findData('mujoco'))
        w.runtime_launcher.backend = 'mujoco'
        w.runtime_launcher.pending = {'preview_only': True}
        monkeypatch.setattr(w.runtime_launcher, 'poll', lambda: None)
        monkeypatch.setattr(w.bus, 'owner_alive', lambda: True)
        monkeypatch.setattr(w.bus, 'status', lambda: {'state': 'CLOSED', 'backend': 'isaac'})
        w.refresh()
        assert w.live_backend.currentData() == 'mujoco'
        assert w.workflow_panel.backend.currentData() == 'mujoco'
    finally:
        w.runtime_launcher.pending = None
        w.close(); app.processEvents()


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


@pytest.mark.parametrize('renderer', ['software-egl', 'osmesa'])
def test_mujoco_real_sim_worker_preview_and_disconnect(tmp_path, renderer):
    import os
    import subprocess
    from soarm101_lab.real2sim.workspaces.package import ROOT
    script = ROOT / 'scripts/real2sim/real2sim_sim.py'
    code = """
import runpy, sys
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
assert not any(n.split('.')[0] in ('isaaclab','omni','pxr') for n in sys.modules)
"""
    bus = Bus(tmp_path / 'ipc')
    log = tmp_path / 'process.log'
    with log.open('w') as stream:
        process = subprocess.Popen([sys.executable, '-c', code, str(script), '--workspace', str(tmp_path),
                                    '--backend', 'mujoco', '--renderer', renderer, '--headless'],
                                   cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'source/soarm101_lab')},
                                   stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 20
            while not bus.responsive(bus.status()):
                assert process.poll() is None, log.read_text()
                assert time.monotonic() < deadline, log.read_text()
                time.sleep(.05)
            assert bus.status()['backend'] == 'mujoco'
            command = bus.send('connect', {'profile': str(ROOT / 'tests/real2sim/fixtures/scene_profile.json'),
                                          'hardware': {}, 'preview_only': True})
            assert bus.wait(command, timeout=20) == 'Connected; follower not commanded'
            while not all((bus.root / (name+'.jpg')).exists() for name in ('sim_side', 'sim_wrist')):
                assert process.poll() is None, log.read_text()
                assert time.monotonic() < deadline, log.read_text()
                time.sleep(.05)
            assert bus.status()['state'] == 'PREVIEW'
            assert not bus.status()['real_connected']
            assert not (tmp_path / 'generated').exists()
            assert list(bus.root.glob('scene_mujoco_*/scene.xml'))
            bus.send('disconnect')
            assert process.wait(timeout=15) == 0, log.read_text()
            assert bus.status()['state'] == 'CLOSED'
            assert not list(bus.root.glob('scene_mujoco_*'))
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


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


def test_capture_start_is_explicit_and_always_real_only(tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.real2sim.ui import Window
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    from soarm101_lab.real2sim import launcher
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui,'Registry',lambda:Registry(tmp_path/'registry'))
    monkeypatch.setattr(launcher.ctypes.util,'find_library',lambda name:None)
    spawn=Mock();monkeypatch.setattr(launcher.subprocess,'Popen',spawn)
    w=Window(tmp_path/'live')
    try:
        assert not w.runtime_enabled and w.runtime_launcher.pending is None
        assert not (w.bus.root/'ui_heartbeat.json').exists()
        w.refresh();w.tabs.setCurrentIndex(1);w.tabs.setCurrentIndex(0)
        spawn.assert_not_called()
        connect=Mock(return_value=None);w.runtime_launcher.connect=connect
        from soarm101_lab.real2sim.workspaces.package import ROOT
        w.selected(ROOT / 'tests/real2sim/fixtures/scene_profile.json')
        # The sim selection and even an invalid Pipeline workspace must not affect Capture.
        w.live_backend.setCurrentIndex(w.live_backend.findData('isaac'))
        w.workflow_panel.workspace.setText('/not/a/published/workspace')
        w.start_capture()
        assert connect.call_args.kwargs['backend']=='real'
        payload=connect.call_args.args[0]
        assert payload['preview_only'] is False and 'workspace' not in payload
        assert w.requested_backend=='real'
        assert not list(w.bus.root.glob('cmd_*.json'))  # No automatic Start Teleop.
    finally:
        w.runtime_enabled=False;w.close();app.processEvents()


def test_old_runtime_does_not_disable_or_override_sim_selection(tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.real2sim.ui import Window
    app=QApplication.instance() or QApplication([])
    owner=Bus(tmp_path/'ipc',owner=True)
    owner.publish(state='BUILDING_ENVIRONMENT',backend='isaac',real_connected=False,sim_connected=True)
    w=Window(tmp_path)
    try:
        assert not w.runtime_enabled and '대기' in w.status.text()
        assert w.live_backend.isEnabled()
        w.live_backend.setCurrentIndex(w.live_backend.findData('mujoco'));w.refresh()
        assert w.live_backend.currentData()=='mujoco'
        assert w.workflow_panel.backend.currentData()=='mujoco'
        assert w.capture_start.isEnabled()
        assert not w.capture_button.isEnabled()
        assert not (w.bus.root/'ui_heartbeat.json').exists()
        assert not list(w.bus.root.glob('cmd_*.json'))
    finally:w.close();owner.close();app.processEvents()


def test_cancelled_connection_does_not_leave_capture_start_disabled(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.real2sim.ui import Window
    app=QApplication.instance() or QApplication([])
    w=Window(tmp_path)
    try:
        w.runtime_enabled=True
        w.pending['cancelled-connect']='connect'
        w.refresh()
        assert not w.capture_start.isEnabled()
        w.runtime_launcher.disconnect=Mock(return_value=None)
        w.disconnect_sim();w.refresh()
        assert 'cancelled-connect' not in w.pending
        assert w.capture_start.isEnabled() and w.live_backend.isEnabled()
        assert not w.runtime_enabled
    finally:w.close();app.processEvents()
