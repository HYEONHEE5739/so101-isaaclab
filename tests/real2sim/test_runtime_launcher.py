from types import SimpleNamespace
import os
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


@pytest.mark.parametrize('state', ['ERROR', 'PREVIEW', 'CONNECTED'])
def test_isaac_reconnect_uses_fresh_worker_after_environment_exists(tmp_path, monkeypatch, state):
    service, bus, _, calls, python = setup(tmp_path, monkeypatch)
    bus.alive = True
    bus.data = {'session': 'old', 'state': state, 'backend': 'isaac', 'sim_connected': True}
    assert service.connect({'preview_only': True}, python) is None
    assert bus.sent == [('disconnect', None)] and not calls
    service.poll()
    assert not calls  # No new Kit/context while the previous worker still owns devices.
    bus.alive = False
    service.poll()
    assert len(calls) == 1 and service.backend == 'isaac'
    service.cancel_connect()


def test_mujoco_reconnect_still_reuses_existing_worker(tmp_path, monkeypatch):
    service, bus, _, calls, python = setup(tmp_path, monkeypatch)
    bus.alive = True
    bus.data = {'session': 'old', 'state': 'PREVIEW', 'backend': 'mujoco', 'renderer': 'egl', 'sim_connected': True}
    assert service.connect({}, python, backend='mujoco') == 'command-id'
    assert bus.sent == [('connect', {})] and not calls


def test_failed_capture_retry_waits_for_old_worker_exit(tmp_path, monkeypatch):
    service, bus, _, calls, python = setup(tmp_path, monkeypatch)
    bus.alive = True
    bus.data = {'session': 'old', 'state': 'ERROR', 'backend': 'real'}
    payload = {'profile': 'fixture'}
    assert service.connect(payload, python, backend='real') is None
    assert bus.sent == [('disconnect', None)] and not calls
    service.poll()
    assert not calls
    bus.alive = False
    service.poll()
    assert len(calls) == 1 and service.backend == 'real'
    assert service.pending == payload
    assert bus.sent == [('disconnect', None)]
    bus.alive = True
    bus.data = {'session': 'new', 'state': 'IDLE', 'backend': 'real'}
    service.poll()
    assert bus.sent == [('disconnect', None), ('connect', payload)]


def test_isaac_worker_does_not_forward_mujoco_renderer(tmp_path, monkeypatch):
    import runpy
    import sys
    from unittest.mock import Mock
    from pathlib import Path
    from soarm101_lab.real2sim import runtime
    app = Mock()
    received = []
    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument('--headless', action='store_true')
        def __init__(self, args):
            received.append(args)
            self.app = app
    monkeypatch.setitem(sys.modules, 'isaaclab.app', SimpleNamespace(AppLauncher=AppLauncher))
    worker = Mock()
    monkeypatch.setattr(runtime, 'Runtime', worker)
    script = Path(launcher.__file__).resolve().parents[4] / 'scripts/real2sim/real2sim_sim.py'
    monkeypatch.setattr(sys, 'argv', [str(script), '--workspace', str(tmp_path), '--backend', 'isaac', '--headless'])
    runpy.run_path(str(script), run_name='__main__')
    assert len(received) == 1
    assert received[0].enable_cameras and not hasattr(received[0], 'renderer')
    worker.return_value.run.assert_called_once()
    app.close.assert_called_once()


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
    service.poll()
    process.code = None
    service.connect({}, python, headless=False)
    assert '--headless' not in calls[1][0][0]
    service.cancel_connect(); bus.alive = True
    assert service.disconnect() == 'command-id'
    assert bus.sent == [('disconnect', None)]


def test_worker_does_not_inherit_gui_python_packages(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', '/gui/venv/lib/python3.11/site-packages')
    monkeypatch.setenv('PYTHONHOME', '/gui/venv')
    monkeypatch.setenv('SO101_CUSTOM_OPTION', 'retained')
    service, bus, process, calls, python = setup(tmp_path, monkeypatch)
    service.connect({}, python)
    env = calls[0][1]['env']
    assert env['PYTHONPATH'] == str(tmp_path / 'source/soarm101_lab')
    assert 'PYTHONHOME' not in env and env['PYTHONNOUSERSITE'] == '1'
    assert env['SO101_CUSTOM_OPTION'] == 'retained'
    assert os.environ['PYTHONHOME'] == '/gui/venv'


def test_mujoco_launch_and_wrong_engine_rejected(tmp_path, monkeypatch):
    service, bus, process, calls, python = setup(tmp_path, monkeypatch)
    monkeypatch.setenv('LIBGL_ALWAYS_SOFTWARE', '1')
    monkeypatch.setenv('__NV_PRIME_RENDER_OFFLOAD', '1')
    service.connect({}, python, backend='mujoco', renderer='software-egl')
    argv, env = calls[0][0][0], calls[0][1]['env']
    assert argv[argv.index('--backend')+1] == 'mujoco'
    assert argv[argv.index('--renderer')+1] == 'software-egl'
    assert 'LIBGL_ALWAYS_SOFTWARE' not in env and '__NV_PRIME_RENDER_OFFLOAD' not in env
    bus.alive = True; bus.data = {'session': 'new', 'state': 'IDLE', 'backend': 'isaac'}
    with pytest.raises(RuntimeError, match='backend differs'): service.poll()
    assert not bus.sent
    assert service.connect({}, python, backend='mujoco') is None
    assert service.replacement['backend']=='mujoco'
    assert bus.sent==[('disconnect',None)]


def test_orphan_build_can_be_cancelled_without_ipc_command(tmp_path):
    import sys,subprocess,time
    from soarm101_lab.real2sim.storage import Bus as RealBus
    root=tmp_path/'repository';script=root/'scripts/real2sim/real2sim_sim.py';script.parent.mkdir(parents=True)
    script.write_text('''
import sys,time
from pathlib import Path
from soarm101_lab.real2sim.storage import Bus
b=Bus(Path(sys.argv[sys.argv.index('--workspace')+1])/'ipc',owner=True)
b.publish(state='BUILDING_ENVIRONMENT',backend='isaac',real_connected=False)
while True:time.sleep(.05)
''')
    workspace=tmp_path/'live';bus=RealBus(workspace/'ipc')
    process=subprocess.Popen([sys.executable,'-B',str(script),'--workspace',str(workspace)])
    service=launcher.RuntimeLauncher(bus,workspace,root)
    try:
        deadline=time.monotonic()+10
        while not (bus.owner_alive() and bus.status().get('state')=='BUILDING_ENVIRONMENT'):
            assert process.poll() is None
            assert time.monotonic()<deadline
            time.sleep(.02)
        assert service.disconnect() is None
        assert service.stopping and service.process is None
        assert process.wait(timeout=10)<0
        service.poll()
        assert not bus.owner_alive() and not service.stopping
        assert not list(bus.root.glob('cmd_*.json'))
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=10)


def test_build_with_connected_robot_cannot_be_force_killed(tmp_path,monkeypatch):
    service,bus,_,_,_=setup(tmp_path,monkeypatch)
    bus.alive=True;bus.data={'session':'new','backend':'isaac','state':'BUILDING_ENVIRONMENT','real_connected':True}
    signal=__import__('unittest.mock',fromlist=['Mock']).Mock()
    monkeypatch.setattr(launcher.os,'kill',signal)
    with pytest.raises(RuntimeError,match='강제 종료'):service.disconnect()
    signal.assert_not_called()


def test_mode_switch_waits_for_disconnect_before_new_worker(tmp_path,monkeypatch):
    service,bus,process,calls,python=setup(tmp_path,monkeypatch)
    bus.alive=True;bus.data={'session':'old','state':'IDLE','backend':'isaac'}
    assert service.connect({'profile':'measured.json'},python,backend='real') is None
    assert bus.sent==[('disconnect',None)] and not calls
    service.poll();assert not calls
    bus.alive=False
    assert service.poll() is None
    assert len(calls)==1 and service.backend=='real'
    argv=calls[0][0][0]
    assert argv[argv.index('--backend')+1]=='real' and '--renderer' not in argv
    assert bus.sent==[('disconnect',None)]
    service.cancel_connect()


def test_missing_osmesa_uses_software_egl_for_live_runtime(tmp_path, monkeypatch):
    service, bus, _, calls, python = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher.ctypes.util, 'find_library', lambda name: None)
    assert launcher.resolve_live_renderer('osmesa') == 'software-egl'
    service.connect({}, python, backend='mujoco', renderer='osmesa')
    argv = calls[0][0][0]
    assert argv[argv.index('--renderer')+1] == 'software-egl'
    service.cancel_connect()


@pytest.mark.parametrize('state,old_renderer', [('ERROR', None), ('PREVIEW', 'osmesa'), ('IDLE', None), ('PREVIEW', None)])
def test_mujoco_failed_or_changed_renderer_restarts_worker(tmp_path, monkeypatch, state, old_renderer):
    service, bus, _, calls, python = setup(tmp_path, monkeypatch)
    bus.alive = True
    bus.data = {'session': 'old', 'backend': 'mujoco', 'state': state, 'renderer': old_renderer}
    service.connect({}, python, backend='mujoco', renderer='software-egl')
    assert bus.sent == [('disconnect', None)] and not calls
    service.poll(); assert not calls
    bus.alive = False; service.poll()
    assert calls[0][0][0][calls[0][0][0].index('--renderer')+1] == 'software-egl'
    service.cancel_connect()
