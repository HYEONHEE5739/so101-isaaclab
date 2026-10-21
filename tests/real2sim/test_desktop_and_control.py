import os
from unittest.mock import Mock
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from soarm101_lab.real2sim import hardware
from soarm101_lab.real2sim.core import JOINTS
from soarm101_lab.real2sim.runtime import Runtime


@pytest.fixture(autouse=True)
def isolated_workflow_registry(tmp_path,monkeypatch):
    from soarm101_lab.workflow import ui
    monkeypatch.setattr(ui,'ROOT',tmp_path)


def test_qt_startup_layout_and_heartbeat(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.real2sim.ui import Window, ROOT

    app = QApplication.instance() or QApplication([])
    w = Window(tmp_path)
    w.show()
    app.processEvents()
    w.refresh()
    assert list(w.views) == ["real_side", "sim_side", "real_wrist", "sim_wrist"]
    assert (ROOT / "tests/real2sim/fixtures/partial_profile.json").exists()
    assert not (tmp_path / "ipc/ui_heartbeat.json").exists()
    assert w.capture_start.isEnabled() and w.live_backend.isEnabled()
    from soarm101_lab.real2sim.profile import load
    assert not w.fields['profile'].text()
    with pytest.raises(ValueError, match='Profile'):
        w.profile_path()
    w.selected(ROOT / 'tests/real2sim/fixtures/partial_profile.json')
    assert load(w.profile_path())['schema'] == 'so101.real2sim/2'
    assert not hasattr(w, 'editor')
    assert w.tabs.count() == 4
    assert "model" not in w.fields
    assert not hasattr(w, "measured")
    assert not hasattr(w, "initial_gpt")
    assert not hasattr(w, "agent_loop")
    assert not hasattr(w, 'motion_fields') and not hasattr(w, 'joint_status')
    w.close()
    app.processEvents()


def test_removed_guard_settings_are_not_loaded_sent_or_saved(tmp_path,monkeypatch):
    import json
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.real2sim.ui import Window, ROOT
    from soarm101_lab.workflow import ui
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    (tmp_path/'ui_settings.json').write_text(json.dumps({'leader_port':'/fake/leader',
        'motion_guard':{'startup_deg':1,'velocity_deg_s':2,'tracking_deg':3}}))
    app=QApplication.instance() or QApplication([]);w=Window(tmp_path)
    try:
        assert w.fields['leader_port'].text()=='/fake/leader'
        assert not hasattr(w,'motion_fields') and not hasattr(w,'joint_status')
        w.fields['profile'].setText(str(ROOT/'tests/real2sim/fixtures/partial_profile.json'))
        w.runtime_launcher.connect=Mock(return_value='fake_request')
        w.runtime_launcher.disconnect=Mock(return_value=None)
        w.connect_hw(backend='real')
        payload=w.runtime_launcher.connect.call_args.args[0]
        assert 'motion_guard' not in payload
        assert 'precision_options' in payload
        assert 'motion_guard' not in json.loads((tmp_path/'ui_settings.json').read_text())
        assert w.runtime_launcher.process is None
    finally:w.close();app.processEvents()


def runtime_stub():
    r = Runtime.__new__(Runtime)
    r.follower = Mock()
    r.leader = Mock()
    r.running = False
    r.activated = False
    r.limits = np.array([[-2.0, 2.0]] * 6)
    r.follower.limits = {n: [-180, 180] for n in JOINTS}
    r.env = Mock()
    r.bus = Mock()
    real = {n: 0.0 for n in JOINTS}
    r.leader.read.return_value = (real.copy(), {})
    r.follower.read.return_value = (real.copy(), {})
    return r


def test_capture_ui_close_disconnects_worker(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.real2sim.ui import Window

    app = QApplication.instance() or QApplication([])
    window = Window(tmp_path)
    window.runtime_enabled = True
    window.requested_backend = 'real'
    window.runtime_launcher.disconnect = Mock()
    window.bus.send = Mock()
    window.close()
    app.processEvents()
    window.runtime_launcher.disconnect.assert_called_once()
    window.bus.send.assert_not_called()


def test_start_allows_valid_goal_without_five_degree_guard():
    r = runtime_stub()
    r.leader.read.return_value[0]["elbow_flex"] = 140
    r.start()
    assert r.running
    r.follower.activate.assert_called_once(); r.follower.send.assert_not_called()
    r.stop(); assert not r.running


def test_follower_boundary_rejects_native_targets_outside_motor_range():
    f = hardware.Follower.__new__(hardware.Follower)
    f.device = Mock(); f.limits = {n: [-1, 1] for n in JOINTS}
    with pytest.raises(ValueError,match='motor range'): f.send(dict.fromkeys(JOINTS,200.))
    f.device.send_action.assert_not_called()


def test_follower_mismatch_never_configures_or_calibrates(monkeypatch):
    device = Mock()
    device.is_calibrated = False
    device.bus.is_connected = True
    f = hardware.Follower.__new__(hardware.Follower)
    f.device = device
    f.allow_calibration = False
    with pytest.raises(RuntimeError, match="mismatch"):
        f.connect()
    device.configure.assert_not_called()
    device.calibrate.assert_not_called()
    device.send_action.assert_not_called()


def test_leader_missing_file_never_auto_calibrates(monkeypatch):
    import sys, types

    class FakeLeader:
        def __init__(self, env, cfg):
            self.calibrate()

        def calibrate(self):
            raise AssertionError("Parent auto calibration must never run")

    fake = types.ModuleType("soarm101_lab.devices")
    fake.SO101Leader = FakeLeader
    fake.SO101LeaderCfg = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "soarm101_lab.devices", fake)
    with pytest.raises(RuntimeError, match="automatic calibration is disabled"):
        hardware.Leader(None, "fake", "missing.json")


@pytest.fixture
def actual_leader_bus(monkeypatch):
    from soarm101_lab.devices.lerobot import so101_leader

    bus = Mock()
    bus.is_connected = False
    bus.is_calibrated = True
    bus.model_resolution_table = {"sts3215": 4096}
    bus.apply_drive_mode = False
    bus.connect.side_effect = lambda: setattr(bus, "is_connected", True)
    bus.sync_read.return_value = {n: 2048 for n in JOINTS}
    bus._normalize.side_effect = lambda raw_by_id: {i: 0.0 for i in raw_by_id}

    def factory(**kwargs):
        bus.motors = kwargs["motors"]
        bus.calibration = kwargs["calibration"]
        return bus

    monkeypatch.setattr(so101_leader, "FeetechMotorsBus", factory)
    monkeypatch.setattr(so101_leader.SO101Leader, "configure", Mock())
    return so101_leader, bus


def test_actual_strict_leader_constructs_and_connects_read_only(actual_leader_bus):
    import inspect

    module, bus = actual_leader_bus
    assert not inspect.isabstract(module.SO101Leader)
    leader = hardware.Leader(None, "/fake/leader", "so101_leader.json", use_precision=False)
    assert isinstance(leader.device, module.SO101Leader)
    bus.connect.assert_not_called()
    leader.connect()
    native, interval = leader.read()
    assert native == dict.fromkeys(JOINTS, 0.0)
    assert interval["end"]["monotonic_ns"] >= interval["start"]["monotonic_ns"]
    bus.sync_read.assert_called_once_with("Present_Position", normalize=False)
    bus.write.assert_not_called()
    bus.disable_torque.assert_not_called()
    bus.configure_motors.assert_not_called()
    module.SO101Leader.configure.assert_not_called()


def test_actual_leader_advance_uses_raw_feedback_and_propagates_failed_reads(actual_leader_bus):
    module, bus = actual_leader_bus
    device = module.SO101Leader(module.SO101LeaderCfg(use_precision=False))
    bus.is_connected = True
    target = device.advance()
    assert tuple(target.shape) == (6,)
    assert device.last_raw == dict.fromkeys(JOINTS, 2048)
    bus.sync_read.assert_called_once_with("Present_Position", normalize=False)
    bus.sync_read.side_effect = TimeoutError("fake read failure")
    with pytest.raises(TimeoutError, match="fake read failure"):
        device.advance()
    bus.sync_read.side_effect = None
    bus.sync_read.return_value = {}
    with pytest.raises(KeyError):
        device.advance()
    bus.write.assert_not_called()


def test_actual_strict_leader_missing_calibration_still_blocks_implicit_calibration(actual_leader_bus, tmp_path):
    _, bus = actual_leader_bus
    with pytest.raises(RuntimeError, match="automatic calibration is disabled"):
        hardware.Leader(None, "/fake/leader", str(tmp_path / "missing.json"), use_precision=False)
    bus.connect.assert_not_called()
    bus.write.assert_not_called()




def test_hold_readback_precedes_torque_enable():
    f = hardware.Follower.__new__(hardware.Follower)
    f.device = Mock()
    f.device.is_calibrated = True
    f.device.bus.sync_read.side_effect = [{"shoulder_pan": 2000}, {"shoulder_pan": 1000}]
    with pytest.raises(RuntimeError, match="readback"):
        f.activate()
    f.device.configure.assert_not_called()


def test_scene_apply_rolls_back_on_api_failure(monkeypatch):
    import copy
    from pathlib import Path
    from soarm101_lab.real2sim import profile, runtime

    p = profile.load(Path(__file__).resolve().parents[2] / "tests/real2sim/fixtures/partial_profile.json")
    r = runtime_stub()
    r.profile = p
    changed = copy.deepcopy(p)
    changed["workspace"]["T_world"][0][3] = 0.05
    calls = []

    def apply(env, candidate):
        calls.append(candidate)
        if candidate is changed:
            raise RuntimeError("Sensor API failure")

    monkeypatch.setattr(runtime, "apply_profile", apply)
    with pytest.raises(RuntimeError):
        r.apply(changed)
    assert calls == [changed, p]
    assert r.profile == p


def test_capture_and_real_sim_share_scrollable_right_settings(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication, QPushButton
    from PyQt6.QtCore import QPoint
    from soarm101_lab.real2sim.ui import Window
    from soarm101_lab.workflow import ui
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    app = QApplication.instance() or QApplication([])
    w = Window(tmp_path / 'live')
    w.resize(1280, 720); w.show(); app.processEvents()
    try:
        assert not any(b.text() == '연결 설정' for b in w.findChildren(QPushButton))
        fields = dict(w.fields)
        for index in (0, 1, 0):
            w.tabs.setCurrentIndex(index); app.processEvents()
            assert w.settings_scroll.isVisible() and w.log.isVisible()
            assert w.fields == fields
            assert w.settings_scroll.mapTo(w, QPoint()).x() >= w.tabs.mapTo(w, QPoint()).x()+w.tabs.width()
            assert w.log.mapTo(w, QPoint()).y() >= w.content.mapTo(w, QPoint()).y()+w.content.height()
            bar = w.settings_scroll.verticalScrollBar()
            assert bar.maximum() > 0
            bar.setValue(bar.maximum()); app.processEvents()
            target = w.capture_scene_state if index == 0 else w.connection_group.findChildren(QPushButton)[-1]
            assert w.settings_scroll.viewport().rect().intersects(target.rect().translated(target.mapTo(w.settings_scroll.viewport(), QPoint())))
        assert not w.runtime_enabled and w.runtime_launcher.process is None
        w.tabs.setCurrentIndex(2); app.processEvents()
        assert w.settings_scroll.isHidden()
    finally:
        w.close(); app.processEvents()


def test_workspace_connection_does_not_replace_capture_profile(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.real2sim.ui import Window, ROOT
    from soarm101_lab.real2sim.workspaces import package
    app = QApplication.instance() or QApplication([])
    window = Window(tmp_path)
    camera_profile = ROOT/'tests/real2sim/fixtures/partial_profile.json'
    window.selected(camera_profile)
    scene = tmp_path/'builder_workspace'
    window.workflow_panel.workspace_path = str(scene)
    window.live_workspace.setChecked(True)
    monkeypatch.setattr(package, 'load', lambda p: {'root':scene})
    monkeypatch.setattr(window.workflow_panel.task_selector,'selection',lambda **kwargs:[])
    window.runtime_launcher.connect = Mock(return_value='request')
    window.connect_hw(backend='isaac')
    payload = window.runtime_launcher.connect.call_args.args[0]
    assert payload['workspace'] == str(scene)
    assert payload['profile'] == str(scene/'profile.json')
    assert window.profile_path() == str(camera_profile)
    window.runtime_launcher.connect.reset_mock()
    window.start_capture()
    payload = window.runtime_launcher.connect.call_args.args[0]
    assert 'workspace' not in payload
    assert payload['profile'] == str(camera_profile)
    window.runtime_enabled = False
    window.close()
    app.processEvents()


def test_follower_motion_diagnostics_read_only_and_preserve_settings():
    from soarm101_lab.so101_joint_mapping import JOINTS
    f=hardware.Follower.__new__(hardware.Follower)
    f.device=Mock();f.metadata={}
    registers={'Operating_Mode':0,'Goal_Velocity':100,'Acceleration':254,'Phase':8}
    f.device.bus.sync_read.side_effect=lambda name,normalize:{n:registers[name] for n in JOINTS}
    f.read_motion_settings('connect')
    assert f.metadata['motion_settings']['registers']['Goal_Velocity']==dict.fromkeys(JOINTS,100)
    f.device.bus.write.assert_not_called();f.device.bus.sync_write.assert_not_called()
    f.device.configure.assert_not_called()
    f.device.bus.sync_read.side_effect=RuntimeError('diagnostic register unavailable')
    f.read_motion_settings('connect')
    assert 'read_error' in f.metadata['motion_settings']['registers']['Goal_Velocity']
    f.device.bus.write.assert_not_called()
