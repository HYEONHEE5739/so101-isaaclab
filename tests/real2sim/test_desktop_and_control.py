import os
from unittest.mock import Mock
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from soarm101_lab.real2sim import hardware
from soarm101_lab.real2sim.core import JOINTS
from soarm101_lab.real2sim.runtime import Runtime


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
    w.close()
    app.processEvents()


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


def test_start_accepts_pose_difference_and_outside_limits():
    r = runtime_stub()
    r.leader.read.return_value[0]["elbow_flex"] = 140
    r.limits[0] = [-0.01, -0.001]
    r.start()
    assert r.running
    r.follower.activate.assert_called_once()
    r.stop()
    assert not r.running


def test_direct_targets_are_not_rejected_or_clamped():
    from soarm101_lab.real2sim.core import to_sim
    target = {n: 200.0 for n in JOINTS}
    r = runtime_stub()
    assert np.allclose(r.check_target(target), list(to_sim(target).values()))
    f = hardware.Follower.__new__(hardware.Follower)
    f.device = Mock()
    f.device.send_action.return_value = {n + '.pos': v for n, v in target.items()}
    f.limits = {n: [-1, 1] for n in JOINTS}
    sent, _ = f.send(target)
    assert sent == target
    f.device.send_action.assert_called_once_with({n + '.pos': v for n, v in target.items()})


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
    from soarm101_lab.workflow.artifacts import Registry
    monkeypatch.setattr(ui, 'Registry', lambda: Registry(tmp_path / 'registry'))
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
    window.workflow_panel.workspace.setText(str(scene))
    window.live_workspace.setChecked(True)
    monkeypatch.setattr(package, 'load', lambda p: {'root':scene})
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
