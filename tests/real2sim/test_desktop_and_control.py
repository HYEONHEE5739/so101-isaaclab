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
    assert (ROOT / "docs/real2sim/measurements.example.json").exists()
    assert (tmp_path / "ipc/ui_heartbeat.json").exists()
    w.template()
    assert '"schema": "so101.real2sim/2"' in w.editor.toPlainText()
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
    r.capture_requests = []
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


def test_stop_cancels_pending_capture_requests():
    r = runtime_stub()
    r.capture_requests = [{"id": "a"}, {"id": "b"}]
    r.stop()
    assert not r.capture_requests
    assert r.bus.reply.call_count == 2


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

    p = profile.load(Path(__file__).resolve().parents[2] / "docs/real2sim/measurements.example.json")
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
