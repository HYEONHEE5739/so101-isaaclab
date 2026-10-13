
from unittest.mock import Mock
import numpy as np
from soarm101_lab.real2sim.runtime import Runtime
from soarm101_lab.real2sim.core import JOINTS


def test_mujoco_bounds_both_commands_without_stopping_and_rejects_nonfinite(tmp_path):
    from types import SimpleNamespace
    from soarm101_lab.real2sim.core import from_sim, to_sim, stamp
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    backend = MujocoBackend.__new__(MujocoBackend)
    backend.limits = np.array([[-2., 2.]] * 3 + [[-1.65806, 1.65806], [-2., 2.], [0., 1.74533]])
    r = Runtime(SimpleNamespace(workspace=str(tmp_path), backend='mujoco'), None)
    r.env = SimpleNamespace(limits=backend.limits, _target=backend._target, step=Mock())
    r.leader = Mock(); r.follower = Mock()
    interval = {'start': stamp(), 'end': stamp()}
    measured = dict(zip(JOINTS, [0.] * 5 + [30.]))
    r.follower.read.return_value = measured, interval
    r.follower.send.side_effect = lambda target: (dict(target), interval)
    r.leader.metadata = r.follower.metadata = {}
    r.follower.limits={n:[0,100] if n=='gripper' else [-180,180] for n in JOINTS}
    r.leader.read.return_value = measured.copy(), interval
    r.limits=backend.limits
    r.profile = {'workspace': {'T_world': np.eye(4).tolist()}, 'objects': {}}
    r.hardware_identity = 'fixture'
    rgb = np.zeros((8, 8, 3), np.uint8)
    r.snapshot = lambda: ({'sim_side': rgb, 'sim_wrist': rgb}, interval,
        dict(zip(JOINTS, r.env.step.call_args.args[0])), {}, {}, np.eye(4).tolist())
    r.start()
    try:
        requested = dict(zip(JOINTS, [0., 0., 0., 1.672447, 0., .5]))
        r.leader.read.return_value = from_sim(requested), interval
        r.tick()
        sent = r.follower.send.call_args.args[0]
        model_target = np.array(list(to_sim(sent).values()))
        np.testing.assert_allclose(r.env.step.call_args.args[0], model_target, atol=1e-12)
        assert r.running and r.state == 'RUNNING'
        assert 1.65805 < model_target[3] <= 1.65806
        assert r.target_limits['wrist_flex']['requested_rad'] == requested['wrist_flex']
        assert '목표 제한' in r.message
        # Returning the leader into range resumes tracking without another Start.
        interval.update(start=stamp(), end=stamp())
        requested['wrist_flex'] = 1.6
        r.leader.read.return_value = from_sim(requested), interval
        r.tick()
        assert r.running and not r.target_limits
        assert abs(r.env.step.call_args.args[0][3] - 1.6) < 1e-12
        # Invalid input still stops before any physical command is sent.
        interval.update(start=stamp(), end=stamp())
        native = from_sim(requested); native['wrist_flex'] = float('nan')
        r.leader.read.return_value = native, interval
        calls = r.follower.send.call_count
        r.tick()
        assert not r.running and r.state == 'STOPPED'
        assert r.follower.send.call_count == calls and 'finite' in r.message
    finally:
        r.writer.shutdown(); r.bus.close()


def test_outside_measured_pose_keeps_preview_and_reports_invalid_sim_pose(tmp_path, monkeypatch):
    import soarm101_lab.real2sim.runtime as module
    r = Runtime.__new__(Runtime)
    r.env = Mock()
    r.env.scene = {'camera_sideview': Mock(), 'camera_wristview': Mock()}
    r.follower = Mock()
    r.leader = Mock()
    native = {n: 0.0 for n in JOINTS}
    interval = {'start': {'monotonic_ns': 1}, 'end': {'monotonic_ns': 2}}
    r.follower.read.return_value = (native, interval)
    r.leader.read.return_value = (native, interval)
    r.running = False
    r.set_measured_pose = Mock(side_effect=ValueError('Outside Sim limits'))
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    r.snapshot = Mock(return_value=({'sim_side': rgb, 'sim_wrist': rgb}, interval, native, {}, {}, []))
    r.cameras = {n: Mock() for n in ('real_side', 'real_wrist')}
    for c in r.cameras.values():
        c.nearest.return_value = {'rgb': rgb}
        c.metadata = {}
    r.real_history = Mock()
    r.real_history.history = []
    r.sim_history = Mock()
    r.step = 0
    r.profile = {'workspace': {'T_world': []}, 'objects': {}}
    r.hardware_identity = 'test'
    r.leader.metadata = r.follower.metadata = {}
    r.follower.limits={n:[0,100] if n=='gripper' else [-180,180] for n in JOINTS}
    r.pending = None
    r.bus = Mock()
    r.bus.root = tmp_path
    r.last_preview = 0
    monkeypatch.setattr(module, 'assess_pair', lambda *a: {'eligible_for_calibration': True, 'reasons': []})
    r.tick()
    assert len(list(tmp_path.glob('*.jpg'))) == 4
    assert not r.snapshot_data['quality']['eligible_for_calibration']
    assert 'Measured pose' in r.snapshot_data['quality']['reasons'][0]
    r.bus.reply.assert_not_called()
    r.follower.send.assert_not_called()
