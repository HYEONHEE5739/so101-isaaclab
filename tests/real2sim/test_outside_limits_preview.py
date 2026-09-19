
from unittest.mock import Mock
import numpy as np
from soarm101_lab.real2sim.runtime import Runtime
from soarm101_lab.real2sim.core import JOINTS


def test_outside_measured_pose_keeps_images_but_rejects_capture(tmp_path, monkeypatch):
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
    r.sim_history = Mock()
    r.step = 0
    r.profile = {'workspace': {'T_world': []}, 'objects': {}}
    r.hardware_identity = 'test'
    r.leader.metadata = r.follower.metadata = {}
    r.capture_requests = [{'id': 'capture'}]
    r.pending = None
    r.bus = Mock()
    r.bus.root = tmp_path
    r.last_preview = 0
    monkeypatch.setattr(module, 'assess_pair', lambda *a: {'eligible_for_calibration': True, 'reasons': []})
    r.tick()
    assert len(list(tmp_path.glob('*.jpg'))) == 4
    assert not r.snapshot_data['quality']['eligible_for_calibration']
    assert 'Measured pose' in r.snapshot_data['quality']['reasons'][0]
    assert r.bus.reply.call_args.args[:2] == ('capture', False)
    r.follower.send.assert_not_called()
