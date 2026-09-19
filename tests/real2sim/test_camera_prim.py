from types import SimpleNamespace
import pytest
from pxr import Usd, UsdGeom
from soarm101_lab.real2sim.scene import camera_prim


def test_camera_uses_resolved_sensor_paths():
    stage = Usd.Stage.CreateInMemory()
    path = '/World/envs/env_0/Robot/gripper/Camera_R2S_Wrist'
    UsdGeom.Camera.Define(stage, path)
    sensor = SimpleNamespace(cfg=SimpleNamespace(prim_path='/World/envs/env_.*/Robot/gripper/Camera_R2S_Wrist'),
                             _view=SimpleNamespace(prim_paths=[path]))
    assert str(camera_prim(stage, sensor).GetPath()) == path
    stage.RemovePrim(path)
    with pytest.raises(RuntimeError, match='Camera prim missing'):
        camera_prim(stage, sensor)


def test_uninitialized_or_multiple_cameras_are_rejected():
    stage = Usd.Stage.CreateInMemory()
    for view in [None, SimpleNamespace(prim_paths=['/a', '/b'])]:
        with pytest.raises(RuntimeError, match='one initialized camera'):
            camera_prim(stage, SimpleNamespace(_view=view))


def test_generated_camera_config_tracks_pose_after_robot_or_profile_moves(monkeypatch, tmp_path):
    """CameraData must follow the moving prim, rather than retain its reset pose."""
    import sys
    from pathlib import Path
    from soarm101_lab.real2sim.profile import load
    from soarm101_lab.real2sim.scene import configure_environment

    class CameraConfig(SimpleNamespace):
        OffsetCfg = SimpleNamespace

    sim = SimpleNamespace(UsdFileCfg=SimpleNamespace, PinholeCameraCfg=SimpleNamespace)
    monkeypatch.setitem(sys.modules, 'isaaclab', SimpleNamespace(sim=sim))
    monkeypatch.setitem(sys.modules, 'isaaclab.sim', sim)
    monkeypatch.setitem(sys.modules, 'isaaclab.assets', SimpleNamespace(AssetBaseCfg=SimpleNamespace))
    monkeypatch.setitem(sys.modules, 'isaaclab.sensors', SimpleNamespace(CameraCfg=CameraConfig))
    p = load(Path(__file__).resolve().parents[2] / 'docs/real2sim/measurements.example.json')
    cfg = SimpleNamespace(scene=SimpleNamespace(
        robot=SimpleNamespace(init_state=SimpleNamespace()),
        dome_light=SimpleNamespace(spawn=SimpleNamespace())))
    configure_environment(cfg, p, tmp_path / 'workspace.usda')
    for view in ('side', 'wrist'):
        sensor = getattr(cfg.scene, 'camera_' + view + 'view')
        assert sensor.update_latest_camera_pose is True
        assert sensor.offset.convention == 'ros'
    assert '/Robot/gripper/' in cfg.scene.camera_wristview.prim_path
