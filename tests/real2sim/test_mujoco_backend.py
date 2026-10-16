"""Physical integration checks, independent URDF FK and invalid-contract checks."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soarm101_lab.simulation.mjcf import ROOT, build_scene, transform
from soarm101_lab.so101_joint_mapping import JOINTS


def scene_inputs():
    from soarm101_lab.real2sim.workspaces.package import default_task
    p = json.loads((ROOT / 'tests/real2sim/fixtures/scene_profile.json').read_text())
    return {'profile': p, 'task': default_task(p), 'root': ROOT}


def test_identity_sim_uses_original_urdf_without_mutating_workspace(tmp_path):
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    inputs=scene_inputs()
    inputs['profile']['simulation_joint_offsets_rad']={'value':[.04,0,0,0,0,0],
        'provenance':'estimated','evidence':['synthetic fixture']}
    original=copy.deepcopy(inputs)
    with MujocoBackend(tmp_path/'original_frame',inputs,identity_sim=True) as env:
        q=env.home.copy();q[0]=.4
        env.set_joint_state(q)
        expected=np.linalg.inv(transform(inputs['profile']['workspace']['T_world'])) @ transform(inputs['profile']['robot_base']['T_world']) @ urdf_fk(ROOT/'assets/SO101/urdf/so101_isaaclab.urdf',dict(zip(JOINTS,q)))
        np.testing.assert_allclose(env.get_eef_pose(),expected,atol=1e-9)
        assert env.metadata['simulation_joint_frame']=='urdf'
        for _ in range(20):assert np.isfinite(env.step(q)['joint_pos']).all()
    assert inputs==original


@pytest.fixture
def env(tmp_path):
    from soarm101_lab.simulation.mujoco_backend import configure_rendering
    configure_rendering('none')
    pytest.importorskip('mujoco')
    # tests must set MUJOCO_GL before import; renderer isolation is checked in a subprocess.
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    with MujocoBackend(tmp_path / 'scene', scene_inputs()) as backend:
        yield backend


def urdf_fk(urdf, q):
    """Independent tree traversal; does not call MJCF origin/pose conversion helpers."""
    root = ET.parse(urdf).getroot()
    frames = {'base': np.eye(4)}
    pending = list(root.findall('joint'))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find('parent').get('link')
            if parent not in frames:
                continue
            o = joint.find('origin')
            t = np.eye(4)
            t[:3, 3] = np.fromstring(o.get('xyz'), sep=' ')
            r, p, y = np.fromstring(o.get('rpy'), sep=' ')
            rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
            ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
            rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
            t[:3, :3] = rz @ ry @ rx
            motion = np.eye(4)
            if joint.get('type') == 'revolute':
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                motion[:3, :3] = Rotation.from_rotvec(axis * q[joint.get('name')]).as_matrix()
            frames[joint.find('child').get('link')] = frames[parent] @ t @ motion
            pending.remove(joint)
            progressed = True
        assert progressed, 'Broken URDF tree'
    return frames['tool0']


def test_compile_fk_camera_and_import_boundary(env):
    assert env.model.nu == 6 and env.model.ncam == 1
    assert env.model.nq == 13 and env.model.nv == 12
    assert env.renderer is None
    rng = np.random.default_rng(123)
    base = np.array(env.workspace['profile']['robot_base']['T_world'])
    errors = []
    for q in [env.home, *rng.uniform(env.limits[:, 0], env.limits[:, 1], size=(20, 6))]:
        env.data.qpos[env.q_indices] = q
        env.mj.mj_forward(env.model, env.data)
        expected = env.workspace_from_world @ base @ urdf_fk(ROOT / 'assets/SO101/urdf/so101_isaaclab.urdf', dict(zip(JOINTS, q)))
        errors.append(np.max(np.abs(env.get_eef_pose() - expected)))
    assert max(errors) < 1e-9
    camera = env.model.camera('side')
    expected_cam = np.array(env.workspace['profile']['cameras']['side']['T_parent_camera'])
    actual_cam = Rotation.from_quat(camera.quat[[1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(actual_cam, expected_cam[:3, :3] @ np.diag([1, -1, -1]), atol=1e-9)
    np.testing.assert_allclose(camera.pos, expected_cam[:3, 3], atol=1e-9)
    assert not any(n.split('.')[0] in ('isaaclab', 'omni', 'pxr') for n in sys.modules)


def test_full_workspace_preserves_objects_reset_camera_and_physics_roles(tmp_path, monkeypatch):
    from soarm101_lab.simulation import mjcf
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend, configure_rendering
    from soarm101_lab.real2sim.workspaces.reset import WorkspaceRandomizer
    from soarm101_lab.real2sim.workspaces.physics import compile_physics
    configure_rendering('none')
    w, urdf, source = mjcf.load_inputs(scene_inputs())
    w['root'] = ROOT
    monkeypatch.setattr(mjcf, 'load_inputs', lambda *args: (w, urdf, source))
    with MujocoBackend(tmp_path / 'full', full_workspace=True) as full:
        assert full.model.nq == 6 + 7 * len(w['task']['dynamic'])
        assert full.model.ncam == 3
        for pad in full.metadata['contact_pads']['pads']:
            geom = full.model.geom(pad['link']+'_contact_pad')
            np.testing.assert_allclose(geom.size*2, pad['dimensions_m'])
            np.testing.assert_allclose(geom.pos, np.asarray(pad['T_link'])[:3, 3])
            assert geom.contype == 2 and geom.conaffinity == 1
        state = full.reset(seed=123)
        reference = WorkspaceRandomizer(w, seed=123).sample()
        assert set(state['object_poses']) == set(w['task']['dynamic']) | set(w['task']['targets'])
        for name, xyz in reference['cube_positions'].items():
            np.testing.assert_allclose(full.get_object_pose(name)[:3, 3], xyz, atol=1e-10)
        for name, visual in w['task']['targets'].items():
            np.testing.assert_allclose(full.get_object_pose(name), w['profile']['objects'][visual]['T_workspace'], atol=1e-10)
        for _ in range(50): full.step(full.home)
        wrist = full.model.camera('wrist').id
        gripper = full.model.body('robot_gripper').id
        pose = np.eye(4); pose[:3, :3] = full.data.xmat[gripper].reshape(3, 3); pose[:3, 3] = full.data.xpos[gripper]
        expected = pose @ np.array(w['profile']['cameras']['wrist']['T_parent_camera'])
        np.testing.assert_allclose(full.data.cam_xpos[wrist], expected[:3, 3], atol=1e-9)
        np.testing.assert_allclose(full.data.cam_xmat[wrist].reshape(3, 3), expected[:3, :3] @ np.diag([1, -1, -1]), atol=1e-9)
        from soarm101_lab.real2sim.runtime import Runtime
        live = Runtime.__new__(Runtime)
        live.backend, live.env, live.step, live.profile = 'mujoco', full, 0, w['profile']
        monkeypatch.setattr(full, 'get_rgb', lambda view: np.zeros((2, 2, 3), dtype=np.uint8))
        _, _, joints, cameras, links, base = live.snapshot()
        np.testing.assert_allclose(links['tool0'], urdf_fk(urdf, joints), atol=1e-9)
        np.testing.assert_allclose(np.asarray(cameras['wrist']['T_world_optical']), expected, atol=1e-9)
        assert full.metadata['physics_roles'] == compile_physics(w['profile'], w['task'])['objects']
        assert not any(n.split('.')[0] in ('isaaclab', 'omni', 'pxr') for n in sys.modules)


def test_step_settling_restore_and_target_validation(env):
    initial = env.reset(seed=42)
    snapshot = env.get_initial_state()
    target = env.home.copy()
    target[0] += .1
    for _ in range(150):
        state = env.step(dict(zip(JOINTS, target)))
    assert state['timestamp_s'] == pytest.approx(3.)
    assert np.isfinite(state['joint_pos']).all()
    assert abs(state['joint_pos'][0] - target[0]) < .03
    assert env.success_details()['speed_m_s'] < .03
    assert not env.is_success()
    env.restore_state(snapshot)
    np.testing.assert_allclose(env.get_state()['joint_pos'], initial['joint_pos'])
    np.testing.assert_allclose(env.get_object_pose(env.metadata['pick']), initial['object_poses'][env.metadata['pick']])
    env.reset(seed=42)
    np.testing.assert_allclose(env.get_object_pose(env.metadata['pick']), initial['object_poses'][env.metadata['pick']])
    for bad in ([0] * 5, [float('nan')] * 6, [100] * 6, {'shoulder_pan': 0}):
        with pytest.raises(ValueError):
            env.step(bad)
    with pytest.raises(ValueError):
        env.restore_state({**snapshot, 'model_sha256': 'wrong-engine-model'})
    with pytest.raises(ValueError):
        env.set_object_pose(env.metadata['target'], np.eye(4))
    with pytest.raises(RuntimeError, match='Renderer disabled'):
        env.get_rgb()


def test_open_bin_success_height_walls_and_speed(env):
    pose = env.get_object_pose(env.metadata['target']).copy()
    pose[:3, 3] += pose[:3, :3] @ np.array([0, 0, -.015])
    env.set_object_pose(env.metadata['pick'], pose)
    for _ in range(100):
        env.step(env.home)
    assert env.is_success(), env.success_details()
    env.data.qvel[env.object_v] = .2
    env.mj.mj_forward(env.model, env.data)
    assert not env.success_details()['speed_ok']
    outside = pose.copy()
    outside[:3, 3] += pose[:3, :3] @ np.array([.06, 0, 0])
    env.set_object_pose(env.metadata['pick'], outside)
    assert not env.success_details()['wall_ok']
    above = pose.copy()
    above[:3, 3] += pose[:3, :3] @ np.array([0, 0, .1])
    env.set_object_pose(env.metadata['pick'], above)
    assert not env.success_details()['height_ok']


def test_output_and_transform_guards(tmp_path):
    from soarm101_lab.simulation.mjcf import load_inputs
    with pytest.raises(ValueError, match='explicit Workspace'):
        load_inputs()
    with pytest.raises(ValueError, match='explicit Workspace'):
        build_scene(tmp_path / 'no_input')
    assert not (tmp_path / 'no_input').exists()
    with pytest.raises(ValueError):
        transform(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        build_scene(tmp_path, width=0)
    build_scene(tmp_path / 'valid', scene_inputs())
    with pytest.raises(FileExistsError):
        build_scene(tmp_path / 'valid', scene_inputs())
    info = json.loads((tmp_path / 'valid/scene_metadata.json').read_text())
    assert info['physics_device'] == 'cpu' and len(info['asset_sha256']) == 14


def test_workspace_frame_roundtrip_with_rotated_origin(tmp_path, monkeypatch):
    from soarm101_lab.simulation import mjcf
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend, configure_rendering
    configure_rendering('none')
    pytest.importorskip('mujoco')
    w, urdf, info = mjcf.load_inputs(scene_inputs())
    w = copy.deepcopy(w)
    frame = np.eye(4)
    frame[:3, :3] = Rotation.from_euler('z', .3).as_matrix()
    frame[:3, 3] = [.3, -.1, .2]
    w['profile']['workspace']['T_world'] = frame.tolist()
    monkeypatch.setattr(mjcf, 'load_inputs', lambda _: (w, urdf, info))
    with MujocoBackend(tmp_path / 'rotated') as env:
        target = np.eye(4)
        target[:3, :3] = Rotation.from_euler('xyz', [.2, -.1, .5]).as_matrix()
        target[:3, 3] = [.03, -.02, .05]
        env.set_object_pose(env.metadata['pick'], target)
        np.testing.assert_allclose(env.get_object_pose(env.metadata['pick']), target, atol=1e-12)
        actual_world = env.data.xpos[env.model.body(env.metadata['pick']).id]
        np.testing.assert_allclose(actual_world, (frame @ target)[:3, 3], atol=1e-12)


def test_rgb_software_camera_projection(tmp_path):
    if not sys.platform.startswith('linux'):
        pytest.skip('software EGL is Linux-specific')
    from soarm101_lab.simulation.mujoco_backend import configure_rendering
    configure_rendering('none')
    pytest.importorskip('mujoco')
    code = '''
import sys
import numpy as np
from soarm101_lab.simulation.mujoco_backend import MujocoBackend
from soarm101_lab.real2sim.workspaces.package import ROOT
import json
p = json.loads((ROOT / 'tests/real2sim/fixtures/scene_profile.json').read_text())
with MujocoBackend(sys.argv[1], {'profile': p}, render_backend='software-egl') as env:
    image = env.get_rgb()
    assert image.shape == (240, 320, 3) and image.dtype == np.uint8
    assert image.std() > 1
    assert env.metadata['render_context']['software']
    image = image.astype(float)
    mask = (image[:,:,0] > 30) & (image[:,:,0] > 2*image[:,:,1]) & (image[:,:,0] > 2*image[:,:,2])
    y, x = np.nonzero(mask)
    assert len(x) > 10, 'Red cube not visible'
    pose = env.get_object_pose(env.metadata['pick'])
    world = env.world_from_workspace @ pose
    camera = np.array(env.workspace['profile']['cameras']['side']['T_parent_camera'])
    optical = np.linalg.inv(camera) @ world
    k = np.array(env.metadata['camera']['K'])
    pixel = k @ optical[:3,3]
    pixel = pixel[:2] / pixel[2]
    assert np.linalg.norm(pixel - [x.mean(), y.mean()]) < 4, (pixel, x.mean(), y.mean())
    assert not any(n.split('.')[0] in ('isaaclab','omni','pxr') for n in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path / 'rgb')],
                            env={**os.environ, 'PYTHONPATH': str(ROOT / 'source/soarm101_lab')},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_clean_worker_imports_and_render_mode_isolation(tmp_path):
    from soarm101_lab.simulation.mujoco_backend import configure_rendering
    configure_rendering('none')
    pytest.importorskip('mujoco')
    # A clean worker is stronger evidence than the pytest process's existing module list.
    code = '''
import sys
from soarm101_lab.simulation.mujoco_backend import MujocoBackend
from soarm101_lab.real2sim.workspaces.package import ROOT
import json
p = json.loads((ROOT / 'tests/real2sim/fixtures/scene_profile.json').read_text())
with MujocoBackend(sys.argv[1], {'profile': p}) as env:
    env.step(env.home)
    assert not any(n.split('.')[0] in ('isaaclab', 'omni', 'pxr') for n in sys.modules)
    try:
        MujocoBackend(sys.argv[1] + '_other', {'profile': p}, render_backend='egl')
    except RuntimeError:
        pass
    else:
        raise AssertionError('mode change after import was accepted')
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path / 'worker')],
                            env={**os.environ, 'PYTHONPATH': str(ROOT / 'source/soarm101_lab')},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_grasp_lift_and_hold_uses_real_contacts(tmp_path):
    from soarm101_lab.simulation.mjcf import load_inputs
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend, configure_rendering
    configure_rendering('none')
    w, _, _ = load_inputs(scene_inputs()); w['root'] = ROOT
    with MujocoBackend(tmp_path / 'grasp', w, full_workspace=True, timestep=1/300, decimation=10) as e:
        q = e.home.copy(); q[5] = .1
        e.set_joint_state(q)
        pads = [e.model.geom(n + '_contact_pad').id for n in ('gripper', 'jaw')]
        pose = np.eye(4)
        pose[:3, 3] = e.data.geom_xpos[pads].mean(axis=0)
        pose[:3, :3] = e.data.geom_xmat[pads[0]].reshape(3, 3)
        e.set_object_pose(e.metadata['pick'], e.workspace_from_world @ pose)
        initial = e.get_object_pose(e.metadata['pick'])[:3, 3].copy()
        q[5] = 0
        for i in range(150):
            if i >= 30: q[3] = e.home[3] - .3 * min(1, (i-30)/30)
            e.step(q)
        assert e.finger_contacts() == [True, True]
        assert e.get_object_pose(e.metadata['pick'])[2, 3] > initial[2] + .015
        # Object has no weld/equality attachment to the fingers.
        assert e.model.neq == 0
        assert np.max(np.abs(e.model.actuator_forcerange)) == 30


def test_legacy_physics_remains_replayable_and_cannot_mix_with_new_profile(tmp_path):
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend, configure_rendering
    configure_rendering('none')
    with MujocoBackend(tmp_path / 'legacy', scene_inputs(), physics_profile='legacy_urdf') as old:
        initial = old.get_initial_state()
        old.step(old.home)
        expected = old.get_state()['joint_pos']
        assert np.max(np.abs(old.model.actuator_forcerange)) == 10
    with MujocoBackend(tmp_path / 'replay', scene_inputs(), physics_profile='legacy_urdf') as replay:
        replay.restore_state(initial)
        replay.step(replay.home)
        np.testing.assert_allclose(replay.get_state()['joint_pos'], expected, atol=1e-12)
    with MujocoBackend(tmp_path / 'new', scene_inputs()) as new:
        with pytest.raises(ValueError, match='Incompatible'):
            new.restore_state(initial)


def test_simulation_offsets_preserve_raw_joint_coordinates(tmp_path):
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend, configure_rendering
    inputs = scene_inputs()
    offsets = np.array([.03, -.04, .06, -.02, .01, 0.])
    inputs['profile']['simulation_joint_offsets_rad'] = {
        'value': offsets.tolist(), 'provenance': 'optimized',
        'evidence': ['Synthetic zero-frame test; not physical calibration']}
    configure_rendering('none')
    with MujocoBackend(tmp_path/'calibrated', workspace=inputs) as sim:
        q = sim.home.copy()
        sim.set_joint_state(q)
        np.testing.assert_allclose(sim.get_state()['joint_pos'], q, atol=1e-12)
        expected = sim.workspace_from_world @ np.array(inputs['profile']['robot_base']['T_world']) @ urdf_fk(
            ROOT/'assets/SO101/urdf/so101_isaaclab.urdf', dict(zip(JOINTS, q+offsets)))
        np.testing.assert_allclose(sim.get_eef_pose(), expected, atol=1e-9)
        np.testing.assert_allclose(sim.data.ctrl[sim.actuator_ids], q, atol=1e-12)
        assert not any(n.split('.')[0] in ('isaaclab','omni','pxr') for n in sys.modules)


def test_simulation_offset_contract_rejects_unknown_or_nonfinite():
    from soarm101_lab.so101_robot_spec import simulation_joint_offsets
    for values in ([0]*5, [float('nan')]*6, [1]*6):
        with pytest.raises(ValueError):
            simulation_joint_offsets({'value':values,'provenance':'optimized','evidence':['test']},JOINTS)
    with pytest.raises(ValueError):
        simulation_joint_offsets({'value':[0]*6,'provenance':'calibrated','evidence':['unsupported physical assertion']},JOINTS)


def test_shared_model_threaded_state_and_scratch_independence(env,tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    clones=[MujocoBackend(tmp_path/str(i),render_backend='none',shared_backend=env) for i in range(4)]
    try:
        assert len({id(c.model) for c in clones+[env]})==1
        assert len({id(d) for c in clones for d in (c.data,c._fk_data,c._ik_data)})==12
        original=env.data.qpos.copy();snapshots=[c.data.qpos.copy() for c in clones]
        def run(i):
            q=clones[i].home.copy();q[0]+=.03*i
            for _ in range(25):clones[i].step(q)
            return clones[i].get_state()['joint_pos'].copy()
        with ThreadPoolExecutor(4) as executor:results=list(executor.map(run,range(4)))
        assert not np.allclose(results[0],results[3])
        np.testing.assert_array_equal(env.data.qpos,original)
        other=clones[3].data.qpos.copy();clones[0].reset(seed=19)
        np.testing.assert_array_equal(clones[3].data.qpos,other)
    finally:
        for clone in clones:clone.close()


def test_portable_initial_velocities_roundtrip(env,tmp_path):
    import threading
    from soarm101_lab.workflow.adapters import MujocoAdapter
    # This storage-independent adapter test needs the standard Side/Wrist facade.
    env.workspace['task']['observation_cameras']=['side','wrist']
    adapter=MujocoAdapter(env,30,tmp_path,1,threading.Event())
    env.data.qvel[env.v_indices]=np.linspace(.01,.06,6)
    pick=env.metadata['pick'];_,index=env.object_indices[pick]
    env.data.qvel[index:index+6]=[.02,-.01,.03,.1,.2,-.1]
    env.mj.mj_forward(env.model,env.data)
    state=adapter.state();initial=adapter.initial_state()
    expected_q=env.data.qvel.copy()
    values={'initial_state':initial,'obs':{'datagen_info':{'object_pose':{k:v[None] for k,v in state['object_poses'].items()}}}}
    adapter.restore(values,{'simulation_backend':'isaac'})
    np.testing.assert_allclose(env.data.qvel,expected_q,atol=1e-9)
    adapter.buffer.abort()
