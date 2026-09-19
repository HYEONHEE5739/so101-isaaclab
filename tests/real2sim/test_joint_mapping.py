"""Exercise real boundary functions without opening serial ports or Isaac runtime."""
import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from soarm101_lab import so101_joint_mapping as mapping
from soarm101_lab.so101_joint_mapping import SO101SimJointMapper, JOINTS
from soarm101_lab.real2sim.core import to_sim, from_sim, joint_error
from soarm101_lab.real2sim.runtime import Runtime
from soarm101_lab.so101_dataset_contract import (
    current_contract, write_contract, require_resume_contract, inherit_contract,
    require_joint_replay, read_contract, JOINT_SPACE, MIMIC_SPACE,
)

ROOT = Path(__file__).resolve().parents[2]


def function_from_source(path, name, namespace, cls=None):
    """Execute actual functions, excluding script AppLauncher/import-time hardware setup."""
    tree = ast.parse((ROOT / path).read_text())
    body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body if cls else tree.body
    node = copy.deepcopy(next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    code = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(code, str(path), 'exec'), namespace)
    return namespace[name]


def test_identity_preserves_legacy_all_six_units_and_batches():
    x = np.random.default_rng(22).uniform(-200, 200, (100, 6))
    expected = np.array([[v / 180. * math.pi for v in row] for row in x])
    np.testing.assert_allclose(mapping.calibrated_to_urdf(x), expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(mapping.urdf_to_calibrated(expected), x, atol=1e-12)
    assert mapping.DEFAULT_MAPPER.evidence == 'legacy_unverified'


def test_known_sign_offset_gripper_and_roundtrip():
    m = SO101SimJointMapper(arm_sign=(1,-1,1,-1,1),
        arm_zero_offset_rad=(.1,.2,.3,.4,.5), gripper_rad_per_percent=.012,
        gripper_zero_offset_rad=-.2, evidence='synthetic_test_only')
    d = dict(zip(JOINTS,[0,90,-90,30,180,50]))
    q = m.calibrated_to_urdf(d)
    np.testing.assert_allclose(list(q.values()), [.1,-math.pi/2+.2,-math.pi/2+.3,-math.pi/6+.4,math.pi+.5,.4])
    np.testing.assert_allclose(list(m.urdf_to_calibrated(q).values()), list(d.values()), atol=1e-12)
    assert m.metadata()['sha256'] != mapping.DEFAULT_MAPPER.metadata()['sha256']
    assert 'scale' not in m.metadata()  # no freely fitted arm scale


@pytest.mark.parametrize('bad',[np.zeros(5), [0,0,0,0,0,float('nan')], {'gripper': 0}])
def test_invalid_coordinates_fail(bad):
    with pytest.raises(ValueError): mapping.calibrated_to_urdf(bad)


def test_live_leader_and_measured_render_use_same_authoritative_mapper(monkeypatch):
    m = SO101SimJointMapper(arm_zero_offset_rad=(.1,.2,.3,.4,.5), evidence='synthetic_test_only')
    monkeypatch.setattr(mapping, 'DEFAULT_MAPPER', m)
    advance = function_from_source('source/soarm101_lab/soarm101_lab/devices/lerobot/so101_leader.py',
        'advance', {'torch':torch,'SO101_MOTOR_NAMES':list(JOINTS),'calibrated_to_urdf':mapping.calibrated_to_urdf}, 'SO101Leader')
    native = dict(zip(JOINTS,[12,-20,35,-15,9,42]))
    live = advance(SimpleNamespace(_get_raw_data=lambda:native))
    r = Runtime.__new__(Runtime)
    np.testing.assert_allclose(live.numpy(), r.check_target(native), atol=1e-7)
    np.testing.assert_allclose(list(from_sim(to_sim(native)).values()),list(native.values()),atol=1e-12)
    assert joint_error(native, to_sim(native))['max_arm_error_rad'] < 1e-12


def test_measured_pose_writes_mapped_targets_without_follower_command():
    native = dict(zip(JOINTS,[12,-20,35,-15,9,42]))
    writes = []
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1,6)),
        write_joint_state_to_sim=lambda q,v:writes.append(q.clone()),
        set_joint_position_target=lambda q:writes.append(q.clone()))
    scene = {'robot':robot, **{f'camera_{v}view':SimpleNamespace(update=lambda *a,**kw:None) for v in ('side','wrist')}}
    class Scene(dict):
        def update(self,*args): pass
    r=Runtime.__new__(Runtime);r.ids=list(range(6));r.env=SimpleNamespace(scene=Scene(scene),device='cpu',step_dt=.03,
        sim=SimpleNamespace(forward=lambda:None,render=lambda:None))
    r.set_measured_pose(native)
    np.testing.assert_allclose(writes[0].numpy()[0],list(to_sim(native).values()),atol=1e-7)
    assert torch.equal(writes[0],writes[1])


def test_source_joint_target_record_and_replay_are_not_remapped(tmp_path):
    # HDF5 roundtrip of model targets; changing the mapper afterwards cannot reinterpret them.
    import h5py
    inputs=np.array([[0,10,20,30,40,50],[5,-10,25,20,-40,12.]])
    q=mapping.calibrated_to_urdf(inputs).astype(np.float32)
    path=tmp_path/'demo.hdf5';contract=current_contract(MIMIC_SPACE)
    with h5py.File(path,'w') as f:
        f.create_dataset('data/demo_0/joint_targets',data=q)
    write_contract(path,contract);require_joint_replay(path)
    with h5py.File(path,'r') as f: replay=f['data/demo_0/joint_targets'][:]
    np.testing.assert_array_equal(q,replay)
    # Actual Isaac JointPositionAction apply boundary forwards processed model radians unchanged.
    apply=function_from_source_external('/home/hyeonhee/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py',
        'JointPositionAction','apply_actions')
    targets=[];action=SimpleNamespace(processed_actions=torch.tensor(replay),_joint_ids=list(range(6)),
        _asset=SimpleNamespace(set_joint_position_target=lambda q,**kw:targets.append(q.clone())))
    apply(action);np.testing.assert_array_equal(targets[0].numpy(),q)


def function_from_source_external(path, cls, name):
    path=Path(path)
    if not path.exists(): pytest.skip('Local Isaac Lab source not available')
    return function_from_source(path,name,{},cls)


def test_mimic_cartesian_delta_and_generated_joint_recorder_keep_model_coordinates(monkeypatch):
    # Zero-rotation case verifies translation/gripper without loading Isaac.
    import sys
    fake_math=SimpleNamespace(matrix_from_quat=lambda q:torch.eye(3).unsqueeze(0),
        quat_from_matrix=lambda r:torch.tensor([[1.,0,0,0]]),axis_angle_from_quat=lambda q:torch.zeros(1,3))
    monkeypatch.setitem(sys.modules,'isaaclab',SimpleNamespace(utils=SimpleNamespace(math=fake_math)))
    monkeypatch.setitem(sys.modules,'isaaclab.utils',SimpleNamespace(math=fake_math))
    monkeypatch.setitem(sys.modules,'isaaclab.utils.math',fake_math)
    delta=function_from_source('scripts/envs/teleoperation/record_mimic_dataset.py','pose_delta_action',{})
    prev=torch.tensor([.1,.2,.3,1.,0,0,0]);nxt=prev.clone();nxt[:3]+=torch.tensor([.01,-.02,.03])
    g=torch.tensor(.37);a=delta(prev,nxt,g)
    torch.testing.assert_close(a,torch.tensor([.01,-.02,.03,0,0,0,.37]))
    record=function_from_source('source/soarm101_lab/soarm101_lab/tasks/manager_based/soarm101_lab/mdp/so101_mimic_recorders.py',
        'record_post_step',{'torch':torch},'PostStepJointTargetsRecorder')
    arm=torch.tensor([[.1,.2,.3,.4,.5]]);grip=g.reshape(1,1)
    terms={'arm':SimpleNamespace(joint_position_targets=arm),'gripper':SimpleNamespace(processed_actions=grip)}
    name,target=record(SimpleNamespace(_env=SimpleNamespace(action_manager=SimpleNamespace(get_term=terms.__getitem__))))
    assert name=='joint_targets';torch.testing.assert_close(target,torch.cat([arm,grip],-1))


def test_contract_inheritance_resume_and_legacy_unknown(tmp_path):
    source=tmp_path/'source.hdf5';annotated=tmp_path/'annotated.hdf5';generated=tmp_path/'generated.hdf5'
    c=current_contract(MIMIC_SPACE);write_contract(source,c)
    a=inherit_contract(source,annotated,MIMIC_SPACE);b=inherit_contract(annotated,generated,MIMIC_SPACE)
    assert a['mapping']==b['mapping']==c['mapping'];assert b['robot_asset']==c['robot_asset']
    require_resume_contract(source,c)
    changed=copy.deepcopy(c);changed['mapping']['sha256']='different'
    with pytest.raises(ValueError): require_resume_contract(source,changed)
    with pytest.raises(ValueError): write_contract(source,changed)
    legacy=tmp_path/'legacy_directory';legacy.mkdir()
    with pytest.warns(UserWarning): assert read_contract(legacy)['mapping'] is None
    with pytest.warns(UserWarning),pytest.raises(ValueError):require_resume_contract(legacy,c)


def test_capture_mapping_mismatch_is_explicit(monkeypatch):
    mapping.require_capture_mapping({})  # historical capture, current legacy semantics
    mapping.require_capture_mapping({'sim_joint_mapping':mapping.DEFAULT_MAPPER.metadata()})
    with pytest.raises(ValueError):mapping.require_capture_mapping({'sim_joint_mapping':{'sha256':'other'}})
    monkeypatch.setattr(mapping,'DEFAULT_MAPPER',SO101SimJointMapper(arm_zero_offset_rad=(.1,0,0,0,0),evidence='test'))
    with pytest.raises(ValueError):mapping.require_capture_mapping({})


def test_actual_motor_normalization_has_midpoint_zero_and_separate_percent():
    from enum import Enum
    class Modes(Enum):
        DEGREES=1
        RANGE_0_100=2
        RANGE_M100_100=3
    path='source/soarm101_lab/soarm101_lab/devices/lerobot/motors/motors_bus.py'
    normalize=function_from_source(path,'_normalize',{'MotorNormMode':Modes},'MotorsBus')
    inverse=function_from_source(path,'_unnormalize',{'MotorNormMode':Modes},'MotorsBus')
    bus=SimpleNamespace(calibration={'arm':SimpleNamespace(range_min=943,range_max=3242,drive_mode=0),
        'gripper':SimpleNamespace(range_min=1906,range_max=3349,drive_mode=0)},
        apply_drive_mode=True,model_resolution_table={'sts3215':4096},
        motors={'arm':SimpleNamespace(norm_mode=Modes.DEGREES),'gripper':SimpleNamespace(norm_mode=Modes.RANGE_0_100)},
        _id_to_name=lambda i: 'arm' if i==1 else 'gripper',_id_to_model=lambda i:'sts3215')
    assert normalize(bus,{1:2092.5,2:1906})=={1:0.,2:0.}
    assert normalize(bus,{2:3349})[2]==100
    raw={1:2310,2:2200}; roundtrip=inverse(bus,normalize(bus,raw))
    assert all(abs(roundtrip[i]-v)<=1 for i,v in raw.items())
    # DEGREES does not use drive_mode inversion; percentage does.
    bus.calibration['arm'].drive_mode=1;bus.calibration['gripper'].drive_mode=1
    assert normalize(bus,{1:3242})[1]>0
    assert normalize(bus,{2:1906})[2]==100
