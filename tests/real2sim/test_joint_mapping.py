"""Exercise real boundary functions without opening serial ports or Isaac runtime."""
import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from soarm101_lab.real2sim.precision import PrecisionMapper, native_from_raw, raw_from_native, fit_joint, review_joint
from test_precision_calibration import profile, reference, FakeCalibrationDevice, URDF
import numpy as np
import pytest
import torch
from soarm101_lab import so101_joint_mapping as mapping
from soarm101_lab.so101_joint_mapping import SO101SimJointMapper, JOINTS, DEFAULT_MAPPER
from soarm101_lab.real2sim.core import to_sim, from_sim, joint_error
from soarm101_lab.real2sim.runtime import Runtime
from soarm101_lab.so101_joint_mapping import observed_positions
from soarm101_lab.so101_robot_spec import joint_limits
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
        'advance', {'torch':torch,'SO101_MOTOR_NAMES':list(JOINTS),'DEFAULT_MAPPER':m,'observed_positions':observed_positions,'np':np,'joint_limits':joint_limits}, 'SO101Leader')
    native = dict(zip(JOINTS,[12,-20,35,-15,9,42]))
    live = advance(SimpleNamespace(_get_raw_data=lambda:dict.fromkeys(JOINTS,2048), _bus=SimpleNamespace(sync_read=lambda *a,**kw:dict.fromkeys(JOINTS,2048),
        motors={n:SimpleNamespace(id=i) for i,n in enumerate(JOINTS)},_normalize=lambda raw:{i:native[n] for i,n in enumerate(JOINTS)})))
    r = Runtime.__new__(Runtime)
    np.testing.assert_allclose(live.numpy(), list(observed_positions(r.mapper(),native).values()), atol=1e-7)
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
    r.profile={'cameras': {'side': {}, 'wrist': {}}}
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
    delta=function_from_source('scripts/teleoperation/record_mimic_dataset.py','pose_delta_action',{})
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


def runtime(mapper=None, backend='real'):
    r = Runtime.__new__(Runtime)
    r.backend=backend; r.env=None; r.running=False; r.activated=False
    r.precision_options={'use_precision':mapper is not None}; r.robot_model_path=URDF
    r.follower=SimpleNamespace(limits={n:[0,100] if n=='gripper' else [-180,180] for n in JOINTS}, activate=Mock(), send=Mock())
    r.leader=SimpleNamespace(activate=Mock())
    if mapper: r.follower.precision_mapper=r.leader.precision_mapper=mapper
    return r


@pytest.mark.parametrize('backend',['real','isaac','mujoco'])
@pytest.mark.parametrize('measured_degrees',[-125.,0.,125.])
def test_simple_teleop_limits_only_targets_without_recovery_or_motion_guards(backend,measured_degrees):
    mapper=PrecisionMapper(profile());spec=mapper.profile['joints']['shoulder_pan']['motor']
    requested=dict.fromkeys(JOINTS,0.);requested['shoulder_pan']=native_from_raw(2800,spec)  # 117°
    held={**requested,'shoulder_pan':native_from_raw(1500+measured_degrees/90*1000,spec)}
    r=runtime(mapper,backend)
    r.leader.read=lambda:(requested,{})
    r.follower.read=lambda:(held,{})
    r.start()
    assert r.running and r.state=='RUNNING'
    r.follower.send.assert_not_called()
    q,target=r.command_target(requested)
    assert math.degrees(q[0])==pytest.approx(114.93)
    np.testing.assert_allclose(q,list(mapper.calibrated_to_urdf(target).values()),atol=1e-12)
    observed,warnings=r.observation_status(held)
    assert math.degrees(observed['shoulder_pan'])==pytest.approx(measured_degrees)
    assert bool(warnings)==(abs(measured_degrees)>115)
    # No taught start window, tracking difference, target slew or boundary recovery.
    requested['shoulder_pan']=native_from_raw(400,spec)  # -99° immediately, no rate guard
    q,target=r.command_target(requested)
    assert math.degrees(q[0])==pytest.approx(-99)
    assert not r.target_limits
    assert not hasattr(r,'motion_guard') and not hasattr(r,'limiter')
    assert not hasattr(r,'prepare_command') and not hasattr(r,'send_command')


@pytest.mark.parametrize('backend', ['real','isaac','mujoco'])
def test_motor_range_saturates_teleop_and_invalid_data_remain_rejected(backend):
    m=PrecisionMapper(profile());m.profile['joints']['shoulder_pan']['motor']['calibration'].update(range_min=900,range_max=2100)
    r=runtime(m,backend)
    native=dict.fromkeys(JOINTS,0.);spec=m.profile['joints']['shoulder_pan']['motor']
    native['shoulder_pan']=native_from_raw(2700,spec)  # URDF-valid, motor range cannot represent it.
    r.leader.read=lambda:(native,{})
    r.follower.read=lambda:({**native,'shoulder_pan':native_from_raw(1500,spec)},{})
    r.start()
    assert r.running
    q, command = r.command_target(native)
    assert q[0] == pytest.approx(math.radians(54))
    assert raw_from_native(command['shoulder_pan'],spec) == pytest.approx(2100)
    assert 'shoulder_pan' in r.target_limits
    native['shoulder_pan']=native_from_raw(300,spec)
    q, command = r.command_target(native)
    assert raw_from_native(command['shoulder_pan'],spec) == pytest.approx(900)
    native['shoulder_pan']=native_from_raw(1600,spec)
    q, command = r.command_target(native)
    assert raw_from_native(command['shoulder_pan'],spec) == pytest.approx(1600)
    assert not r.target_limits  # Returns to tracking without another Start.
    r.activated=False
    r.follower.activate.reset_mock();r.leader.activate.reset_mock()
    native['shoulder_pan']=float('nan')
    with pytest.raises(ValueError):r.start()
    r.follower.activate.assert_not_called()
    native['shoulder_pan']=native_from_raw(3048,spec)
    with pytest.raises(ValueError,match='half-turn'):r.start()
    r.follower.activate.assert_not_called()
    native['shoulder_pan']=native_from_raw(1500,spec)
    m.profile['urdf_sha256']='wrong'
    with pytest.raises(ValueError,match='profile/model identity'):r.start()
    r.follower.activate.assert_not_called()


def test_joint_limits_are_extended_once_and_other_model_fields_unchanged():
    from soarm101_lab.so101_robot_spec import model_without_limits
    original=ROOT/'outputs/workflow_validation/joint_limit_extension/so101_before.urdf'
    if not original.exists():
        import subprocess,xml.etree.ElementTree as ET
        tree=ET.fromstring(subprocess.check_output(['git','show','HEAD:assets/SO101/urdf/so101_isaaclab.urdf'],cwd=ROOT))
        before={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in tree.findall('joint') if j.get('name') in JOINTS}
    else:
        before=joint_limits(original)
        assert model_without_limits(original)==model_without_limits(URDF)
    after=joint_limits(URDF)
    for i,n in enumerate(JOINTS):
        np.testing.assert_array_equal(after[n],np.array(before[n])+(np.array([-1,1])*math.radians(5) if i<4 else 0))


def test_sdk_saturated_gripper_native_retains_raw_measured_angle_in_capture_history():
    from soarm101_lab.real2sim.capture_dataset import align_measured
    spec=FakeCalibrationDevice('follower').motor_spec('gripper')
    spec['calibration'].update(range_min=2000,range_max=3048)
    entry=review_joint(fit_joint([reference('gripper',2048,0),reference('gripper',3048,90)],
                       spec,'gripper',list(joint_limits()['gripper'])))
    mapper=PrecisionMapper(profile(joints={'gripper':entry}))
    raw=dict.fromkeys(JOINTS,2048);raw['gripper']=3200
    native=dict.fromkeys(JOINTS,0.);native['gripper']=100.  # SDK _normalize saturation
    q=observed_positions(mapper,native,raw=raw)
    assert q['gripper']==pytest.approx(math.radians(103.68))
    aligned=align_measured([{'timestamp_ns':100,'native':native,'raw':raw}],100,mapper=mapper,observation_only=True)
    assert aligned[-1]==q['gripper']
    with pytest.raises(ValueError,match='packet mismatch'):
        observed_positions(mapper,{**native,'gripper':90},raw=raw)


def test_connect_outside_pose_is_read_only_and_motion_failure_does_not_disconnect(tmp_path,monkeypatch):
    import soarm101_lab.real2sim.runtime as module
    from soarm101_lab.real2sim.core import stamp
    leader=Mock();follower=Mock()
    leader.metadata=follower.metadata={'calibration':{}}
    follower.limits={n:[0,100] if n=='gripper' else [-180,180] for n in JOINTS}
    native=dict.fromkeys(JOINTS,0.);native['shoulder_lift']=120.
    interval={'start':stamp(),'end':stamp()}
    leader.read.return_value=follower.read.return_value=(native,interval)
    monkeypatch.setattr(module,'Leader',lambda *a,**k:leader)
    monkeypatch.setattr(module,'Follower',lambda *a,**k:follower)
    monkeypatch.setattr(module,'load',lambda path:{'cameras':{'side':{},'wrist':{}}})
    r=Runtime(SimpleNamespace(workspace=str(tmp_path),backend='real'),None)
    r.open_cameras=Mock()
    try:
        r.connect({'profile':'fixture','hardware':{'leader_port':'/fake/leader','follower_port':'/fake/follower',
            'follower_id':'fixture','leader_calibration':'fixture','side_camera':'/fake/side','wrist_camera':'/fake/wrist'}})
        assert r.state=='CONNECTED'
        leader.activate.assert_not_called();follower.activate.assert_not_called();follower.send.assert_not_called()
        r.tick_real()
        assert 'joint_control' not in r.snapshot_data
        assert r.snapshot_data['joint_limit_warnings']['shoulder_lift']['measured_rad']==pytest.approx(math.radians(120))
        r.start()
        assert r.follower is follower and r.running
        follower.close.assert_not_called();follower.send.assert_not_called()
    finally:r.writer.shutdown();r.bus.close()


def test_passive_snapshot_keeps_unclipped_raw_without_retired_control_metadata(tmp_path):
    from test_snapshots import fixture_input, observation
    from soarm101_lab.real2sim.capture_dataset import prepare_snapshot
    p,_,images=fixture_input(tmp_path)
    specs={n:FakeCalibrationDevice('follower').motor_spec(n) for n in JOINTS}
    specs['gripper']['calibration'].update(range_min=2000,range_max=3048)
    raw=dict.fromkeys(JOINTS,2048);raw['gripper']=3200
    native={n:native_from_raw(raw[n],specs[n]) for n in JOINTS};native['gripper']=100.
    q=observed_positions(DEFAULT_MAPPER,native,raw=raw,motor_specs=specs)
    state=observation(p)
    state.update(sent_target=None,precision_observation_only=True,physical_follower_measured_joint_state=native,
        precision_raw_feedback={'follower':raw},hardware={'follower':{'motor_specs':specs}},
        joint_limit_warnings={'gripper':{'measured_rad':q['gripper'],'limits_rad':list(joint_limits()['gripper'])}},
        joint_control={'observation_valid':True,'motion_allowed':False,'mapping_sha256':DEFAULT_MAPPER.metadata()['sha256']})
    for sample in state['measured_samples']:sample.update(native=copy.deepcopy(native),raw=copy.deepcopy(raw))
    result,_,warnings=prepare_snapshot(state,images,p)
    assert result['q_measured'][-1]==q['gripper'] and q['gripper']>math.radians(100)
    assert result['q_native']['gripper']==100 and result['precision_raw_feedback']['follower']['gripper']==3200
    assert 'joint_control' not in result and 'motion_allowed' not in result and warnings


@pytest.mark.parametrize('precision,drive',[(False,False),(True,False),(True,True)])
def test_actual_installed_sdk_integer_encoding_matches_sim_goal_without_a_port(precision,drive):
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    from soarm101_lab.real2sim.hardware import Follower
    d=FakeCalibrationDevice('follower')
    specs={n:d.motor_spec(n) for n in JOINTS}
    for spec in specs.values():spec['apply_drive_mode']=drive
    specs['gripper']['calibration']['drive_mode']=int(drive)
    bus=FeetechMotorsBus.__new__(FeetechMotorsBus)
    bus.apply_drive_mode=drive
    bus.motors={n:Motor(i+1,'sts3215',MotorNormMode[specs[n]['mode']]) for i,n in enumerate(JOINTS)}
    bus.calibration={n:MotorCalibration(**specs[n]['calibration']) for n in JOINTS}
    bus._id_to_name_dict={i+1:n for i,n in enumerate(JOINTS)}
    bus._id_to_model_dict={i+1:'sts3215' for i in range(6)}
    f=Follower.__new__(Follower);f.device=SimpleNamespace(bus=bus);f.motor_specs=specs
    mapper=DEFAULT_MAPPER
    if precision:
        entries={n:review_joint(fit_joint([reference(n,2048,0),reference(n,3048,-90 if n=='wrist_roll' else 90)],
                 specs[n],n,list(joint_limits()[n]))) for i,n in enumerate(JOINTS)}
        mapper=PrecisionMapper(profile(joints=entries));f.precision_mapper=mapper
    f.limits={n:[0,100] if n=='gripper' else [-180,180] for n in JOINTS}
    r=runtime(mapper if precision else None);r.follower=f
    rng=np.random.default_rng(42)
    lower=np.array([-.5]*5+[.2]);upper=np.array([.5]*5+[1.3])
    for target in [lower,upper]+list(rng.uniform(lower,upper,(100,6))):
        requested=mapper.urdf_to_calibrated(dict(zip(JOINTS,target)))
        actual,native=r.command_target(requested)
        raw=bus._unnormalize({i+1:native[n] for i,n in enumerate(JOINTS)})
        assert raw==f.last_encoded_goal_raw
        normalized=bus._normalize(raw)
        q=mapper.calibrated_to_urdf({n:normalized[i+1] for i,n in enumerate(JOINTS)})
        np.testing.assert_allclose(actual,list(q.values()),atol=1e-12)
        assert np.isfinite(actual).all()


def test_real_only_connect_with_existing_limit_only_profiles(tmp_path, monkeypatch):
    """Actual Runtime.connect + saved six-joint profiles, all device IO replaced."""
    import hashlib
    import shutil
    import dataclasses
    from soarm101_lab.real2sim.precision import PROFILE_ROOT, ProfileStore
    import soarm101_lab.real2sim.runtime as module
    from soarm101_lab.real2sim.core import stamp
    if not all((PROFILE_ROOT/role/'active.json').exists() for role in ('leader','follower')):
        pytest.skip('Local saved user calibration is not available')
    profiles={role:ProfileStore(PROFILE_ROOT).active(role) for role in ('leader','follower')}
    if any(p['urdf_sha256']==hashlib.sha256(URDF.read_bytes()).hexdigest() for p in profiles.values()):
        pytest.skip('This integration fixture requires the original pre-limit active profiles')
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in PROFILE_ROOT.rglob('*.json')}
    adapters={}
    for role,p in profiles.items():
        target=tmp_path/'joint_calibration'/role
        (target/'profiles').mkdir(parents=True)
        shutil.copyfile(PROFILE_ROOT/role/'active.json',target/'active.json')
        shutil.copyfile(PROFILE_ROOT/role/'profiles'/(p['id']+'.json'),target/'profiles'/(p['id']+'.json'))
        specs={n:e['motor'] for n,e in p['joints'].items()}
        motor_type=dataclasses.make_dataclass('SavedCalibration',[(k,int) for k in next(iter(specs.values()))['calibration']])
        bus=SimpleNamespace(motors={n:SimpleNamespace(**p['identity']['motors'][n],norm_mode=SimpleNamespace(name=sp['mode'])) for n,sp in specs.items()},
            calibration={n:motor_type(**sp['calibration']) for n,sp in specs.items()},
            model_resolution_table={'sts3215':4096},apply_drive_mode=next(iter(specs.values()))['apply_drive_mode'])
        raw={n:e['anchor_raw'] for n,e in p['joints'].items()}
        native={n:native_from_raw(raw[n],specs[n]) for n in JOINTS}
        adapter=Mock();adapter.device=SimpleNamespace(**{'_bus' if role=='leader' else 'bus':bus})
        adapter.metadata={'port':p['identity']['port'],'id':p['identity']['id'],
            'calibration':{'sha256':p['identity']['calibration_sha256'],'path':'fake-only'}}
        adapter.last_raw=raw;adapter.motor_specs=specs
        adapter.read.return_value=(native,{'start':stamp(),'end':stamp()})
        adapters[role]=adapter
    monkeypatch.setattr(module,'Leader',lambda *a,**kw:adapters['leader'])
    monkeypatch.setattr(module,'Follower',lambda *a,**kw:adapters['follower'])
    monkeypatch.setattr(module,'load',lambda path:{'cameras':{'side':{},'wrist':{}}})
    r=Runtime(SimpleNamespace(workspace=str(tmp_path),backend='real'),None);r.open_cameras=Mock()
    try:
        r.connect({'profile':'fake','precision_options':{'use_precision':True},'hardware':{
            'leader_port':'/fake/leader','follower_port':'/fake/follower','follower_id':'fake',
            'leader_calibration':'fake','side_camera':'/fake/side','wrist_camera':'/fake/wrist'}})
        assert r.state=='CONNECTED'
        for role,adapter in adapters.items():
            assert adapter.precision_mapper.profile['id']==profiles[role]['id']
            for name,entry in profiles[role]['joints'].items():
                actual=adapter.precision_mapper.profile['joints'][name]
                assert actual['limits']==list(joint_limits()[name])
                assert {k:v for k,v in actual.items() if k!='limits'}=={k:v for k,v in entry.items() if k!='limits'}
            adapter.activate.assert_not_called();adapter.send.assert_not_called()
        r.tick_real()
        assert r.snapshot_data is not None
        assert source_hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in PROFILE_ROOT.rglob('*.json')}
        for role,p in profiles.items():
            assert (tmp_path/'joint_calibration'/role/'active.json').read_bytes()==(PROFILE_ROOT/role/'active.json').read_bytes()
            assert len(list((tmp_path/'joint_calibration'/role/'profiles').glob('*.json')))==1
    finally:r.writer.shutdown();r.bus.close()
