import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from soarm101_lab.real2sim.precision import (Calibration, FakeCalibrationDevice, ProfileStore, PrecisionMapper,
    fit_joint, validate_joint, native_from_raw, raw_from_native, mapper_from_metadata, joint_specs)
from soarm101_lab.so101_joint_mapping import JOINTS

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT/'assets/SO101/urdf/so101_isaaclab.urdf'


def reference(joint, raw, degrees, split='train', **extra):
    return dict(joint=joint, raw=raw, angle_rad=math.radians(degrees), split=split, calibration_hash='fake', **extra)


def entry(role='follower', joint='shoulder_pan'):
    device = FakeCalibrationDevice(role)
    refs = [reference(joint,1000,-45),reference(joint,2000,45),reference(joint,1500,0,'validation')]
    return validate_joint(fit_joint(refs,device.motor_spec(joint),joint,joint_specs(URDF)[joint]['limits']))


def profile(role='follower', joints=None):
    return dict(schema='so101.precision-calibration/1', id='test',role=role,
                identity=FakeCalibrationDevice(role).identity,urdf_sha256=hashlib.sha256(URDF.read_bytes()).hexdigest(),joints=joints or {'shoulder_pan':entry(role)})


def test_raw_mapping_uses_range_mid_not_homing_twice():
    spec=FakeCalibrationDevice('follower').motor_spec('shoulder_pan')
    spec['calibration']['homing_offset']=999
    assert native_from_raw(2047.5,spec)==0
    assert raw_from_native(0,spec)==2047.5
    mapped=PrecisionMapper(profile())
    native={n:0. for n in JOINTS};native['shoulder_pan']=native_from_raw(1500,spec)
    assert mapped.calibrated_to_urdf(native)['shoulder_pan']==pytest.approx(0,abs=1e-12)
    untouched={n:0. for n in JOINTS};untouched['shoulder_lift']=30
    untouched['shoulder_pan']=native['shoulder_pan']
    assert mapped.calibrated_to_urdf(untouched)['shoulder_lift']==pytest.approx(math.pi/6)


@pytest.mark.parametrize('role',['leader','follower'])
def test_hardware_adapter_binds_active_mapping_without_motor_writes(tmp_path,role):
    from dataclasses import make_dataclass
    from unittest.mock import Mock
    from soarm101_lab.real2sim.hardware import attach_active_precision
    p=profile(role);spec=p['joints']['shoulder_pan']['motor']
    cal_type=make_dataclass('CalibrationRecord',[(k,int) for k in spec['calibration']])
    calibration=cal_type(**spec['calibration'])
    motor=SimpleNamespace(id=calibration.id,model='sts3215',norm_mode=SimpleNamespace(name=spec['mode']))
    bus=SimpleNamespace(motors={'shoulder_pan':motor},calibration={'shoulder_pan':calibration},
        model_resolution_table={'sts3215':spec['resolution']},apply_drive_mode=spec['apply_drive_mode'],write=Mock())
    metadata={'port':str(tmp_path/'motor'), 'id':'test_robot',
              'calibration':{'path':str(tmp_path/'robot.json'),'sha256':'fake'}}
    p['identity']={'role':role,'port':metadata['port'],'id':'test_robot','calibration_sha256':'fake',
                   'motors':{'shoulder_pan':{'id':calibration.id,'model':'sts3215'}}}
    device=SimpleNamespace(**{'_bus' if role=='leader' else 'bus':bus})
    adapter=SimpleNamespace(device=device,metadata=metadata)
    store=ProfileStore(tmp_path/'profiles');path=store.save(role,p['identity'],'urdf',p['joints'])
    store.activate(role,path,p['identity'],'urdf',True)
    mapper=attach_active_precision(adapter,role,store.root,'urdf')
    assert mapper._forward(2500,'shoulder_pan')==pytest.approx(math.pi/2)
    assert adapter.metadata['precision_mapping']['profile']['role']==role
    bus.write.assert_not_called()
    # Unknown model evidence must still be refused without touching the pointer.
    metadata_before=copy.deepcopy(adapter.metadata)
    with pytest.raises(ValueError,match='URDF identity conflict') as model_error:
        attach_active_precision(adapter,role,store.root,'new_urdf')
    assert 'evidence unavailable' in str(model_error.value)
    assert adapter.precision_mapper is mapper
    assert adapter.metadata==metadata_before
    assert store.active(role)['urdf_sha256']=='urdf'
    bus.write.assert_not_called()
    adapter.metadata['id']='wrong_robot'
    with pytest.raises(ValueError,match='device/calibration identity conflict') as device_error:
        attach_active_precision(adapter,role,store.root,'new_urdf')
    assert f'{role}: id' in str(device_error.value)
    assert 'Limit 변경' not in str(device_error.value)
    bus.write.assert_not_called()


def test_hardware_policy_uses_precision_once_and_preserves_source_contract(tmp_path):
    from soarm101_lab.representation import real_observation, real_action, require_compatible
    from soarm101_lab.real2sim.precision import collection_coordinates, uses_urdf_frame, require_hardware_policy_frame
    from soarm101_lab.so101_dataset_contract import current_contract, MIMIC_SPACE, write_contract, require_resume_contract
    mapper=PrecisionMapper(profile());spec=mapper.profile['joints']['shoulder_pan']['motor']
    native={n+'.pos':0. for n in JOINTS};native['shoulder_pan.pos']=native_from_raw(2500,spec)
    q=real_observation(native,mapper)
    assert q[0]==pytest.approx(math.pi/2)
    assert real_action(q,mapper)['shoulder_pan.pos']==pytest.approx(native['shoulder_pan.pos'])
    baseline=current_contract(MIMIC_SPACE);recorded=collection_coordinates(baseline,mapper)
    assert uses_urdf_frame(recorded) and not uses_urdf_frame(baseline)
    assert recorded['mapping']==baseline['mapping']  # Model radians are never mapped a second time.
    require_compatible(recorded,baseline,MIMIC_SPACE)
    path=tmp_path/'demo.hdf5';write_contract(path,recorded)
    require_resume_contract(path,recorded)
    with pytest.raises(ValueError,match='provenance'):require_resume_contract(path,baseline)
    damaged=copy.deepcopy(recorded);damaged['hardware_mapping']['profile']['role']='wrong'
    with pytest.raises(ValueError,match='hash mismatch'):uses_urdf_frame(damaged)
    scene={'simulation_joint_offsets_rad':{'value':[.01,0,0,0,0,0],'provenance':'estimated','evidence':['fixture']}}
    require_hardware_policy_frame(recorded,scene,mapper)
    require_hardware_policy_frame(baseline,{},mapper)
    with pytest.raises(ValueError,match='historical Scene'):require_hardware_policy_frame(baseline,scene,mapper)


def test_direction_scale_zero_and_one_point_status():
    e=entry();assert e['status']=='VALIDATED_MANUAL'
    assert e['rad_per_count']==pytest.approx(math.pi/2000)
    assert e['zero_raw_unwrapped']==pytest.approx(1500)
    spec=FakeCalibrationDevice('leader').motor_spec('shoulder_pan')
    refs=[reference('shoulder_pan',1000,45),reference('shoulder_pan',2000,-45),reference('shoulder_pan',1500,0,'validation')]
    assert validate_joint(fit_joint(refs,spec,'shoulder_pan',[-2,2]))['direction']==-1
    one=fit_joint(refs[:1],spec,'shoulder_pan',[-2,2]);assert one['status']=='OFFSET_ONLY_NEEDS_MORE_DATA'
    assert validate_joint(one)['status']=='NEEDS_MORE_DATA'


def test_wraparound_forward_and_inverse():
    d=FakeCalibrationDevice('follower');spec=d.motor_spec('shoulder_pan')
    refs=[reference('shoulder_pan',4000,-10),reference('shoulder_pan',400,10),reference('shoulder_pan',152,0,'validation')]
    e=validate_joint(fit_joint(refs,spec,'shoulder_pan',[-2,2]));assert e['status']=='VALIDATED_MANUAL'
    m=PrecisionMapper(profile(joints={'shoulder_pan':e}));native={n:0. for n in JOINTS};native['shoulder_pan']=native_from_raw(152,spec)
    q=m.calibrated_to_urdf(native);assert q['shoulder_pan']==pytest.approx(0,abs=1e-12)
    encoded=m.urdf_to_calibrated(q);assert raw_from_native(encoded['shoulder_pan'],spec)==pytest.approx(152)
    with pytest.raises(ValueError,match='Ambiguous encoder branch'):m.calibrated_to_urdf({**native,'shoulder_pan':0.})


@pytest.mark.parametrize('joint',JOINTS)
@pytest.mark.parametrize('direction',[1,-1])
def test_two_reference_mapping_covers_urdf_limits_not_reference_extent(joint,direction):
    from soarm101_lab.real2sim.precision import review_joint
    spec=FakeCalibrationDevice('follower').motor_spec(joint)
    refs=[reference(joint,2048,0),reference(joint,2048+direction*1000,90)]
    limits=joint_specs(URDF)[joint]['limits']
    e=review_joint(fit_joint(refs,spec,joint,limits))
    original=copy.deepcopy(e)
    mapper=PrecisionMapper(profile(joints={joint:e}))
    # Include both model endpoints, negative angles, and >90 degrees if allowed.
    for angle in np.linspace(*limits,17):
        raw=e['anchor_raw']+(angle-e['reference_angle_rad'])/e['rad_per_count']
        assert mapper._forward(raw % spec['resolution'],joint)==pytest.approx(angle)
        native={n:0. for n in JOINTS};native[joint]=native_from_raw(raw,spec)
        goal=mapper.urdf_to_calibrated({**mapper.calibrated_to_urdf(native),joint:angle})
        decoded=mapper.calibrated_to_urdf(goal)[joint]
        assert limits[0]<=decoded<=limits[1]
        assert abs(decoded-angle)<=abs(e['rad_per_count'])+1e-9
    for angle in [limits[0]-.1,limits[1]+.1]:
        raw=e['anchor_raw']+(angle-e['reference_angle_rad'])/e['rad_per_count']
        with pytest.raises(ValueError,match='URDF'):mapper._forward(raw % spec['resolution'],joint)
    assert e==original  # Operational bounds do not invent new reference evidence.


def test_urdf_branch_resolves_far_side_of_encoder_and_rejects_ambiguous_goal():
    from soarm101_lab.real2sim.precision import review_joint
    spec=FakeCalibrationDevice('follower').motor_spec('wrist_roll')
    refs=[reference('wrist_roll',3500,0),reference('wrist_roll',404,90)]
    e=review_joint(fit_joint(refs,spec,'wrist_roll',joint_specs(URDF)['wrist_roll']['limits']))
    m=PrecisionMapper(profile(joints={'wrist_roll':e}))
    for degree in [-160,-90,0,90,160]:
        q=math.radians(degree);raw=(3500+degree*1000/90)%4096
        assert m._forward(raw,'wrist_roll')==pytest.approx(q)
        target={n:0. for n in JOINTS};target['wrist_roll']=q
        assert m.calibrated_to_urdf(m.urdf_to_calibrated(target))['wrist_roll']==pytest.approx(q,abs=e['rad_per_count'])
    # A scale implying several possible revolutions cannot authorize a goal.
    spec=FakeCalibrationDevice('follower').motor_spec('shoulder_pan')
    e=entry();e['motor']['calibration']['range_max']=8192
    m=PrecisionMapper(profile(joints={'shoulder_pan':e}))
    with pytest.raises(ValueError,match='ambiguous command'):m.urdf_to_calibrated({n:0. for n in JOINTS})


def test_display_range_does_not_expand_calibrated_motor_command_limits():
    e=entry();e['motor']['calibration'].update(range_min=1000,range_max=2000)
    m=PrecisionMapper(profile(joints={'shoulder_pan':e}))
    assert m._forward(2500,'shoulder_pan')==pytest.approx(math.pi/2)
    q={n:0. for n in JOINTS};q['shoulder_pan']=math.pi/2
    with pytest.raises(ValueError,match='calibrated motor range'):m.urdf_to_calibrated(q)


@pytest.mark.parametrize('direction',[1,-1])
def test_teleop_command_limits_integer_endpoints_and_profile_preserved(direction):
    spec=FakeCalibrationDevice('follower').motor_spec('shoulder_pan')
    refs=[reference('shoulder_pan',1000,-45*direction),reference('shoulder_pan',2000,45*direction),
          reference('shoulder_pan',1500,0,'validation')]
    e=validate_joint(fit_joint(refs,spec,'shoulder_pan',joint_specs(URDF)['shoulder_pan']['limits']))
    e['motor']['calibration'].update(range_min=1100,range_max=1950)
    m=PrecisionMapper(profile(joints={'shoulder_pan':e}));before=copy.deepcopy(m.profile)
    bounds={n:joint_specs(URDF)[n]['limits'] for n in JOINTS}
    available=m.command_limits(bounds)
    for q_pan in np.linspace(*available['shoulder_pan'],101):
        native=m.urdf_to_calibrated({**dict.fromkeys(JOINTS,0.),'shoulder_pan':q_pan})
        assert 1100-1e-7 <= raw_from_native(native['shoulder_pan'],e['motor']) <= 1950+1e-7
    assert m.profile==before


def test_teleop_command_limits_reject_disconnected_encoder_domain():
    spec=FakeCalibrationDevice('follower').motor_spec('shoulder_pan')
    refs=[reference('shoulder_pan',0,0),reference('shoulder_pan',1024,90),
          reference('shoulder_pan',512,45,'validation')]
    e=validate_joint(fit_joint(refs,spec,'shoulder_pan',joint_specs(URDF)['shoulder_pan']['limits']))
    e['motor']['calibration'].update(range_min=100,range_max=3996)
    m=PrecisionMapper(profile(joints={'shoulder_pan':e}))
    with pytest.raises(ValueError,match='single unambiguous'):
        m.command_limits({n:joint_specs(URDF)[n]['limits'] for n in JOINTS})


def test_manual_live_active_mapping_displays_beyond_ninety_without_writes(tmp_path):
    c=Calibration(tmp_path,URDF);c.connect({'role':'follower','fake':True})
    c.command('release',{'supported_and_approved':True});d=c.devices['follower']
    for raw,degree in [(2048,0),(3048,90)]:
        c.command('fake_raw',{'raw':raw});c.command('record',{'angle_rad':math.radians(degree),'split':'train'})
    c.command('fit',{});c.command('review',{})
    path=c.command('save',{})['path'];c.command('activate',{'path':path,'approved':True})
    for raw,degree in [(3159,99.99),(1048,-90)]:
        c.command('fake_raw',{'raw':raw});c.tick()
        assert c.samples['follower']['urdf_angle_rad']==pytest.approx(math.radians(degree))
        assert not c.errors
    c.command('fake_raw',{'raw':3500});c.tick()
    assert c.samples['follower']['urdf_angle_rad'] is None
    assert c.samples['follower']['calibration_status']=='MAPPING_ERROR'
    assert d.writes==[('Torque_Enable','shoulder_pan',0)]
    c.close()


def test_mujoco_precision_gripper_command_not_clamped_to_taught_points():
    from soarm101_lab.real2sim.precision import review_joint
    from soarm101_lab.real2sim.runtime import Runtime
    from unittest.mock import Mock
    spec=FakeCalibrationDevice('follower').motor_spec('gripper')
    e=review_joint(fit_joint([reference('gripper',2048,0),reference('gripper',3048,90)],
                           spec,'gripper',joint_specs(URDF)['gripper']['limits']))
    m=PrecisionMapper(profile(joints={'gripper':e}))
    r=Runtime.__new__(Runtime);r.backend='mujoco';r.precision_options={'use_precision':True}
    r.follower=SimpleNamespace(precision_mapper=m)
    r.env=SimpleNamespace(limits=np.array([joint_specs(URDF)[n]['limits'] for n in JOINTS]),_target=Mock())
    native={n:0. for n in JOINTS};native['gripper']=native_from_raw(3100,spec)
    r.leader=SimpleNamespace(precision_mapper=m)
    r.follower.limits={n:[-180,180] for n in JOINTS};r.follower.limits['gripper']=[0,100]
    applied,result=r.command_target(native)
    assert m.calibrated_to_urdf(result)['gripper']==pytest.approx(math.radians(94.68))
    r.env._target(applied)
    assert applied[-1]==pytest.approx(math.radians(94.68))
    assert not r.target_limits


def test_invalid_range_wrap_and_inconsistent_validation():
    d=FakeCalibrationDevice('follower');spec=d.motor_spec('shoulder_pan')
    with pytest.raises(ValueError,match='distinct'):fit_joint([reference('shoulder_pan',1000,0),reference('shoulder_pan',1001,30)],spec,'shoulder_pan',[-2,2])
    with pytest.raises(ValueError,match='half-turn'):fit_joint([reference('shoulder_pan',0,0),reference('shoulder_pan',2048,30)],spec,'shoulder_pan',[-2,2])
    spec['calibration']['range_max']=0
    with pytest.raises(ValueError,match='range'):native_from_raw(50,spec)
    spec=d.motor_spec('shoulder_pan');refs=[reference('shoulder_pan',1000,-45),reference('shoulder_pan',2000,45),reference('shoulder_pan',1500,20,'validation')]
    assert validate_joint(fit_joint(refs,spec,'shoulder_pan',[-2,2]))['status']=='NEEDS_MORE_DATA'


def test_old_gripper_endpoint_profiles_preserve_original_validation():
    d=FakeCalibrationDevice('follower');spec=d.motor_spec('gripper');spec['apply_drive_mode']=True;spec['calibration']['drive_mode']=1
    assert native_from_raw(0,spec)==100
    assert raw_from_native(100,spec)==0
    refs=[reference('gripper',1000,0,reference_kind='closed',opening_m=0),reference('gripper',2000,45,reference_kind='open',opening_m=.04),reference('gripper',1500,22.5,'validation',reference_kind='intermediate',opening_m=.02)]
    old=fit_joint(refs,spec,'gripper',[-.1745,1.745],reference_mode='legacy_endpoints')
    old.pop('reference_mode')  # Saved profiles before angle-only mode did not carry this field.
    assert validate_joint(old)['status']=='VALIDATED_MANUAL'
    saved=profile(joints={'gripper':validate_joint(old)});original=copy.deepcopy(saved)
    PrecisionMapper(saved)
    assert saved==original  # Old files still need their evidence reader, not their obsolete worker actions.
    refs[1]['reference_kind']='angle'
    old=fit_joint(refs,spec,'gripper',[-.1745,1.745],reference_mode='legacy_endpoints');old.pop('reference_mode')
    assert validate_joint(old)['status']=='NEEDS_MORE_DATA'


@pytest.mark.parametrize('role',['follower','leader'])
def test_gripper_angle_only_fit_and_normalized_goal_roundtrip(tmp_path,role):
    c=Calibration(tmp_path,URDF);c.connect({'role':role,'fake':True});d=c.devices[role]
    c.command('select',{'joint':'gripper'});c.command('release',{'supported_and_approved':True})
    for raw,degree in [(1000,0),(2000,45),(1500,22.5)]:
        c.command('fake_raw',{'raw':raw})
        ref=c.command('record',{'angle_rad':math.radians(degree),'split':'train','reference_kind':'angle'})
        assert 'opening_m' not in ref
    fit=c.command('fit',{});assert fit['reference_mode']=='angle'
    assert c.command('review',{})['status']=='REVIEWED_MANUAL'
    path=c.command('save',{})['path'];c.command('activate',{'path':path,'approved':True});c.tick()
    assert c.samples[role]['urdf_angle_rad']==pytest.approx(math.pi/8)
    mapper=PrecisionMapper(c.store.active(role));spec=d.motor_spec('gripper')
    native={n:0. for n in JOINTS};native['gripper']=native_from_raw(1750,spec)
    q=mapper.calibrated_to_urdf(native)
    assert q['gripper']==pytest.approx(math.radians(33.75))
    assert raw_from_native(mapper.urdf_to_calibrated(q)['gripper'],spec)==pytest.approx(1750)
    assert d.writes==[('Torque_Enable','gripper',0)]
    with pytest.raises(ValueError,match='URDF'):
        c.command('record',{'angle_rad':math.pi,'split':'train'})
    c.close()


def test_profile_save_approval_identity_restore_and_tamper(tmp_path):
    store=ProfileStore(tmp_path);p=profile();path=store.save('follower',p['identity'],'urdf',p['joints'])
    with pytest.raises(ValueError,match='approval'):store.activate('follower',path,p['identity'],'urdf')
    with pytest.raises(ValueError,match='identity'):store.activate('follower',path,{'wrong':1},'urdf',True)
    saved=store.activate('follower',path,p['identity'],'urdf',True);assert store.active('follower')==saved
    store.restore('follower',True);assert store.active('follower') is None
    m=PrecisionMapper(p).metadata();m['profile']['joints']['shoulder_pan']['rad_per_count']*=2
    with pytest.raises(ValueError,match='hash'):mapper_from_metadata(m)


def test_partial_activation_keeps_other_joints_and_save_collects_reviewed_candidates(tmp_path):
    from soarm101_lab.real2sim.precision import review_joint
    c=Calibration(tmp_path,URDF);c.connect({'role':'follower','fake':True});d=c.devices['follower']
    for joint in ('shoulder_pan','shoulder_lift'):
        c.command('select',{'joint':joint});c.command('release',{'supported_and_approved':True})
        for raw,degree in [(1000,-45),(2000,45)]:
            c.command('fake_raw',{'raw':raw});c.command('record',{'angle_rad':math.radians(degree),'split':'train'})
        c.command('fit',{});c.command('review',{})
    # Save once at the end: both reviewed joints must survive.
    path=c.command('save',{})['path'];c.command('activate',{'path':path,'approved':True})
    assert set(c.store.active('follower')['joints'])=={'shoulder_pan','shoulder_lift'}
    # A different partial saved file cannot drop already activated joints.
    refs=[reference('elbow_flex',1000,0),reference('elbow_flex',2000,45)]
    new=review_joint(fit_joint(refs,d.motor_spec('elbow_flex'),'elbow_flex',c.specs['elbow_flex']['limits']))
    partial=c.store.save('follower',d.identity,c.urdf_hash,{'elbow_flex':new})
    applied=c.store.activate('follower',partial,d.identity,c.urdf_hash,True)
    assert set(applied['joints'])=={'shoulder_pan','shoulder_lift','elbow_flex'}
    assert applied['provenance']['requested_profile']==Path(partial).stem
    assert set(__import__('json').loads(Path(partial).read_text())['joints'])=={'elbow_flex'}
    c.close()
    restarted=Calibration(tmp_path,URDF);restarted.connect({'role':'follower','fake':True})
    assert set(restarted.status()['robots']['follower']['references'])==set(applied['joints'])
    assert restarted.loaded_profiles['follower']==applied['id'] and not restarted.candidates['follower']
    assert restarted.devices['follower'].writes==[]
    restarted.close()


def test_repeated_save_and_activation_do_not_create_files_or_extend_history(tmp_path):
    store=ProfileStore(tmp_path);p=profile()
    path=store.save('follower',p['identity'],'urdf',p['joints'])
    assert store.save('follower',p['identity'],'urdf',p['joints'])==path
    store.activate('follower',path,p['identity'],'urdf',True)
    pointer=tmp_path/'follower/active.json';original=pointer.read_bytes()
    for _ in range(3):
        assert store.save('follower',p['identity'],'urdf',p['joints'])==path
        store.activate('follower',path,p['identity'],'urdf',True)
    assert pointer.read_bytes()==original and len(list((tmp_path/'follower/profiles').glob('*.json')))==1
    # Changed evidence gets one new immutable version; repeating it reuses that version.
    changed=copy.deepcopy(p['joints']);changed['shoulder_pan']['references'][0]['note']='new manual evidence'
    newpath=store.save('follower',p['identity'],'urdf',changed)
    assert newpath!=path and store.save('follower',p['identity'],'urdf',changed)==newpath
    assert len(list((tmp_path/'follower/profiles').glob('*.json')))==2
    assert __import__('json').loads(Path(path).read_text())['joints']==p['joints']


def test_two_point_residual_is_not_a_raw_spacing_check():
    from soarm101_lab.real2sim.precision import review_joint
    spec=FakeCalibrationDevice('follower').motor_spec('shoulder_pan')
    # A 90-degree label over 1100 counts changes gain from the encoder's nominal 1024 counts.
    refs=[reference('shoulder_pan',1000,0),reference('shoulder_pan',2100,90)]
    checked=review_joint(fit_joint(refs,spec,'shoulder_pan',[-2,2]),.5)
    assert checked['review']['max_error_deg']<1e-10 and checked['review']['passed']
    assert checked['rad_per_count']!=pytest.approx(2*math.pi/spec['resolution'])
    # A third conflicting reference makes the residual check informative.
    refs.append(reference('shoulder_pan',1550,48))
    checked=review_joint(fit_joint(refs,spec,'shoulder_pan',[-2,2]),.5)
    assert checked['review']['max_error_deg']>.5 and not checked['review']['passed']


def test_selected_joint_capture_no_other_writes_and_fake_workflow(tmp_path):
    c=Calibration(tmp_path,URDF);c.connect({'role':'follower','fake':True});d=c.devices['follower']
    c.command('select',{'joint':'shoulder_pan'})
    with pytest.raises(ValueError,match='approve'):c.command('release',{})
    c.command('release',{'supported_and_approved':True})
    assert d.writes==[('Torque_Enable','shoulder_pan',0)]
    for raw,degree in [(1000,-45),(2000,45),(1500,0)]:
        c.command('fake_raw',{'raw':raw});ref=c.command('record',{'angle_rad':math.radians(degree)})
        assert ref['split']=='train' and ref['reference_kind']=='angle' and 'opening_m' not in ref
    assert set(c.references['follower'])=={'shoulder_pan'}
    c.command('fit',{});assert c.command('review',{})['status']=='REVIEWED_MANUAL'
    with pytest.raises(ValueError,match='Unsupported calibration command'):c.command('validate',{})
    for old_options in ({'split':'validation'},{'reference_kind':'closed'},{'opening_m':.02}):
        with pytest.raises(ValueError,match='only angle'):c.command('record',{'angle_rad':0.,**old_options})
    assert len(c.references['follower']['shoulder_pan'])==3
    path=c.command('save',{})['path'];c.command('activate',{'path':path,'approved':True});c.tick()
    assert c.status()['robots']['follower']['sample']['urdf_angle_rad']==pytest.approx(0,abs=1e-12)
    assert len(d.writes)==1
    assert c.command('restore',{'approved':True}) is None
    assert len(list((tmp_path/'fake/follower/references').glob('*.json')))==1
    c.close()


def test_all_train_manual_live_review_save_and_no_motion(tmp_path):
    c=Calibration(tmp_path,URDF);c.connect({'role':'follower','fake':True});d=c.devices['follower']
    c.command('release',{'supported_and_approved':True})
    for raw,degree in [(1000,-45),(2000,45),(1500,0)]:
        c.command('fake_raw',{'raw':raw});c.command('record',{'angle_rad':math.radians(degree),'split':'train'})
    c.command('fit',{});result=c.command('review',{})
    assert result['status']=='REVIEWED_MANUAL' and result['review']['tolerance_deg']==.5
    assert result['review']['independent_validation'] is False and result['physical_verified'] is False
    assert 'validation' not in result
    for raw in [1200,1700,1800]:
        c.command('fake_raw',{'raw':raw});c.tick()
        sample=c.status()['robots']['follower']['sample']
        assert sample['raw']==raw and 'legacy_urdf_angle_rad' in sample
    assert len(c.references['follower']['shoulder_pan'])==3
    path=c.command('save',{})['path']
    c.command('activate',{'path':path,'approved':True});c.tick()
    assert c.samples['follower']['urdf_angle_rad']==pytest.approx(math.radians(27))
    assert d.writes==[('Torque_Enable','shoulder_pan',0)]
    # Profile checks cannot be bypassed by editing the review report or coefficients.
    from soarm101_lab.real2sim.precision import review_joint
    damaged=profile(joints={'shoulder_pan':result});damaged['joints']['shoulder_pan']['rad_per_count']*=1.1
    with pytest.raises(ValueError,match='coefficients'):PrecisionMapper(damaged)
    rejected_refs=[reference('shoulder_pan',1000,-45),reference('shoulder_pan',2000,45),reference('shoulder_pan',1500,3)]
    rejected=review_joint(fit_joint(rejected_refs,d.motor_spec('shoulder_pan'),'shoulder_pan',[-2,2]))
    assert rejected['status']=='NEEDS_MORE_DATA' and not rejected['review']['passed']
    with pytest.raises(ValueError,match='Unvalidated'):PrecisionMapper(profile(joints={'shoulder_pan':rejected}))
    c.close()


def test_exclude_reference_preserves_originals_and_invalidates_only_selected_joint(tmp_path,monkeypatch):
    import json
    c=Calibration(tmp_path,URDF);c.connect({'role':'follower','fake':True});d=c.devices['follower']
    c.command('release',{'supported_and_approved':True})
    for raw,degree in [(1000,-45),(2000,45),(1500,0)]:
        c.command('fake_raw',{'raw':raw});c.command('record',{'angle_rad':math.radians(degree),'split':'train'})
    c.command('fit',{});c.command('review',{})
    saved=c.command('save',{})['path'];c.command('activate',{'path':saved,'approved':True})
    active=copy.deepcopy(c.store.active('follower'));originals=copy.deepcopy(c.references['follower']['shoulder_pan'])
    identifier=originals[1]['reference_id'];request={'reference_id':identifier,'approved':True,'reason':'wrong manual pose'}
    with pytest.raises(ValueError,match='approval'):c.command('exclude_reference',{**request,'approved':False})
    with pytest.raises(ValueError,match='not found'):c.command('exclude_reference',{**request,'joint':'elbow_flex'})
    before=copy.deepcopy(c.candidates['follower']['shoulder_pan']);c.candidates['follower']['elbow_flex']={'untouched':True}
    with monkeypatch.context() as m:
        from soarm101_lab.real2sim import precision
        def fail(*args):raise OSError('disk failure')
        m.setattr(precision,'atomic',fail)
        with pytest.raises(OSError,match='disk'):c.command('exclude_reference',request)
    assert c.candidates['follower']['shoulder_pan']==before and not c.exclusions['follower']
    assert c.command('exclude_reference',request)['remaining']==2
    assert c.references['follower']['shoulder_pan']==originals
    assert c.candidates['follower']=={'elbow_flex':{'untouched':True}}
    assert c.store.active('follower')==active and Path(saved).exists()
    journal=next((tmp_path/'fake/follower/references').glob('*.json'));audit=json.loads(journal.read_text())
    assert audit['references']['shoulder_pan']==originals and audit['exclusions'][0]['reason']=='wrong manual pose'
    c.command('fit',{});assert len(c.candidates['follower']['shoulder_pan']['references'])==2
    with pytest.raises(ValueError,match='already excluded'):c.command('exclude_reference',request)
    c.command('fake_raw',{'raw':2000});new=c.command('record',{'angle_rad':math.pi/4,'split':'train'})
    assert new['reference_id']!=identifier and len(c.active_references('follower','shoulder_pan'))==3
    assert len(json.loads(journal.read_text())['references']['shoulder_pan'])==4
    assert len(json.loads(journal.read_text())['exclusions'])==1
    for ref in list(c.active_references('follower','shoulder_pan')):
        c.command('exclude_reference',{**request,'reference_id':ref['reference_id']})
    with pytest.raises(ValueError,match='reference first'):c.command('fit',{})
    assert d.writes==[('Torque_Enable','shoulder_pan',0)]
    c.close()


def test_corrupt_active_profile_can_restore_without_a_motor_write(tmp_path):
    from soarm101_lab.real2sim.storage import atomic
    c=Calibration(tmp_path,URDF);c.connect({'role':'follower','fake':True});d=c.devices['follower']
    path=c.store.save('follower',d.identity,c.urdf_hash,{'shoulder_pan':entry()})
    c.store.activate('follower',path,d.identity,c.urdf_hash,True)
    import json
    damaged=json.loads(Path(path).read_text());damaged['joints']['shoulder_pan']['rad_per_count']+=.1;atomic(path,damaged)
    c.tick();s=c.status()['robots']['follower']
    assert s['active_profile'] is None and s['active_pointer_exists']
    assert c.command('restore',{'approved':True}) is None
    assert d.writes==[] and Path(path).exists()
    c.close()


def test_leader_follower_bidirectional_common_coordinates():
    from soarm101_lab.real2sim.runtime import Runtime
    follower=PrecisionMapper(profile());spec=FakeCalibrationDevice('leader').motor_spec('shoulder_pan')
    refs=[reference('shoulder_pan',1400,45),reference('shoulder_pan',2400,-45),reference('shoulder_pan',1900,0,'validation')]
    leader=PrecisionMapper(profile('leader',{'shoulder_pan':validate_joint(fit_joint(refs,spec,'shoulder_pan',[-2,2]))}))
    r=Runtime.__new__(Runtime);r.precision_options={'use_precision':True};r.leader=SimpleNamespace(precision_mapper=leader);r.follower=SimpleNamespace(precision_mapper=follower)
    native={n:0. for n in JOINTS};native['shoulder_pan']=native_from_raw(1600,spec)
    r.follower.limits={n:[-180,180] for n in JOINTS};r.follower.limits['gripper']=[0,100]
    actual,target=r.command_target(native)
    common=dict(zip(JOINTS,actual))
    assert common['shoulder_pan']==pytest.approx(leader.calibrated_to_urdf(native)['shoulder_pan'],abs=.002)
    assert r.tracking(native,target,target)['tracking_error_rad']['shoulder_pan']==0
    assert r.precision_teleop_ready()
    r.leader=SimpleNamespace();assert not r.precision_teleop_ready()


def test_precision_start_with_one_robot_unvalidated_never_activates_torque():
    from soarm101_lab.real2sim.runtime import Runtime
    from unittest.mock import Mock
    r=Runtime.__new__(Runtime);r.precision_options={'use_precision':True};r.leader=SimpleNamespace()
    r.follower=SimpleNamespace(precision_mapper=PrecisionMapper(profile()),activate=Mock())
    with pytest.raises(ValueError,match='same joints'):r.start()
    r.follower.activate.assert_not_called()


def test_precision_start_large_valid_pose_difference_has_no_new_guard():
    from soarm101_lab.real2sim.runtime import Runtime
    from unittest.mock import Mock
    mapper=PrecisionMapper(profile());spec=FakeCalibrationDevice('follower').motor_spec('shoulder_pan')
    native={n:0. for n in JOINTS};native['shoulder_pan']=native_from_raw(3500,spec)
    held={**native,'shoulder_pan':native_from_raw(1500,spec)}
    r=Runtime.__new__(Runtime);r.precision_options={'use_precision':True};r.activated=False
    r.leader=SimpleNamespace(precision_mapper=mapper,read=lambda:(native,{}),activate=Mock())
    r.follower=SimpleNamespace(precision_mapper=mapper,limits={n:[0,100] if n=='gripper' else [-180,180] for n in JOINTS},read=lambda:(held,{}),activate=Mock())
    r.env=None
    r.start()
    assert r.running
    r.follower.activate.assert_called_once()


@pytest.mark.parametrize('real_only',[True,False])
def test_unknown_precision_feedback_pauses_without_goal_or_disconnect(real_only):
    from soarm101_lab.real2sim.runtime import Runtime
    from unittest.mock import Mock
    mapper=PrecisionMapper(profile());spec=FakeCalibrationDevice('follower').motor_spec('shoulder_pan')
    native={n:0. for n in JOINTS};native['shoulder_pan']=native_from_raw(3048,spec)
    r=Runtime.__new__(Runtime);r.precision_options={'use_precision':True};r.running=True;r.snapshot_data={'old':1}
    r.leader=SimpleNamespace(read=lambda:(native,{}),precision_mapper=mapper)
    r.follower=SimpleNamespace(read=lambda:(native,{}),precision_mapper=mapper,send=Mock(),close=Mock())
    r.env=None if real_only else Mock();r.backend='real' if real_only else 'mujoco'
    if real_only:r.tick_real()
    else:r.tick()
    assert not r.running and r.snapshot_data is None
    r.follower.send.assert_not_called();r.follower.close.assert_not_called()


def test_precision_snapshot_interpolates_common_coordinates():
    from soarm101_lab.real2sim.capture_dataset import align_measured
    m=PrecisionMapper(profile());spec=FakeCalibrationDevice('follower').motor_spec('shoulder_pan')
    a={n:0. for n in JOINTS};b=a.copy();a['shoulder_pan']=native_from_raw(1000,spec);b['shoulder_pan']=native_from_raw(2000,spec)
    samples=[{'timestamp_ns':0,'native':a},{'timestamp_ns':100,'native':b}]
    assert align_measured(samples,50,mapper=m)[0]==pytest.approx(0,abs=1e-12)
    from soarm101_lab.so101_joint_mapping import require_capture_mapping
    require_capture_mapping({'sim_joint_mapping':m.metadata()})


def test_passive_measurements_outside_urdf_do_not_become_motor_goals():
    from soarm101_lab.real2sim.capture_dataset import align_measured
    m=PrecisionMapper(profile());spec=m.profile['joints']['shoulder_pan']['motor']
    native={n:0. for n in JOINTS};native['shoulder_pan']=native_from_raw(2800,spec)
    with pytest.raises(ValueError,match='URDF'):m.calibrated_to_urdf(native)
    q=m.calibrated_to_observation(native)
    assert q['shoulder_pan']==pytest.approx(math.radians(117))
    assert align_measured([{'timestamp_ns':100,'native':native}],100,mapper=m,observation_only=True)[0]==pytest.approx(q['shoulder_pan'])
    with pytest.raises(ValueError,match='URDF'):m.urdf_to_calibrated(q)
    native['shoulder_pan']=native_from_raw(3048,spec)
    with pytest.raises(ValueError,match='half-turn'):m.calibrated_to_observation(native)


def test_passive_capture_refreshes_frames_outside_urdf_without_sending_goal():
    from soarm101_lab.real2sim.runtime import Runtime
    from soarm101_lab.real2sim.core import Stability
    from unittest.mock import Mock
    import time
    m=PrecisionMapper(profile());spec=m.profile['joints']['shoulder_pan']['motor']
    native={n:0. for n in JOINTS};native['shoulder_pan']=native_from_raw(2800,spec)
    timestamp=time.monotonic_ns()
    interval={'end':{'monotonic_ns':timestamp}}
    r=Runtime.__new__(Runtime);r.backend='real';r.running=False;r.precision_options={'use_precision':True}
    r.leader=SimpleNamespace(read=lambda:(native,interval),precision_mapper=m,metadata={})
    r.follower=SimpleNamespace(read=lambda:(native,interval),precision_mapper=m,send=Mock(),metadata={})
    r.real_history=Stability();r.profile={};r.hardware_identity='fixture';r.last_preview=time.monotonic()
    r.save_snapshot_requests=Mock()
    r.cameras={'real_side':SimpleNamespace(metadata={},nearest=lambda t:{'rgb':np.zeros((2,2,3),np.uint8),'sequence':1})}
    r.tick_real()
    assert r.snapshot_data is not None and r.snapshot_data['precision_observation_only']
    assert r.snapshot_data['joint_limit_warnings']['shoulder_pan']['measured_rad']==pytest.approx(math.radians(117))
    assert r.snapshot_data['tracking']['follower_measured_rad']['shoulder_pan']==pytest.approx(math.radians(117))
    assert 'real_side' in r.images and '117.00°' in r.message
    r.follower.send.assert_not_called();r.save_snapshot_requests.assert_called_once()


def test_offset_conflict_blocks_before_any_device_or_scene(tmp_path):
    from soarm101_lab.real2sim.runtime import Runtime
    from soarm101_lab.real2sim.storage import atomic
    from types import SimpleNamespace
    import json
    p=json.loads((ROOT/'tests/real2sim/fixtures/partial_profile.json').read_text())
    p['simulation_joint_offsets_rad']={'value':[.01,0,0,0,0,0],'provenance':'estimated','evidence':['test']}
    atomic(tmp_path/'profile.json',p)
    r=Runtime(SimpleNamespace(workspace=tmp_path,backend='mujoco'),None)
    with pytest.raises(ValueError,match='conflicts'):r.connect({'profile':str(tmp_path/'profile.json'),'hardware':{},'precision_options':{'use_precision':True}})
    assert r.follower is None and r.env is None
    r.writer.shutdown();r.bus.close()


def test_precision_snapshot_writer_preserves_mapping_and_native(tmp_path):
    from test_snapshots import fixture_input, observation
    from soarm101_lab.real2sim.capture_dataset import SnapshotDataset
    from soarm101_lab.real2sim.core import Stability
    import json
    scene,model,images=fixture_input(tmp_path);state=observation(scene);m=PrecisionMapper(profile())
    state['sim_joint_mapping']=m.metadata();state['sent_target']=None
    spec=FakeCalibrationDevice('follower').motor_spec('shoulder_pan');stability=Stability()
    for row in state['measured_samples']:
        row['native']['shoulder_pan']=native_from_raw(1500,spec)
        stability.add(row['timestamp_ns'],row['native'])
    state['physical_follower_measured_joint_state']=state['measured_samples'][-1]['native']
    state['real_stability']=stability.assess()
    store=SnapshotDataset(tmp_path/'snapshots','precision fixture',scene,model)
    result=store.save(state,images);manifest=json.loads((store.root/'capture_manifest.yaml').read_text())
    saved=json.loads((Path(result['path'])/'state.json').read_text())
    assert manifest['joint_mapping']['schema']=='so101.precision-joint-mapping/1'
    assert not manifest['robot']['mapping_verified']
    assert saved['q_measured'][0]==pytest.approx(0,abs=1e-12)
    assert saved['q_native']['shoulder_pan']==pytest.approx(native_from_raw(1500,spec))


def test_selected_joint_raw_and_normalized_reads_are_separate():
    from soarm101_lab.real2sim.hardware import CalibrationDevice
    d=CalibrationDevice.__new__(CalibrationDevice);fake=FakeCalibrationDevice('follower');log=[]
    class Bus:
        motors={'shoulder_pan':SimpleNamespace(id=1)}
        def read(self,name,joint,normalize=True):
            log.append((name,joint,normalize))
            return (1500 if not normalize else native_from_raw(1500,fake.motor_spec(joint))) if name=='Present_Position' else 0
        def _unnormalize(self,value):return {1:1500}
    d.bus=Bus();d.identity=fake.identity;d.motor_spec=fake.motor_spec
    sample=d.sample('shoulder_pan')
    assert log==[('Present_Position','shoulder_pan',False),('Present_Position','shoulder_pan',True),('Torque_Enable','shoulder_pan',False)]
    assert sample['raw']==1500 and sample['raw_timestamp_ns']<=sample['native_timestamp_ns']<=sample['timestamp_ns']


def test_live_angles_use_one_raw_packet_while_recording_still_checks_movement():
    from soarm101_lab.real2sim.hardware import CalibrationDevice
    fake=FakeCalibrationDevice('follower');spec=fake.motor_spec('shoulder_pan')
    d=CalibrationDevice.__new__(CalibrationDevice);reads=[]
    class MovingBus:
        motors={'shoulder_pan':SimpleNamespace(id=1)}
        def read(self,name,joint,normalize=True):
            reads.append((name,normalize))
            if name=='Torque_Enable':return 0
            return native_from_raw(1540,spec) if normalize else 1500
        def _normalize(self,raw):return {1:native_from_raw(raw[1],spec)}
        def _unnormalize(self,native):return {1:round(raw_from_native(native[1],spec))}
    d.bus=MovingBus();d.identity=fake.identity;d.motor_spec=fake.motor_spec
    sample=d.sample('shoulder_pan',live=True)
    assert sample['raw']==1500 and sample['legacy_position']==native_from_raw(1500,spec)
    assert reads==[('Present_Position',False),('Torque_Enable',False)]
    assert sample['raw_timestamp_ns']==sample['native_timestamp_ns']
    assert sample['native_source']=='normalized_same_raw_packet'
    with pytest.raises(ValueError,match='moved'):d.sample('shoulder_pan')


def test_preview_in_fresh_worker_has_no_hardware_writes_or_isaac(tmp_path):
    import subprocess,sys,os
    code='''from pathlib import Path
import sys
from soarm101_lab.real2sim.precision import Calibration
c=Calibration(sys.argv[1],sys.argv[2]);c.connect({'role':'follower','fake':True})
result=c.command('preview',{'joint':'elbow_flex','angle_rad':.4,'renderer':'egl'})
assert result['hardware_commanded'] is False
assert all(Path(result[k]).is_file() for k in ('current','reference'))
assert c.devices['follower'].writes==[]
assert not any(n.split('.')[0] in ('isaaclab','omni','pxr') for n in sys.modules)
c.close()
'''
    env={**os.environ,'PYTHONPATH':str(ROOT/'source/soarm101_lab')};env.pop('MUJOCO_GL',None)
    result=subprocess.run([sys.executable,'-B','-c',code,str(tmp_path),str(URDF)],env=env,capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('role', ['leader', 'follower'])
def test_limit_only_connect_reuses_original_profile_without_saving(tmp_path, monkeypatch, role):
    import json
    from dataclasses import make_dataclass
    from unittest.mock import Mock
    from soarm101_lab.real2sim.hardware import attach_active_precision
    from soarm101_lab.real2sim.precision import profile_for_model
    from soarm101_lab import so101_robot_spec as spec_module
    import xml.etree.ElementTree as ET
    current=URDF.read_bytes(); tree=ET.fromstring(current)
    for j in tree.findall('joint'):
        if j.get('name') in JOINTS[:4]:
            limit=j.find('limit')
            limit.set('lower',str(float(limit.get('lower'))+math.radians(5)))
            limit.set('upper',str(float(limit.get('upper'))-math.radians(5)))
    old=ET.tostring(tree); old_hash=hashlib.sha256(old).hexdigest(); new_hash=hashlib.sha256(current).hexdigest()
    evidence={old_hash:old,new_hash:current}
    original_reader=spec_module.urdf_bytes
    monkeypatch.setattr(spec_module,'urdf_bytes',lambda h:evidence[h] if h in evidence else original_reader(h))
    device=FakeCalibrationDevice(role); p=profile(role); p['urdf_sha256']=old_hash
    p['joints']['shoulder_pan']['limits']=[float(tree.find("joint[@name='shoulder_pan']/limit").get(k)) for k in ('lower','upper')]
    store=ProfileStore(tmp_path); path=store.save(role,device.identity,old_hash,p['joints'])
    store.activate(role,path,device.identity,old_hash,True)
    before={str(x):x.read_bytes() for x in tmp_path.rglob('*.json')}
    spec=p['joints']['shoulder_pan']['motor']; caltype=make_dataclass('MotorCalibration',[(k,int) for k in spec['calibration']])
    cal=caltype(**spec['calibration'])
    bus=SimpleNamespace(motors={'shoulder_pan':SimpleNamespace(id=cal.id,model='sts3215',norm_mode=SimpleNamespace(name=spec['mode']))},
        calibration={'shoulder_pan':cal}, model_resolution_table={'sts3215':spec['resolution']},apply_drive_mode=spec['apply_drive_mode'],write=Mock())
    # Exact hardware identity, with fake buses. No goal/torque API is exercised.
    saved=store.active(role); saved['identity']={'role':role,'port':str(tmp_path/'port'),'id':'test',
        'calibration_sha256':'fake','motors':{'shoulder_pan':{'id':cal.id,'model':'sts3215'}}}
    adapter=SimpleNamespace(device=SimpleNamespace(**{'_bus' if role=='leader' else 'bus':bus}),
        metadata={'port':str(tmp_path/'port'),'id':'test','calibration':{'sha256':'fake','path':'fake'}})
    from soarm101_lab.real2sim.hardware import attach_precision
    attach_precision(adapter,role,saved,new_hash)
    bound=adapter.precision_mapper
    assert bound.profile['id']==saved['id']
    assert bound.profile['joints']['shoulder_pan']['limits']==joint_specs(URDF)['shoulder_pan']['limits']
    for key in ('anchor_raw','reference_angle_rad','rad_per_count','raw_domain','references','motor'):
        assert bound.profile['joints']['shoulder_pan'][key]==saved['joints']['shoulder_pan'][key]
    q=dict.fromkeys(JOINTS,0.);q['shoulder_pan']=math.radians(112)
    native=bound.urdf_to_calibrated(q)
    assert bound.calibrated_to_urdf(native)['shoulder_pan']==pytest.approx(q['shoulder_pan'],abs=.002)
    with pytest.raises(ValueError,match='command outside URDF'):PrecisionMapper(saved).urdf_to_calibrated(q)
    assert bus.write.call_count==0
    calibration=Calibration(tmp_path,URDF,factory=lambda r,h:device)
    try:
        calibration.command('connect',{'role':role,'hardware':{}});calibration.tick()
        assert role not in calibration.errors
        assert calibration.loaded_profiles[role]==store.active(role)['id']
        assert [{k:v for k,v in ref.items() if k!='reference_id'} for ref in calibration.references[role]['shoulder_pan']]==p['joints']['shoulder_pan']['references']
        assert calibration.status()['robots'][role]['active_profile']['joints']['shoulder_pan']['limits']==joint_specs(URDF)['shoulder_pan']['limits']
        assert device.writes==[]
        with pytest.raises(ValueError,match='Unsupported'):calibration.command('rebind_limits',{})
    finally:calibration.close()
    assert before=={str(x):x.read_bytes() for x in tmp_path.rglob('*.json')}
    for needle,replacement in [(b'axis xyz="0 0 1"',b'axis xyz="0 1 0"'),(b'xyz="0.0207909',b'xyz="0.0307909')]:
        changed=current.replace(needle,replacement,1);assert changed!=current
        changed_hash=hashlib.sha256(changed).hexdigest();evidence[changed_hash]=changed
        with pytest.raises(ValueError,match='geometry/origin/axis'):profile_for_model(saved,changed_hash)
    bus.calibration['shoulder_pan'].homing_offset+=1
    with pytest.raises(ValueError,match='normalization/register'):attach_precision(adapter,role,saved,new_hash)
    assert before=={str(x):x.read_bytes() for x in tmp_path.rglob('*.json')}


@pytest.mark.parametrize('backend', ['mujoco', 'isaac'])
@pytest.mark.parametrize('mode', ['bound', 'mismatch', 'identity'])
def test_runtime_precision_bound_sim_residual_preserves_motor_mapping(tmp_path, monkeypatch, backend, mode):
    import json
    from unittest.mock import Mock
    from soarm101_lab.real2sim import runtime as module
    from soarm101_lab.real2sim import hardware
    from soarm101_lab.real2sim.storage import atomic
    saved = {role: profile(role) for role in ('leader','follower')}
    mapped = PrecisionMapper(saved['follower'])
    node = {'value':[.01,0,0,0,0,0], 'provenance':'estimated', 'evidence':['post-Precision contact fit'],
            'precision_mapping_sha256': mapped.metadata()['sha256'] if mode!='mismatch' else 'a'*64}
    data=json.loads((ROOT/'tests/real2sim/fixtures/partial_profile.json').read_text())
    data['simulation_joint_offsets_rad']=node;atomic(tmp_path/'scene.json',data)
    adapters=[]
    def adapter(*args, **kwargs):
        native=dict.fromkeys(JOINTS,0.);native['shoulder_pan']=native_from_raw(1500,saved['follower']['joints']['shoulder_pan']['motor'])
        a=SimpleNamespace(connect=Mock(),disconnect=Mock(),activate=Mock(),send=Mock(),read=lambda:(native,{}),
            metadata={'calibration':{'sha256':'fake'}},last_raw=None,motor_specs=None)
        adapters.append(a);return a
    monkeypatch.setattr(module,'Leader',adapter);monkeypatch.setattr(module,'Follower',adapter)
    monkeypatch.setattr(ProfileStore,'active',lambda store,role:copy.deepcopy(saved[role]))
    def attach(a,role,p,model_hash):
        a.precision_mapper=PrecisionMapper(p)
        a.metadata['precision_mapping']=a.precision_mapper.metadata()
    monkeypatch.setattr(hardware,'attach_precision',attach)
    def build(r,p):
        r.env=SimpleNamespace(close=Mock());r.profile=p;r.robot_hash='fake'
    monkeypatch.setattr(module.Runtime,'build_environment',build)
    monkeypatch.setattr(module.Runtime,'open_cameras',lambda *a:None)
    r=module.Runtime(SimpleNamespace(workspace=tmp_path,backend=backend),None)
    before=copy.deepcopy(saved)
    try:
        request={'profile':str(tmp_path/'scene.json'), 'hardware':{'follower_id':'fake',
            'leader_port':'/tmp/fake-leader','follower_port':'/tmp/fake-follower',
            'leader_calibration':'fake','side_camera':'/tmp/fake-side','wrist_camera':'/tmp/fake-wrist'},
            'precision_options':{'use_precision':True,'identity_sim_for_validation':mode=='identity'}}
        if mode=='mismatch':
            with pytest.raises(ValueError,match='Precision mapping used for its fit'):r.connect(request)
        else:
            r.connect(request);assert r.state=='CONNECTED'
            if mode=='bound':
                assert r.profile['simulation_joint_offsets_rad']==node
                assert 'simulation only' in r.message
            else:assert 'simulation_joint_offsets_rad' not in r.profile
            assert r.mapper().metadata()==mapped.metadata()
        for a in adapters:a.activate.assert_not_called();a.send.assert_not_called()
        assert saved==before
        assert json.loads((tmp_path/'scene.json').read_text())['simulation_joint_offsets_rad']==node
    finally:
        r.disconnect_hardware();r.writer.shutdown();r.bus.close()
