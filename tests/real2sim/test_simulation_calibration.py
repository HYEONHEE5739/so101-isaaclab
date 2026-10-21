"""Model residual policy without hardware; immutable synthetic Builder package."""
import copy,hashlib,json,math
import numpy as np
import pytest
import yaml
from soarm101_lab.real2sim import precision
from soarm101_lab.so101_joint_mapping import JOINTS


def complete_mapping(role):
    from test_precision_calibration import URDF
    device=precision.FakeCalibrationDevice(role);entries={}
    device.identity['motors']={n:{'id':i+1,'model':'sts3215'} for i,n in enumerate(JOINTS)}
    anchor=1500 if role=='leader' else 1600
    for name,definition in precision.joint_specs(URDF).items():
        refs=[dict(joint=name,raw=raw,angle_rad=math.radians(angle),split='train',calibration_hash='fake',
                   device_identity=copy.deepcopy(device.identity),motor=device.motor_spec(name))
              for raw,angle in [(anchor,0),(anchor+1000,90)]]
        entries[name]=precision.review_joint(precision.fit_joint(refs,device.motor_spec(name),name,definition['limits']))
    profile=dict(schema=precision.SCHEMA,id='fixture_'+role,role=role,identity=device.identity,
                 urdf_sha256=hashlib.sha256(URDF.read_bytes()).hexdigest(),joints=entries)
    return precision.PrecisionMapper(profile).metadata()


@pytest.fixture
def bound_workspace(tmp_path):
    from test_builder_workspace import make_export
    from soarm101_lab.real2sim.workspaces import builder,package
    root=make_export(tmp_path);scene=yaml.safe_load((root/'scene.yaml').read_text())
    mappings={role:complete_mapping(role) for role in ('leader','follower')}
    scene['robot_model']['mapping']=mappings['follower']
    scene['robot_model']['simulation_joint_offsets_rad']={'value':[0,.01,-.02,.015,0,0],
        'provenance':'estimated','evidence':['Synthetic fitting evidence; not real calibration'],
        'precision_mapping_sha256':mappings['follower']['sha256']}
    (root/'scene.yaml').write_text(yaml.safe_dump(scene));manifest=yaml.safe_load((root/'scene_manifest.yaml').read_text())
    manifest['files']['scene.yaml']=package.sha(root/'scene.yaml');(root/'scene_manifest.yaml').write_text(yaml.safe_dump(manifest))
    # USD compilation and MuJoCo execution must remain in independent processes.
    import os,sys,subprocess
    from pathlib import Path
    libs=list((Path(sys.prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages/isaacsim/extscache').glob('omni.usd.libs-*'))
    if not libs:pytest.skip('Standalone USD libraries required to compile the synthetic residual fixture')
    env={**os.environ,'PYTHONPATH':str(package.ROOT/'source/soarm101_lab')+os.pathsep+str(libs[0]),'LD_LIBRARY_PATH':str(libs[0]/'bin')}
    code='import sys; from soarm101_lab.real2sim.workspaces.builder import import_scene; print(import_scene(sys.argv[1],registry=sys.argv[2]))'
    result=subprocess.run([sys.executable,'-c',code,str(root),str(tmp_path/'workspaces')],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    w=package.load(result.stdout.strip().splitlines()[-1])
    return w,mappings


@pytest.mark.parametrize('case',['legacy_no_offset','precision_no_offset','bound','legacy_offsets','identity'])
def test_model_mode_decisions_and_role_separation(bound_workspace,case):
    w,mappings=bound_workspace;w=copy.deepcopy(w)
    if case in ('legacy_no_offset','precision_no_offset'):w['profile'].pop('simulation_joint_offsets_rad')
    if case=='legacy_offsets':w['profile']['simulation_joint_offsets_rad'].pop('precision_mapping_sha256')
    active=mappings if case in ('precision_no_offset','bound','identity') else {}
    c=precision.simulation_configuration(w,mappings=active,identity=case=='identity',hardware_role='leader',backend='mujoco')
    expected='precision_bound_residual' if case=='bound' else 'legacy_scene_offsets' if case=='legacy_offsets' else 'urdf_identity'
    assert c['model_mode']==expected and c['identity_sim']==(expected=='urdf_identity')
    assert c['simulation_residual']['applied']==(expected!='urdf_identity')
    if case=='bound':
        assert mappings['leader']['sha256']!=mappings['follower']['sha256']
        assert c['simulation_residual']['binding']=={'role':'follower','mapping_sha256':mappings['follower']['sha256']}
        assert c['robot_model']['urdf'].endswith('so101_sim.urdf')


def test_binding_mismatch_unknown_role_and_missing_collection_mapping(bound_workspace):
    w,mappings=bound_workspace;changed=copy.deepcopy(mappings)
    other=copy.deepcopy(changed['follower']['profile']);other['id']='different'
    changed['follower']=precision.PrecisionMapper(other).metadata()
    with pytest.raises(ValueError,match='mapping used for its fit'):precision.simulation_configuration(w,mappings=changed,hardware_role='leader')
    with pytest.raises(ValueError,match='approved leader'):precision.simulation_configuration(w,mappings={'follower':mappings['follower']},hardware_role='leader')
    unknown=copy.deepcopy(w);unknown['manifest'].pop('builder')
    with pytest.raises(ValueError,match='role UNKNOWN'):precision.simulation_configuration(unknown,mappings=mappings)
    assert precision.simulation_configuration(unknown,mappings=changed,identity=True)['model_mode']=='urdf_identity'


def test_recorded_contract_inherits_mode_and_old_datasets_keep_identity(bound_workspace,monkeypatch):
    from soarm101_lab.real2sim.workspaces.demo import coordinate_contract
    w,mappings=bound_workspace;c=precision.simulation_configuration(w,mappings=mappings,hardware_role='leader')
    mapper=precision.mapper_from_metadata(mappings['leader'])
    coords=precision.collection_coordinates(coordinate_contract(w),mapper,c)
    assert precision.uses_urdf_frame(coords)
    monkeypatch.setattr(precision,'active_precision',lambda *args:pytest.fail('Replay consulted current calibration'))
    replay=precision.simulation_configuration(w,coordinates=coords,backend='mujoco')
    assert replay['model_mode']=='precision_bound_residual' and replay['robot_model']==c['robot_model']
    old=precision.collection_coordinates(coordinate_contract(w),mapper)
    assert precision.simulation_configuration(w,coordinates=old)['model_mode']=='urdf_identity'
    assert precision.simulation_configuration(w,coordinates=coordinate_contract(w))['model_mode']=='legacy_scene_offsets'
    tampered=copy.deepcopy(coords);tampered['simulation_model']['model_mode']='urdf_identity'
    with pytest.raises(ValueError,match='hash mismatch'):precision.simulation_configuration(w,coordinates=tampered)


def test_real_policy_accepts_different_follower_hash_but_requires_full_model_evidence(bound_workspace):
    from soarm101_lab.real2sim.workspaces.demo import coordinate_contract
    w,mappings=bound_workspace;c=precision.simulation_configuration(w,mappings=mappings,hardware_role='leader')
    coords=precision.collection_coordinates(coordinate_contract(w),precision.mapper_from_metadata(mappings['leader']),c)
    follower=precision.mapper_from_metadata(mappings['follower'])
    precision.require_hardware_policy_frame(coords,w['profile'],follower)
    partial=copy.deepcopy(follower.profile);partial['joints'].pop('wrist_roll')
    with pytest.raises(ValueError,match='complete approved Follower'):precision.require_hardware_policy_frame(coords,w['profile'],precision.PrecisionMapper(partial))
    with pytest.raises(ValueError,match='complete approved Follower'):precision.require_hardware_policy_frame(coords,w['profile'],None)
    legacy=copy.deepcopy(w);legacy['profile']['simulation_joint_offsets_rad'].pop('precision_mapping_sha256')
    legacy_contract=precision.collection_coordinates(coordinate_contract(w),None,precision.simulation_configuration(legacy,mappings={}))
    with pytest.raises(ValueError,match='legacy Scene'):precision.require_hardware_policy_frame(legacy_contract,legacy['profile'],follower)


@pytest.mark.parametrize('active',[['cube_red'],['cube_red','cube_blue'],['cube_red','cube_blue','cube_green']])
def test_mujoco_model_residual_applied_once_without_target_adjustment(bound_workspace,tmp_path,active):
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    from soarm101_lab.workflow.robot_preview import RobotPreview
    from soarm101_lab.real2sim.workspaces.tasks import configured
    w,mappings=bound_workspace;w=configured(w,active_object_ids=active,backend='mujoco')
    c=precision.simulation_configuration(w,mappings=mappings,hardware_role='leader',backend='mujoco')
    env=MujocoBackend(w,render_backend='none',active_object_ids=active,simulation_configuration=c)
    try:
        q=env.home.copy();q[:5]=[.1,-.3,.4,.2,-.1];env.data.qpos[env.q_indices]=q;env.mj.mj_forward(env.model,env.data)
        expected=RobotPreview(w['root']/c['robot_model']['urdf']).transforms(q)['tool0']
        actual=env.world_from_workspace@env.get_eef_pose()
        np.testing.assert_allclose(actual,expected,atol=1e-9)
        env.data.ctrl[env.actuator_ids]=env._target(q)
        np.testing.assert_array_equal(env.data.ctrl[env.actuator_ids],q)
        assert env.metadata['simulation_configuration']['model_mode']=='precision_bound_residual'
    finally:env.close()


def test_source_replay_contract_and_fake_hardware_preserve_bound_geometry(bound_workspace,tmp_path,monkeypatch):
    from soarm101_lab.workflow import mujoco as worker
    from soarm101_lab.real2sim import hardware
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    monkeypatch.setattr(MujocoBackend,'get_rgb',lambda *args:np.zeros((480,640,3),np.uint8))
    w,mappings=bound_workspace
    mapper=precision.mapper_from_metadata(mappings['leader'])
    native=mapper.urdf_to_calibrated(dict(zip(JOINTS,[0,-.2,.3,.1,0,.4])))
    calls=[]
    class Leader:
        precision_mapper=mapper
        metadata={'fixture':True,'precision_mapping':mapper.metadata()}
        def __init__(self,*args,**kwargs):pass
        def connect(self):calls.append('connect')
        def activate(self):calls.append('activate')
        def read(self):return native,{}
        def close(self):calls.append('close')
    class Bridge:
        pending=set()
        polls=0
        def poll(self):
            self.polls+=1
            if self.polls>=5:self.pending.add('save')
    monkeypatch.setattr(precision,'approved_mappings',lambda *args,**kwargs:mappings)
    monkeypatch.setattr(hardware,'Leader',Leader)
    monkeypatch.setattr(worker.time,'sleep',lambda *args:None)
    request={'workspace':str(w['root']),'renderer':'none','options':{'port':'FAKE','instruction':'fixture pick',
        'episodes':1,'episode_time_s':10,'active_object_ids':['cube_red'],'width':640,'height':480}}
    source=tmp_path/'source';source.mkdir()
    worker.source(request,source,lambda *args:None,Bridge())
    from soarm101_lab.workflow.dataset import EpisodeDataset,validate_dataset
    reader=EpisodeDataset(source/'dataset.hdf5')
    validate_dataset(reader)
    p=reader.args['legacy_provenance'];c=reader.contract
    assert c['hardware_mapping']['profile']['role']=='leader'
    assert c['simulation_model']['simulation_residual']['binding']['role']=='follower'
    assert c['simulation_model']['model_mode']=='precision_bound_residual'
    assert p['scene']['simulation_configuration']==c['simulation_model']
    assert calls==['connect','activate','close']
    monkeypatch.setattr(precision,'active_precision',lambda *args:pytest.fail('Replay consulted active calibration'))
    from soarm101_lab.workflow.adapters import MujocoAdapter
    import threading
    values,attrs,_=reader.episode(reader.names[0])
    request={'workspace':str(w['root']),'input':str(reader.path),'renderer':'none','options':{'active_object_ids':reader.args['scene_configuration']['active_object_ids']}}
    with worker.simulation_environment.create_mujoco(**worker.prepare_mujoco(request,hz=reader.hz)) as env:
        adapter=MujocoAdapter(env.backend,reader.hz,tmp_path/'replay',1,threading.Event())
        adapter.restore(values,attrs)
        for index,target in enumerate(values['joint_targets']):
            np.testing.assert_allclose(adapter.state()['joint_pos'],values['obs']['policy']['joint_pos'][index],atol=1e-9)
            env.backend.step(target)
        adapter.close()


@pytest.mark.parametrize('complete',[False,True])
def test_real_eval_preflight_and_inverse_use_no_scene_offset(bound_workspace,tmp_path,monkeypatch,complete):
    import time,torch
    from soarm101_lab.workflow import evaluation
    from soarm101_lab.real2sim import hardware,cameras
    from soarm101_lab.real2sim.workspaces.demo import coordinate_contract
    from soarm101_lab.so101_dataset_contract import write_contract,JOINT_SPACE
    w,mappings=bound_workspace
    decision=precision.simulation_configuration(w,mappings=mappings,hardware_role='leader')
    contract=precision.collection_coordinates({**coordinate_contract(w),'action_space':JOINT_SPACE},precision.mapper_from_metadata(mappings['leader']),decision)
    model=tmp_path/'fake_policy';model.mkdir();write_contract(model,contract)
    follower_mapper=precision.mapper_from_metadata(mappings['follower'])
    q=np.array([0,-.2,.3,.1,0,.4]);native=follower_mapper.urdf_to_calibrated(dict(zip(JOINTS,q)))
    calls=[];sent=[]
    class Follower:
        precision_mapper=follower_mapper if complete else None
        metadata={'precision_mapping':follower_mapper.metadata() if complete else None}
        def __init__(self,*args,**kwargs):pass
        def connect(self):calls.append('connect')
        def read(self):calls.append('read');return native,{}
        def activate(self):calls.append('activate')
        def encode_goal(self,value):return value,value
        def send(self,value):sent.append(value);return value,{}
        def close(self):calls.append('close')
    class Camera:
        def __init__(self,*args):pass
        def start(self):pass
        def nearest(self):return {'rgb':np.zeros((24,32,3),np.uint8),'read_end':{'monotonic_ns':time.monotonic_ns()}}
        def close(self):pass
    class Policy:
        def reset(self):pass
        def select_action(self,batch):return torch.tensor(q,dtype=torch.float32)
    monkeypatch.setattr(hardware,'Follower',Follower);monkeypatch.setattr(cameras,'CameraReader',Camera)
    monkeypatch.setattr(evaluation,'load_policy',lambda *args:(Policy(),lambda x:x,lambda x:x))
    monkeypatch.setattr(evaluation.time,'sleep',lambda *args:None)
    output=tmp_path/'real_eval.json'
    request={'stage':'real_eval','workspace':str(w['root']),'input':{'type':'policy','id':'fake','path':str(model)},
        'output':str(output),'options':{'enable_motion':True,'port':'FAKE','follower_id':'FAKE','device':'cpu',
        'camera_devices':{'side':'FAKE_SIDE','wrist':'FAKE_WRIST'},'episodes':1,'max_steps':1}}
    if complete:
        evaluation.evaluate(request)
        assert 'activate' in calls and len(sent)==1
        expected=follower_mapper.urdf_to_calibrated(dict(zip(JOINTS,q)))
        np.testing.assert_allclose(list(sent[0].values()),list(expected.values()),atol=1e-6)
        assert json.loads(output.read_text())['hardware_mapping']['profile']['role']=='follower'
    else:
        with pytest.raises(ValueError,match='complete approved Follower'):evaluation.evaluate(request)
        assert calls==['connect','close'] and not sent and not output.exists()
