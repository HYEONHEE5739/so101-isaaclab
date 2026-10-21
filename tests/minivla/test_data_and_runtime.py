import json
import numpy as np
import pytest
import torch
from soarm101_lab.learning.minivla.config import MiniVLAConfig
from soarm101_lab.data.legacy import JOINTS, UNITS, load_manifest
from soarm101_lab.workflow.policy import ActionGuard
from soarm101_lab.learning.minivla.evaluate import summarize_episode

def create_data(root):
    entries=[]
    for i,split in enumerate(['train','val','test']):
        d=root/f'e{i}'; d.mkdir(); t=5
        q=np.full((t,6),i*.1,np.float32)
        np.save(d/'state.npy',q); np.save(d/'action.npy',q+.1)
        np.save(d/'timestamp.npy',np.arange(t)/30)
        np.save(d/'action_timestamp.npy',np.arange(t)/30+.001)
        np.save(d/'side_timestamp.npy',np.arange(t)/30)
        np.save(d/'side.npy',np.zeros((t,32,32,3),np.uint8))
        entries.append(dict(id=f'e{i}',path=f'e{i}',split=split,group_id=f'g{i}',instruction='pick red cube'))
    m=dict(schema_version=1,joint_names=JOINTS,units=UNITS,control_hz=30,
           action_semantics='absolute_target_after_observation',episodes=entries)
    path=root/'manifest.json'; path.write_text(json.dumps(m)); return path,m



def test_guard_rate_limit_bounds_and_reset():
    g=ActionGuard([-1]*5+[0],[1]*6,[.3]*6,30)
    command=g.apply([5]*6,[0]*6)
    assert g.violations==1 and np.max(command)<=.010001
    with pytest.raises(ValueError): g.apply([float('nan')]*6,[0]*6)
    g.reset(); assert g.last is None and g.violations==0

def trace(target,distractor,grasp=False):
    return dict(target_distance=target,distractor_distance=distractor,grasp=grasp,
                transport=False,place=False,eef=[0,0,0],action=[0]*6,replan=False)

def test_semantic_success_not_grasp_success():
    r=summarize_episode([trace(.2,.3),trace(.1,.3),trace(.02,.3)],hz=30)
    assert r['grounding_success'] is True and r['failure']=='GRASP_FAIL' and not r['grasp']
    r=summarize_episode([trace(.2,.3),trace(.2,.01)],hz=30)
    assert r['failure']=='TARGET_SELECTION_FAIL'

def test_checkpoint_roundtrip(tmp_path):
    from soarm101_lab.learning.minivla.head import ActionHead
    from dataclasses import asdict
    class FrozenStub:
        cfg=MiniVLAConfig()
    c=MiniVLAConfig(vl_dim=32,dim=32,ffn=64)
    h=ActionHead(c); opt=torch.optim.AdamW(h.parameters())
    path=tmp_path/'last.pt'
    torch.save(dict(architecture_schema=c.architecture_schema,config=asdict(c),head=h.state_dict(),adapter=None,statistics={},step=1,control_hz=30),path)
    ck=torch.load(path,weights_only=True)
    restored=ActionHead(MiniVLAConfig(**ck['config'])); restored.load_state_dict(ck['head'])
    from soarm101_lab.learning.minivla.checkpoint import require_architecture
    require_architecture(ck)
    with pytest.raises(ValueError,match='Legacy'):require_architecture({'config':ck['config']})
    assert ck['step']==1 and ck['adapter'] is None
    for p,q in zip(h.parameters(),restored.parameters()): torch.testing.assert_close(p,q)


def test_inference_reads_legacy_config_and_saved_statistics_without_training(tmp_path,monkeypatch):
    from dataclasses import asdict
    import sys
    from types import SimpleNamespace
    from soarm101_lab.learning.minivla import policy
    cfg=MiniVLAConfig();saved=asdict(cfg);saved.pop('batch_size');saved['accumulation']=8
    stats={'state_mean':[1.]*6,'state_std':[2.]*6,'action_mean':[-1.]*6,'action_std':[.5]*6}
    path=tmp_path/'legacy.pt'
    torch.save({'architecture_schema':cfg.architecture_schema,'config':saved,'statistics':stats,
                'control_hz':30,'step':123,'adapter':None,'head':{'fixture':torch.ones(1)}},path)
    class Backbone:
        def __init__(self,cfg):self.cfg=cfg
        def eval(self):self.training=False
    class Head:
        def __init__(self,cfg):self.cfg=cfg
        def to(self,device):return self
        def load_state_dict(self,state):self.state=state
        def eval(self):self.training=False
    monkeypatch.setitem(sys.modules,'soarm101_lab.learning.minivla.backbone',SimpleNamespace(QwenBackbone=Backbone))
    monkeypatch.setattr(policy,'ActionHead',Head)
    restored=policy.MiniVLA(path)
    assert restored.cfg.batch_size==1 and restored.cfg.accumulation==8
    assert restored.checkpoint_step==123 and restored.control_hz==30
    assert not restored.backbone.training and not restored.head.training
    torch.testing.assert_close(restored.head.state['fixture'],torch.ones(1))
    assert restored.pre.stats==stats
    np.testing.assert_array_equal(restored.pre.normalize(np.full(6,3.),'state'),np.ones(6,np.float32))
    np.testing.assert_array_equal(restored.pre.denormalize(np.ones(6,np.float32)),np.full(6,-.5))


def test_explicit_hdf5_import(tmp_path):
    h5py=pytest.importorskip('h5py')
    from soarm101_lab.workflow.episodes import convert_hdf5 as convert
    raw=tmp_path/'raw.hdf5'
    with h5py.File(raw,'w') as f:
        a=np.full((3,6),30.,np.float32); a[:,-1]=50
        f['q']=a; f['a']=a; f['t']=np.arange(3)/30; f['at']=np.arange(3)/30+.001
        f['rgb']=np.zeros((3,32,32,3),np.uint8)
    recipe=dict(raw_joint_names=JOINTS,arm_units='deg',raw_gripper_range=[0,100],image_order='RGB',control_hz=30,
        action_semantics='absolute_target_after_observation',episodes=[dict(id='e0',file='raw.hdf5',split='train',group_id='g0',
        instruction='pick red',state_key='q',action_key='a',timestamp_key='t',action_timestamp_key='at',
        cameras={'side':dict(image_key='rgb',timestamp_key='t')})])
    (tmp_path/'recipe.json').write_text(json.dumps(recipe))
    convert(tmp_path/'recipe.json',tmp_path/'converted')
    q=np.load(tmp_path/'converted/e0/state.npy')
    assert q[0,0]==pytest.approx(np.pi/6) and q[0,-1]==.5


def test_paired_target_identity_and_noise_contract():
    from soarm101_lab.learning.minivla.evaluate import paired_action_differences
    row=dict(pair_id='same',seed=7,initial_observation_sha256='pixels+state',target_body='red',initial_chunk=[[0]*6]*40)
    blue={**row,'target_body':'blue','initial_chunk':[[1]*6]*40}
    assert paired_action_differences([row,blue])[0]['initial_action_difference_rms']==1
    with pytest.raises(ValueError,match='observation'):paired_action_differences([row,{**blue,'seed':8}])
    with pytest.raises(ValueError,match='identities'):paired_action_differences([row,row])










def test_measured_gripper_model_limits_do_not_expand_actions():
    from soarm101_lab.so101_joint_mapping import RobotCoordinates
    limits=[-.174533,1.74533]
    coords=RobotCoordinates(0,np.deg2rad(100),measured_gripper_limits=limits)
    measured=np.zeros((3,6));measured[:,-1]=[-.04528482,0,.4]
    result=coords.to_policy(measured,measured=True)
    assert result[0,5]==0 and coords.saturated_states==1
    np.testing.assert_array_equal(measured[:,-1],[-.04528482,0,.4])
    with pytest.raises(ValueError,match='canonical endpoints'):coords.to_policy(measured)
    with pytest.raises(ValueError,match='canonical endpoints'):RobotCoordinates().to_policy(measured,measured=True)
    for value in (-.18,1.8):
        invalid=np.zeros(6);invalid[5]=value
        with pytest.raises(ValueError,match='recorded model limits'):coords.to_policy(invalid,measured=True)
    assert coords.metadata()['measured_gripper_limits_rad']==limits
    for invalid in ([0,1],[-.1,float('nan')],[1,0],[-.1,2,3]):
        with pytest.raises(ValueError,match='model limits'):RobotCoordinates(measured_gripper_limits=invalid)
