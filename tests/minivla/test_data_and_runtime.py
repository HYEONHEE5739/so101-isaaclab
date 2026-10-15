import json
import numpy as np
import pytest
import torch
from soarm101_lab.learning.minivla.config import MiniVLAConfig
from soarm101_lab.learning.minivla.data import JOINTS, UNITS, load_manifest, fit_statistics, Episodes
from soarm101_lab.learning.minivla.policy import ActionGuard
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

def test_train_only_stats_padding_and_normalization(tmp_path):
    path,m=create_data(tmp_path); stats=fit_statistics(path,tmp_path/'stats.json')
    assert np.allclose(stats['state_mean'],0) and np.allclose(stats['action_mean'],.1)
    ds=Episodes(path,stats,MiniVLAConfig(),'train'); sample=ds[4]
    assert sample['mask'].sum()==1 and torch.all(sample['actions'][1:]==0)
    raw=np.random.randn(6).astype(np.float32)
    np.testing.assert_allclose(ds.pre.denormalize(ds.pre.normalize(raw,'action')),raw,rtol=1e-5,atol=1e-6)

def test_group_leakage_and_alignment_rejected(tmp_path):
    path,m=create_data(tmp_path); m['episodes'][1]['group_id']='g0'; path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='leakage'): load_manifest(path)
    m['episodes'][1]['group_id']='g1'; path.write_text(json.dumps(m))
    np.save(tmp_path/'e0/action_timestamp.npy',np.arange(5)/30-.001)
    with pytest.raises(ValueError,match='follow'): fit_statistics(path,tmp_path/'stats.json')

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
    from soarm101_lab.learning.minivla.train import save_checkpoint
    class FrozenStub:
        cfg=MiniVLAConfig()
    c=MiniVLAConfig(vl_dim=32,dim=32,ffn=64)
    h=ActionHead(c); opt=torch.optim.AdamW(h.parameters())
    path=tmp_path/'last.pt'
    save_checkpoint(path,c,h,FrozenStub(),{},opt,1,.5,0,30)
    ck=torch.load(path,weights_only=True)
    restored=ActionHead(MiniVLAConfig(**ck['config'])); restored.load_state_dict(ck['head'])
    from soarm101_lab.learning.minivla.train import require_architecture
    require_architecture(ck)
    with pytest.raises(ValueError,match='Legacy'):require_architecture({'config':ck['config']})
    assert ck['step']==1 and ck['adapter'] is None
    for p,q in zip(h.parameters(),restored.parameters()): torch.testing.assert_close(p,q)

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


def test_resume_detects_canonical_rgb_change_without_changing_statistics(tmp_path):
    from dataclasses import asdict
    from soarm101_lab.learning.minivla.data import dataset_identity
    from soarm101_lab.learning.minivla.train import validate_resume
    path,_=create_data(tmp_path);stats=fit_statistics(path,tmp_path/'stats.json');m=load_manifest(path)
    identity=dataset_identity(m);cfg=MiniVLAConfig(steps=3)
    ck={'architecture_schema':cfg.architecture_schema,'config':asdict(cfg),'statistics':stats,'control_hz':30,'step':2,'dataset_identity':identity}
    validate_resume(ck,cfg,stats,30,identity)
    pixels=np.load(tmp_path/'e0/side.npy');pixels[0,0,0,0]=123;np.save(tmp_path/'e0/side.npy',pixels)
    with pytest.raises(ValueError,match='byte identity'):validate_resume(ck,cfg,stats,30,dataset_identity(load_manifest(path)))
