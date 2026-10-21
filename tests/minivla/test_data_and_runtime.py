import json
import numpy as np
import pytest
import torch
from soarm101_lab.learning.minivla.config import MiniVLAConfig
from soarm101_lab.learning.minivla.data import JOINTS, UNITS, load_manifest, fit_statistics, Episodes
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


def test_checkpoint_interval_config_and_legacy_resume():
    from dataclasses import asdict
    from soarm101_lab.learning.minivla.config import ARCHITECTURE_SCHEMA
    from soarm101_lab.learning.minivla.train import validate_resume
    cfg=MiniVLAConfig(steps=20,save_freq=3)
    old=asdict(cfg);old.pop('save_freq');old['steps']=10
    checkpoint=dict(architecture_schema=ARCHITECTURE_SCHEMA,config=old,statistics={},control_hz=30,step=10)
    validate_resume(checkpoint,cfg,{},30)
    for invalid in (0,-1,1.5,True):
        with pytest.raises(ValueError):MiniVLAConfig(save_freq=invalid)


@pytest.mark.parametrize('early,stop,batch_size,oom,resume',[(False,False,1,False,False),(True,False,1,False,False),(False,True,1,False,False),(False,False,8,False,False),(False,False,8,True,False),(False,False,1,False,True)])
def test_training_loop_saves_numbered_intervals_and_final(tmp_path,monkeypatch,early,stop,batch_size,oom,resume,capsys):
    """Run actual loop/optimizer/checkpoint IO with tiny CPU stand-ins, no Qwen download."""
    import contextlib,sys,types
    from soarm101_lab.learning.minivla import train as trainer
    cfg=MiniVLAConfig(steps=5,save_freq=2,val_every=1,batch_size=batch_size,accumulation=8 if resume else 1,early_stopping=early,patience=1)
    if resume:cfg.revision='test'
    config=tmp_path/'input.json';cfg.save(config)
    manifest=tmp_path/'input_manifest.json';manifest.write_text('{}')
    stats=tmp_path/'input_stats.json';stats.write_text('{}');out=tmp_path/'out'
    class Data:
        manifest={'control_hz':30};weights=torch.ones(1);arrays=[{'state':np.zeros((2,6))}]
        def __init__(self,*args):pass
        def __getitem__(self,i):return dict(state=torch.zeros(6),actions=torch.zeros(40,6),mask=torch.ones(40,dtype=torch.bool),
            images=[np.zeros((8,8,3),np.uint8)],instruction='pick red',episode='fixture',frame=i)
    class Head(torch.nn.Module):
        def __init__(self,cfg):super().__init__();self.value=torch.nn.Parameter(torch.ones(1))
        def to(self,*args):return self
    class Backbone(torch.nn.Module):
        resolved_revision='test';targets=[]
        def __init__(self,cfg):super().__init__();self.cfg=cfg
    monkeypatch.setitem(sys.modules,'soarm101_lab.learning.minivla.backbone',types.SimpleNamespace(QwenBackbone=Backbone))
    monkeypatch.setattr(trainer,'Episodes',Data);monkeypatch.setattr(trainer,'ActionHead',Head)
    monkeypatch.setattr(trainer,'dataset_identity',lambda m:'fixture')
    monkeypatch.setattr(trainer,'device_batch',lambda b,d:b)
    calls=0;observed_batches=[];backwards=[];updates=[]
    original_step=torch.optim.AdamW.step
    def update(opt,*args,**kwargs):updates.append(True);return original_step(opt,*args,**kwargs)
    monkeypatch.setattr(torch.optim.AdamW,'step',update)
    def loss(backbone,head,batch,augment):
        nonlocal calls
        calls+=1;observed_batches.append(batch['state'].shape[0])
        assert batch['actions'].shape==(batch_size,40,6) and batch['mask'].shape==(batch_size,40)
        assert len(batch['images'])==len(batch['instruction'])==batch_size
        if oom:raise torch.OutOfMemoryError('synthetic OOM; no retry permitted')
        if stop and calls==3:
            import signal
            signal.raise_signal(signal.SIGINT)
        value=head.value.square().sum();value.register_hook(lambda grad:backwards.append(grad.clone()))
        return value,{}
    monkeypatch.setattr(trainer,'compute_loss',loss)
    scores=iter([1.,2.,3.,4.,5.]);monkeypatch.setattr(trainer,'validate',lambda *a:next(scores))
    monkeypatch.setattr(torch,'autocast',lambda *a,**kw:contextlib.nullcontext())
    for name in ('reset_peak_memory_stats','synchronize'):monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ('max_memory_allocated','max_memory_reserved'):monkeypatch.setattr(torch.cuda,name,lambda:0)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'CPU fixture')
    monkeypatch.setattr(sys,'argv',['train','--config',str(config),'--manifest',str(manifest),'--stats',str(stats),'--out',str(out)])
    start=1 if resume else 0
    if resume:
        head=Head(cfg);backbone=Backbone(cfg)
        optimizer=torch.optim.AdamW([{'params':head.parameters(),'lr':cfg.head_lr,'base_lr':cfg.head_lr}])
        saved=tmp_path/'legacy.pt'
        trainer.save_checkpoint(saved,cfg,head,backbone,{},optimizer,start,float('inf'),0,30,'fixture')
        ck=torch.load(saved,weights_only=True);ck['config'].pop('batch_size');torch.save(ck,saved)
        sys.argv+=['--resume',str(saved)]
    if oom:
        with pytest.raises(torch.OutOfMemoryError,match='synthetic OOM'):trainer.main()
        assert observed_batches==[8] and not backwards and not updates and not (out/'metrics.jsonl').exists()
        return
    trainer.main()
    expected=[2,3] if stop else [2] if early else [2,4,5]
    assert sorted(p.name for p in (out/'checkpoints').glob('step_*.pt'))==[f'step_{n:06d}.pt' for n in expected]
    for step in expected:
        ck=torch.load(out/'checkpoints'/f'step_{step:06d}.pt',weights_only=True)
        assert ck['step']==step and ck['optimizer']['state'] and ck['config']['save_freq']==2
    assert torch.load(out/'last.pt',weights_only=True)['step']==(3 if stop else 2 if early else 5)
    assert (out/'best.pt').is_file()
    assert json.loads((out/'training_status.json').read_text())['stopped']==stop
    stdout=capsys.readouterr().out
    assert f'[TRAIN] step:{start+1}/5 smpl:{(start+1)*cfg.accumulation*batch_size} bs:{batch_size} loss:' in stdout and 'data_s:' in stdout and 'updt_s:' in stdout
    assert '[WORKFLOW] Validation' in stdout and '[WORKFLOW] Checkpoint saved' in stdout
    records=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
    assert all(r['target_steps']==5 and r['samples']==r['step']*cfg.accumulation*batch_size for r in records)
    assert all(r['batch_size']==batch_size and r['effective_batch_size']==batch_size*cfg.accumulation for r in records)
    assert observed_batches==[batch_size]*(len(records)*cfg.accumulation)
    assert len(backwards)==len(records)*cfg.accumulation and len(updates)==len(records)
    assert all(r['forward_backward_batches']==cfg.accumulation for r in records)
    assert all(r['update_s']>=0 and r['dataloading_s']>=0 and r['preprocessing_s']>=0 for r in records)


def test_custom_training_config_priority_and_invalid_json(tmp_path):
    from soarm101_lab.learning.minivla.config import training_config
    path=tmp_path/'custom.json'
    cfg=MiniVLAConfig(steps=17,val_every=3,save_freq=5,early_stopping=True,patience=4)
    cfg.save(path)
    selected=training_config(path,{'steps':30,'save_freq':7,'early_stopping':False})
    assert (selected.steps,selected.val_every,selected.save_freq,selected.early_stopping,selected.patience)==(30,3,7,False,4)
    path.write_text('{"unknown_typo": 1}')
    with pytest.raises(ValueError,match='설정 JSON'):training_config(path,{})
    path.write_text('{"dim": "invalid"}')
    with pytest.raises(ValueError,match='설정 JSON'):training_config(path,{})


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


def test_batch_size_default_validation_priority_and_legacy_resume(tmp_path):
    from dataclasses import asdict
    from soarm101_lab.learning.minivla.config import training_config
    from soarm101_lab.learning.minivla.train import validate_resume
    cfg=MiniVLAConfig(steps=10)
    assert cfg.batch_size==8 and cfg.accumulation==1
    for value in (0,-1,1.5,True):
        with pytest.raises(ValueError,match='batch_size'):MiniVLAConfig(batch_size=value)
    path=tmp_path/'config.json';cfg.save(path)
    assert training_config(path,{'batch_size':8}).batch_size==8
    legacy=MiniVLAConfig(steps=10,batch_size=1,accumulation=8)
    old=asdict(legacy);old.pop('batch_size');path.write_text(json.dumps(old))
    assert MiniVLAConfig.load(path).batch_size==1 and MiniVLAConfig.load(path).accumulation==8
    assert training_config(path,{'batch_size':8}).accumulation==1
    restored=training_config(path,{'resume_checkpoint':'legacy.pt','batch_size':1})
    assert restored.batch_size==1 and restored.accumulation==8
    with pytest.raises(ValueError,match='Resume Batch Size'):training_config(path,{'resume_checkpoint':'legacy.pt','batch_size':8})
    ck=dict(architecture_schema=cfg.architecture_schema,config=old,statistics={},control_hz=30,step=1)
    validate_resume(ck,legacy,{},30)
    with pytest.raises(ValueError,match='config/stats'):validate_resume(ck,MiniVLAConfig(steps=10,batch_size=8),{},30)
    cfg.batch_size=8;ck['config']=asdict(cfg)
    validate_resume(ck,cfg,{},30)
    old.pop('accumulation')
    assert MiniVLAConfig.from_dict(old).batch_size==1 and MiniVLAConfig.from_dict(old).accumulation==8
    for value in (0,-1,1.5,True):
        with pytest.raises(ValueError,match='accumulation'):MiniVLAConfig(accumulation=value)


def test_batch8_real_episodes_collate_and_compute_loss(tmp_path):
    from soarm101_lab.learning.minivla.data import collate
    from soarm101_lab.learning.minivla.train import compute_loss
    from soarm101_lab.learning.minivla.head import ActionHead
    path,_=create_data(tmp_path);stats=fit_statistics(path,tmp_path/'stats.json')
    cfg=MiniVLAConfig(vl_dim=32,dim=32,ffn=64)
    dataset=Episodes(path,stats,cfg,'train')
    batch=collate([dataset[i%5] for i in range(8)])
    assert batch['state'].shape==(8,6) and batch['actions'].shape==(8,40,6) and batch['mask'].shape==(8,40)
    assert batch['mask'].dtype==torch.bool and len(batch['images'])==len(batch['instruction'])==8
    class Features(torch.nn.Module):
        def forward(self,images,instructions,augment=False):
            assert len(images)==len(instructions)==8
            return torch.zeros(8,7,32),torch.ones(8,7,dtype=torch.bool),torch.zeros(8,7,2),torch.zeros(8,7,dtype=torch.long)
    head=ActionHead(cfg);optimizer=torch.optim.AdamW(head.parameters(),lr=cfg.head_lr)
    loss,_=compute_loss(Features(),head,batch,augment=True);loss.backward();optimizer.step()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters())
