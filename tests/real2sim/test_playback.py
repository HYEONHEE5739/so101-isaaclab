"""Playback ownership/order, manual controls and worker cleanup without hardware."""
import copy,json,subprocess,sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from soarm101_lab.simulation.playback import restore_episode,replay_steps,step_targets
from soarm101_lab.workflow import annotation,adapters,episodes
from test_common_mimic import episode


def drive(iterator,step):
    try:
        item=next(iterator)
        while True:
            if item is None:item=next(iterator);continue
            try:step(item)
            except (ValueError,RuntimeError) as exc:item=iterator.throw(exc)
            else:item=next(iterator)
    except StopIteration as done:return done.value
    finally:iterator.close()


def adapter_fixture(values):
    events=[];q=np.zeros(6);tick=[0]
    class Adapter:
        workspace={'task':{'dynamic':['cube']}};base_world=np.eye(4);step_index=0;previous=np.zeros(6)
        def restore(self,*args):events.append('restore');q[:]=0;self.step_index=0;tick[0]=0
        def state(self):events.append('state');return {'joint_pos':q.copy(),'object_poses':{'cube':np.eye(4)}}
        def update_grasp(self):self.step_index+=1;events.append('grasp')
        def step(self,command):events.append('step');q[:]=command;tick[0]+=1
    a=Adapter();a.backend=SimpleNamespace(step=a.step);return a,events


def test_shared_playback_preserves_two_phase_clock_and_observation_identity(episode):
    values,attrs,*_=episode;a,events=adapter_fixture(values);seen=[]
    restore_episode(a,values,attrs)
    assert events==['restore','state']
    def before(i):events.append('control')
    def record(i,command,state):seen.append(state);events.append('record')
    stream=replay_steps(a,values,before_step=before,on_observation=record,after_step=a.update_grasp)
    for index in range(len(values['joint_targets'])):
        command=next(stream);assert command is values['joint_targets'][index] or np.shares_memory(command,values['joint_targets'])
        assert events[-3:]==['control','state','record']
        step_targets(a,command);assert next(stream) is None
        assert events[-2:]==['step','grasp']
    with pytest.raises(StopIteration) as end:next(stream)
    assert end.value.value==dict(joint_l2_max_rad=0.,object_position_error_max_m=0.,simulation_error=None)
    assert len(seen)==len(values['actions']) and events.count('state')==len(values['actions'])+1


@pytest.mark.parametrize('error',[ValueError('physics failed'),RuntimeError('physics failed'),KeyboardInterrupt()])
def test_step_error_and_interrupt_propagation(episode,error):
    values,attrs,*_=episode;a,events=adapter_fixture(values);restore_episode(a,values,attrs)
    stream=replay_steps(a,values,after_step=a.update_grasp);next(stream)
    if isinstance(error,KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):stream.throw(error)
    else:
        with pytest.raises(StopIteration) as done:stream.throw(error)
        assert done.value.value['simulation_error']=='physics failed'
    stream.close();assert 'grasp' not in events


@pytest.mark.parametrize('mode',['mark','missing','skip','paused'])
def test_manual_controls_keep_frame_index_and_results(episode,mode,tmp_path,monkeypatch):
    values,attrs,videos,*_=episode;a,events=adapter_fixture(values);a.annotation_auto=False
    a.tracker=SimpleNamespace(completed=False);a.place_success=lambda:True
    a.check_success=lambda:a.tracker.completed;a.failure_category=lambda:'grasp_failure'
    a.record_frame=lambda *args:events.append('record')
    def grasp():a.step_index+=1;a.tracker.completed=a.step_index>=2
    a.update_grasp=grasp
    def finish(attrs,success):
        result=copy.deepcopy(values);result['obs']['events']={}
        return result,{**attrs,'annotation':json.dumps(dict(grasp=2,place=4))},videos
    a.buffer=SimpleNamespace(finish=finish);a.native=SimpleNamespace(sim=SimpleNamespace(render=lambda:events.append('render')))
    bridge=SimpleNamespace(callbacks={},details={})
    def control():
        if a.annotation_position[0]==2:
            if mode in ('mark','paused'):bridge.callbacks['mark']()
            if mode=='skip':bridge.callbacks['skip']()
    bridge.before_step=control
    if mode=='paused':
        a.annotation_paused=True
        def release(delay):assert delay==.01;events.append('pause');a.annotation_paused=False
        monkeypatch.setattr(annotation.time,'sleep',release)
    reader=SimpleNamespace(hz=30,names=['demo_0'],backend='mujoco',args={})
    row,prepared=drive(annotation.annotate_episode(a,reader,'demo_0',values,attrs,bridge),a.step)
    assert row['steps']==(2 if mode=='skip' else 5)
    if mode in ('mark','paused'):
        assert prepared and json.loads(prepared[1]['annotation'])['grasp']==2
        np.testing.assert_array_equal(prepared[0]['obs']['events']['grasp'][:,0],[0,0,1,1,1])
    else:
        assert prepared is None
        assert row['reason']==('User skipped annotation episode' if mode=='skip' else 'Mark one grasp boundary before placement')
    if mode=='paused':assert events.count('render')==events.count('pause')==1


@pytest.mark.parametrize('count',[1,2,4,8])
@pytest.mark.parametrize('cancel',[False,True])
def test_playback_iterator_with_annotation_vector_scheduler(episode,count,tmp_path,monkeypatch,cancel):
    original,attrs,*_=episode;lengths=[5,2,3,2,2,2,2,2,2,2,2];tick=[0];starts={};ends={};saved=[]
    robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros((count,6),dtype=torch.float64)))
    native=SimpleNamespace(scene={'robot':robot},device='cpu',sim=SimpleNamespace(render=lambda:None,step=lambda:None))
    def step(commands):
        if cancel and tick[0]==3:raise KeyboardInterrupt()
        robot.data.joint_pos[:]=commands;tick[0]+=1
    native.step=step
    class Adapter:
        base_world=np.eye(4);workspace={'task':{'dynamic':['cube']}}
        def __init__(self,native,index,*args):
            self.native=native;self.index=index;self.ids=list(range(6));self.seed=1047;self.metrics={};self.annotation_auto=True
            self.buffer=SimpleNamespace(video=None,metrics={},abort=lambda:None,finish=lambda a,success:({},a,{}))
        def restore(self,values,attrs):
            self.ordinal=attrs['ordinal'];self.length=lengths[self.ordinal];self.step_index=0
            self.tracker=SimpleNamespace(completed=False);robot.data.joint_pos[self.index].zero_();starts[self.ordinal]=tick[0]
        def state(self):return {'joint_pos':robot.data.joint_pos[self.index].numpy().copy(),'object_poses':{'cube':np.eye(4)}}
        def update_grasp(self):self.step_index+=1;self.tracker.completed=self.step_index>=2
        def record_frame(self,*args):pass
        def check_success(self):ends[self.ordinal]=tick[0];return True
        def place_success(self):return True
        def failure_category(self):return 'place_failure'
        def close(self):pass
    primary=Adapter(native,0)
    def read(name):
        ordinal=int(name[5:]);n=lengths[ordinal];values=copy.deepcopy(original)
        values['actions']=values['actions'][:n];values['joint_targets']=np.zeros((n,6))
        return values,{**attrs,'ordinal':ordinal},{}
    reader=SimpleNamespace(path=tmp_path/'source',names=[f'demo_{i}' for i in range(len(lengths))],hz=30,backend='isaac',args={},episode=read)
    monkeypatch.setattr(adapters,'IsaacAdapter',Adapter);monkeypatch.setattr(episodes,'source_catalog',lambda *args,**kw:({},None))
    writer=SimpleNamespace(commit=lambda v,a,s,**kw:saved.append(a['ordinal']) or f'demo_{len(saved)-1}')
    result=annotation.run(primary,reader,tmp_path,None,writer,count,input_validated=True)
    if cancel:
        assert result['stopped'] and result['annotation_progress']['phase']=='stopped'
        assert 0 not in saved and len(saved)<len(lengths)  # unfinished long slot never commits
    else:
        assert saved==list(range(len(lengths))) and result['accepted']==len(lengths)
        assert [r['steps'] for r in result['episodes']]==lengths
        if count>1:assert starts[count]==2 and starts[count]<ends[0]


@pytest.mark.parametrize('failure',['restore','step','interrupt','status'])
def test_replay_failure_cleanup_keeps_worker_owned_environment(episode,failure,tmp_path):
    from soarm101_lab.workflow.replay import run
    values,attrs,videos,*_=episode;a,events=adapter_fixture(values)
    a.buffer=SimpleNamespace(abort=lambda:events.append('abort'));a.tracker=SimpleNamespace(completed=False)
    a.check_success=lambda:False;a.place_success=lambda:False;a.failure_category=lambda:'grasp_failure'
    a.close=lambda:pytest.fail('Playback cannot close its caller-owned engine')
    reader=SimpleNamespace(names=['demo_0'],episode=lambda name:(values,attrs,videos),path=tmp_path/'input.hdf5',hz=30,backend='mujoco',args={})
    def fail(*args):raise ValueError('restore failed')
    def step(*args):
        if failure=='interrupt':raise KeyboardInterrupt()
        raise RuntimeError('step failed')
    if failure=='restore':a.restore=fail
    elif failure!='status':a.backend.step=step
    bridge=None
    if failure=='status':
        bridge=SimpleNamespace(details={},before_step=lambda:None,status=lambda:fail())
    if failure in ('restore','interrupt'):
        with pytest.raises(KeyboardInterrupt if failure=='interrupt' else ValueError):run(a,reader,tmp_path,bridge)
    else:
        result=run(a,reader,tmp_path,bridge)
        assert result['episodes'][0]['category']=='simulation_failure' and not result['passed']
    assert events[-1]=='abort' and events.count('abort')==1


def test_playback_has_no_stage_or_engine_imports():
    subprocess.run([sys.executable,'-c',"import sys;import soarm101_lab.simulation.playback;assert not any(k.startswith(('soarm101_lab.workflow','mujoco','isaaclab','omni','pxr','torch','PyQt6')) for k in sys.modules)"],check=True)


@pytest.mark.parametrize('stage,other',[('replay','annotation'),('annotation','replay')])
def test_stage_imports_are_independent_and_do_not_construct_engines(stage,other):
    code=f"import sys,importlib;importlib.import_module('soarm101_lab.workflow.{stage}');assert 'soarm101_lab.workflow.{other}' not in sys.modules;assert not any(k.startswith(('mujoco','isaaclab','omni','pxr','torch','PyQt6')) for k in sys.modules)"
    subprocess.run([sys.executable,'-c',code],check=True)
