"""Scheduling fixtures plus real engine measurements in annotation_throughput."""
from types import SimpleNamespace
import json
import threading
import time

import numpy as np
import pytest
import torch

from soarm101_lab.workflow import annotation,unified,adapters,episodes


def test_paused_physics_control_cancellation_releases_waiter():
    from soarm101_lab.simulation.control import PhysicsControl, GenerationStopped
    control=PhysicsControl(1);stop=threading.Event();started=threading.Event();result=[]
    control.update(True)
    def wait():
        started.set()
        try:control.before_step(0,stop)
        except GenerationStopped:result.append('cancelled')
        else:result.append('advanced')
    worker=threading.Thread(target=wait,daemon=True);worker.start()
    assert started.wait(1)
    stop.set();worker.join(timeout=2)
    assert not worker.is_alive() and result==['cancelled']


@pytest.mark.parametrize('count',[1,2,4,8])
@pytest.mark.parametrize('with_controls',[False,True])
@pytest.mark.parametrize('reject',[False,True])
def test_vector_refills_short_episode_preserves_order_and_holds_idle(count,tmp_path,monkeypatch,with_controls,reject):
    from soarm101_lab.workflow.presentation import Presentation,send
    lengths=[12,2,3,2,2,2,2,2,2,2,2]
    commands=[];commits=[];starts={};ends={};tick=[0];post=[]
    robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros((count,6),dtype=torch.float64)))
    bridge=None;controller=None;control_checks=[]
    if with_controls:
        session=tmp_path/'session';session.mkdir()
        (session/'config.json').write_text(json.dumps(dict(stage='annotate',mode='operator')))
        bridge=Presentation(session)
        monkeypatch.setattr(bridge,'publish',lambda *a,**kw:pytest.fail('Annotation must not send images'))
        def controls():
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                status=json.loads((session/'status.json').read_text())
                if status['state']=='PAUSED':
                    control_checks.append(tick[0]);send(session,'step');break
                time.sleep(.01)
            while time.monotonic()<deadline:
                if tick[0]==4:
                    control_checks.append(tick[0]);send(session,'resume');return
                time.sleep(.01)
        controller=threading.Thread(target=controls);controller.start()
    def step(actions):
        commands.append(actions.numpy().copy());robot.data.joint_pos=actions.clone();tick[0]+=1
        if with_controls and tick[0]==3:send(session,'pause')
    native=SimpleNamespace(scene={'robot':robot},sim=SimpleNamespace(render=lambda:None,step=lambda:None),
                           device='cpu',num_envs=count,step=step)
    class Adapter:
        bind_followup=adapters.SimulationAdapter.bind_followup
        accept_followup=adapters.SimulationAdapter.accept_followup
        def __init__(self,native,index,*args):
            self.native=native;self.index=index;self.ids=list(range(6));self.seed=1
            self.annotation_auto=True;self.step_index=0;self.metrics={}
            self.tracker=SimpleNamespace(completed=False)
            self.buffer=SimpleNamespace(video=None,metrics={},abort=lambda:None)
        def state(self):return {'joint_pos':robot.data.joint_pos[self.index].numpy().copy()}
        def close(self):pass
        def place_success(self):return False
        @property
        def last_images(self):pytest.fail('Annotation executor must not consume preview RGB')
    primary=Adapter(native,0)
    monkeypatch.setattr(adapters,'IsaacAdapter',Adapter)
    monkeypatch.setattr(episodes,'source_catalog',lambda *a,**kw:({},None))
    reader=SimpleNamespace(names=[f'demo_{i}' for i in range(len(lengths))],path=tmp_path/'source.hdf5',hz=30)
    reader.episode=lambda name:({'actions':np.zeros((lengths[int(name[5:])],7))}, {'ordinal':int(name[5:]),'source_lineage':'{}'}, {})
    def replay(a,reader,name,values,attrs,*args,**kw):
        ordinal=attrs['ordinal'];a.step_index=0;starts[ordinal]=tick[0]
        for frame in range(lengths[ordinal]):
            yield np.full(6,ordinal+frame/100)
            a.step_index+=1;post.append((tick[0],ordinal))
            yield None
        ends[ordinal]=tick[0]
        failed=reject and ordinal in (1,4)
        row=dict(episode=name,steps=a.step_index,physical_replay_passed=not failed,annotation_accepted=False,category='grasp_failure' if failed else 'success')
        return row,None if failed else ({},attrs,{})
    monkeypatch.setattr(annotation,'annotate_episode',replay)
    writer=SimpleNamespace(commit=lambda v,a,s,**kw:commits.append(a['ordinal']) or f'demo_{len(commits)-1}')
    result=annotation.run(primary,reader,tmp_path,bridge,writer,count,input_validated=True)
    if controller:
        controller.join(timeout=5)
        assert not controller.is_alive() and control_checks==[3,4]
    expected=[i for i in range(len(lengths)) if not reject or i not in (1,4)]
    assert result['accepted']==len(expected) and commits==expected
    assert [r['episode'] for r in result['episodes']]==reader.names
    assert [r['steps'] for r in result['episodes']]==lengths
    assert len(post)==sum(lengths)
    if count>1:
        assert starts[count]==2 and starts[count]<ends[0]  # no episode batch barrier
        # Env0 ends last; shorter idle slots hold their final command through the tail.
        for index in range(1,count):
            assigned=[r for r in result['assignments'] if r['env_id']==index]
            if assigned:
                ordinal=assigned[-1]['ordinal'];last_tick=ends[ordinal]
                for command in commands[last_tick:]:
                    np.testing.assert_array_equal(command[index],commands[last_tick-1][index])
    assert result['input_video_decode_frames']==result['input_video_reencode_frames']==0
    assert result['annotation_progress']['completed_episodes']==len(lengths)
    assert result['annotation_progress']['saved_episodes']==len(expected)
    assert result['annotation_progress']['failed_episodes']==(2 if reject else 0)
    assert sum(w['failures'] for w in result['annotation_progress']['env_status'].values())==(2 if reject else 0)
    if bridge:
        assert bridge.details['annotation_progress']['phase']=='completed'
        assert bridge.details['annotation_progress']['finished_at'] is not None
        assert not list(session.glob('*.jpg'))


@pytest.mark.parametrize('count',[0,3,True,16])
def test_bad_annotation_environment_count_rejected_before_output(count,tmp_path):
    with pytest.raises(ValueError,match='num_envs'):
        unified.execute(dict(stage='annotate',backend='mujoco',input='missing',workspace='missing',
                             output=str(tmp_path/'new'),options=dict(num_envs=count)))
    assert not (tmp_path/'new').exists()


def test_annotation_status_counts_waiting_save_and_freezes_elapsed():
    from soarm101_lab.workflow.presentation import annotation_status_text
    job=dict(phase='paused',completed_episodes=2,total_episodes=5,saved_episodes=1,failed_episodes=0,
             pending_save_episodes=1,started_at=60,manual=True,marks=[12])
    text=annotation_status_text({'annotation_progress':job},now=150)
    assert '일시정지' in text and '저장 대기 1' in text and '1분 30초' in text and 'frame: 13' in text
    job.update(phase='completed',finished_at=120)
    assert '1분 00초' in annotation_status_text({'annotation_progress':job},now=1000)


def test_manual_parallel_is_explicit_error_before_output(tmp_path):
    with pytest.raises(ValueError,match='Manual Annotation'):
        unified.execute(dict(stage='annotate',backend='isaac',input='missing',workspace='missing',
                             output=str(tmp_path/'new'),options=dict(num_envs=4,auto=False)))
    assert not (tmp_path/'new').exists()
