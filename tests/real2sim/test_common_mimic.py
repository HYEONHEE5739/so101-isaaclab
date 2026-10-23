"""Semantic/data-integrity and concurrency checks for the common episode path."""
import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from soarm101_lab.workflow.dataset import EpisodeDataset, validate_dataset
from soarm101_lab.data.episodes import DatasetWriter, SEMANTICS, SCHEMA, validate_episode, recording_args
from soarm101_lab.so101_dataset_contract import current_contract,MIMIC_SPACE
from soarm101_lab.so101_joint_mapping import JOINTS


@pytest.fixture
def episode(tmp_path):
    from soarm101_lab.utils.async_video_writer import VideoEpisode
    n=5;ts=np.arange(n)/30;poses=np.repeat(np.eye(4)[None],n,axis=0)
    values={'actions':np.zeros((n,7)),'joint_targets':np.zeros((n,6)),'action_timestamp':ts,
        'obs':{'timestamp':ts[:,None],'policy':{'joint_pos':np.zeros((n,6))},
            'camera_timestamps':{'side':ts,'wrist':ts},'object_poses':{'cube':np.tile([0,0,0,1,0,0,0],(n,1))},
            'datagen_info':{'eef_pose':{'tool0':poses},'target_eef_pose':{'tool0':poses},'object_pose':{'cube':poses},
                'subtask_term_signals':{'grasp':np.array([0,0,1,1,1])[:,None]}}},
        'initial_state':{'articulation':{'robot':{'joint_position':np.zeros((1,6)),'joint_velocity':np.zeros((1,6)),
            'root_pose':np.array([[0,0,0,1,0,0,0]]),'root_velocity':np.zeros((1,6))}},
            'rigid_object':{'cube':{'root_pose':np.array([[0,0,0,1,0,0,0]]),'root_velocity':np.zeros((1,6))}}},
        'legacy':{'original_double':np.array([1+2**-40],np.float64)}}
    video=VideoEpisode(tmp_path,30)
    for i in range(n):video.append({c:np.full((480,640,3),i*20,np.uint8) for c in ('side_cam','wrist_cam')})
    specs=video.finish(n);videos={c:{**v,'file':str(video.folder/v['file'])} for c,v in specs.items()}
    contract=current_contract(MIMIC_SPACE)
    args={'backend':'mujoco','control_dt':1/30,'so101_coordinates':contract,'schema_contract':{
        'joint_order':list(JOINTS),'joint_unit':'rad','pose_frame':'robot_base','pose_format':'matrix4',
        'quaternion_order':'wxyz','action_semantics':SEMANTICS,'actions':MIMIC_SPACE}}
    yield values,{'source_group':'original','source_lineage':json.dumps({'kind':'UNKNOWN'})},videos,args,contract
    video.abort()


def test_atomic_common_copy_move_and_precision(tmp_path,episode):
    values,attrs,videos,args,contract=episode
    destination=tmp_path/'bundle/data.hdf5'
    with DatasetWriter(destination,args,contract) as writer:
        assert not destination.exists()
        assert writer.commit(values,attrs,videos)=='demo_0'
    moved=tmp_path/'moved';shutil.move(destination.parent,moved)
    reader=EpisodeDataset(moved/'data.hdf5')
    assert validate_dataset(reader,require_annotations=True)['episodes']==1
    stored,stored_attrs,_=reader.episode('demo_0')
    assert np.array_equal(stored['legacy']['original_double'],values['legacy']['original_double'])
    assert stored['obs']['datagen_info']['eef_pose']['tool0'].dtype==np.float32
    assert stored_attrs['source_group']=='original'
    with pytest.raises(FileExistsError):DatasetWriter(reader.path,args,contract)


@pytest.mark.parametrize('defect',['joint_nan','gripper','timestamp','rotation','annotation','camera_time','video_escape'])
def test_semantic_corruption_rejected(episode,tmp_path,defect):
    values,attrs,videos,_,_=copy.deepcopy(episode)
    if defect=='joint_nan':values['joint_targets'][2,0]=np.nan
    if defect=='gripper':values['actions'][0,-1]=.5
    if defect=='timestamp':values['action_timestamp'][1]=-1
    if defect=='rotation':values['obs']['datagen_info']['object_pose']['cube'][1,0,0]=2
    if defect=='annotation':values['obs']['datagen_info']['subtask_term_signals']['grasp'][3]=0
    if defect=='camera_time':values['obs']['camera_timestamps']['side'][2]+=.01
    if defect=='video_escape':videos['side_cam']['file']='../outside.mp4'
    with pytest.raises((ValueError,FileNotFoundError)):
        validate_episode(values,attrs,videos,30,require_annotations=True,video_root=tmp_path)


def test_header_unit_and_joint_order_rejected(tmp_path,episode):
    values,attrs,videos,args,contract=episode
    with DatasetWriter(tmp_path/'bad.hdf5',args,contract) as writer:
        writer.args['schema_contract']['joint_unit']='degree'
        writer.commit(values,attrs,videos)
    with pytest.raises(ValueError,match='joint_unit'):validate_dataset(EpisodeDataset(tmp_path/'bad.hdf5'))


def test_unfinished_write_is_not_published(tmp_path,episode):
    values,attrs,videos,args,contract=episode
    path=tmp_path/'failed.hdf5'
    with pytest.raises(RuntimeError):
        with DatasetWriter(path,args,contract) as writer:
            writer.commit(values,attrs,videos)
            raise RuntimeError('stop before completion')
    assert not path.exists()
    assert not list(tmp_path.glob('*.tmp'))
    assert not list((tmp_path/'failed.videos').glob('*/side_cam.mp4'))


def test_original_mimic_bodies_are_identical():
    from soarm101_lab.workflow import mimic_reference
    root=Path(mimic_reference.__file__).parent;manifest=json.loads((root/'UPSTREAM.json').read_text())
    checked=0
    for original,spec in manifest['sources'].items():
        original=Path(original)
        if not original.exists():pytest.skip('Installed reference source unavailable')
        name=original.name
        if name=='mimic_env_cfg.py':continue
        local=root/('math.py' if name=='math.py' else name)
        upstream=ast.parse(original.read_text());ported=ast.parse(local.read_text())
        if name=='math.py':
            original_functions={n.name:n for n in upstream.body if isinstance(n,ast.FunctionDef)}
            for function in ported.body:
                if isinstance(function,ast.FunctionDef):
                    assert ast.dump(function,include_attributes=False)==ast.dump(original_functions[function.name],include_attributes=False)
                    checked+=1
        else:
            def body(tree):return ast.dump(ast.Module(body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))],type_ignores=[]),include_attributes=False)
            assert body(upstream)==body(ported),name
            checked+=1
    assert checked==18


def test_scheduler_exact_global_budget_and_resume():
    from soarm101_lab.workflow.mimic import GenerationScheduler
    scheduler=GenerationScheduler(3,100,4);barrier=threading.Barrier(4);committed=[]
    def worker(index):
        attempt=scheduler.claim(index);barrier.wait()
        return scheduler.finish(index,attempt,'success',{},lambda:committed.append(attempt) or f'demo_{len(committed)-1}')
    with ThreadPoolExecutor(4) as executor:list(executor.map(worker,range(4)))
    assert scheduler.successes==3 and len(committed)==3
    assert scheduler.claim(0) is None
    resumed=GenerationScheduler(4,100,2,3,scheduler.records)
    attempt=resumed.claim(1);resumed.finish(1,attempt,'ik_failure',{'steps':0})
    attempt=resumed.claim(0);resumed.finish(0,attempt,'success',{},lambda:'demo_3')
    assert resumed.successes==4 and resumed.progress()['failure_breakdown']=={'ik_failure':1}
    assert resumed.started==6


def test_scheduler_writer_does_not_block_state_and_reserves_exact_target():
    from soarm101_lab.workflow.mimic import GenerationScheduler
    scheduler=GenerationScheduler(1,10,2);entered=threading.Event();release=threading.Event();committed=[]
    def commit():
        entered.set()
        if not release.wait(5):raise TimeoutError('Test did not release its writer')
        committed.append('demo_0');return 'demo_0'
    with ThreadPoolExecutor(2) as executor:
        first=scheduler.claim(0);saving=executor.submit(scheduler.finish,0,first,'success',{},commit)
        try:
            assert entered.wait(2)
            status=executor.submit(scheduler.progress).result(timeout=2)
            assert status['pending_successes']==1 and status['successes']==0
            second=executor.submit(scheduler.claim,1).result(timeout=2)
            extra=scheduler.finish(1,second,'success',{},lambda:pytest.fail('Reserved target exceeded'))
            assert extra['category']=='target_already_reached'
        finally:release.set()
        saving.result(timeout=2)
    assert committed==['demo_0'] and scheduler.successes==1 and scheduler.reservations==0


def test_scheduler_failed_write_rolls_back_reservation():
    from soarm101_lab.workflow.mimic import GenerationScheduler
    scheduler=GenerationScheduler(1,3,1)
    def fail():raise OSError('disk full')
    with pytest.raises(OSError,match='disk full'):
        scheduler.finish(0,scheduler.claim(0),'success',{},fail)
    status=scheduler.progress()
    assert status['successes']==status['pending_successes']==0
    assert status['failure_breakdown']=={'storage_failure':1}
    scheduler.finish(0,scheduler.claim(0),'success',{},lambda:'demo_0')
    assert scheduler.successes==1 and scheduler.progress()['completed_attempts']==2


def test_finalized_owned_video_is_adopted_without_copy(tmp_path,episode,monkeypatch):
    from soarm101_lab.utils.async_video_writer import VideoEpisode
    values,attrs,_,args,contract=episode
    stage=VideoEpisode(tmp_path,30)
    for _ in range(len(values['actions'])):
        stage.append({name:np.zeros((480,640,3),np.uint8) for name in ('side_cam','wrist_cam')})
    specs=stage.finish(len(values['actions']))
    videos={key:{**spec,'file':str(stage.folder/spec['file'])} for key,spec in specs.items()}
    inodes={key:Path(spec['file']).stat().st_ino for key,spec in videos.items()}
    def no_copy(*args,**kwargs):pytest.fail('Fresh private video must be moved, not copied')
    monkeypatch.setattr(shutil,'copyfile',no_copy)
    target=tmp_path/'adopted.hdf5'
    with DatasetWriter(target,args,contract) as writer:writer.commit(values,attrs,videos,video_staging=stage)
    stage.abort()
    reader=EpisodeDataset(target);validate_dataset(reader)
    _,_,saved=reader.episode('demo_0')
    assert all((tmp_path/spec['file']).stat().st_ino==inodes[key] for key,spec in saved.items())


def test_owned_video_commit_failure_removes_only_private_staging(tmp_path,episode,monkeypatch):
    from soarm101_lab.data import episodes as dataset
    from soarm101_lab.utils.async_video_writer import VideoEpisode
    values,attrs,original,args,contract=episode
    original_hashes={name:dataset.file_sha256(spec['file']) for name,spec in original.items()}
    stage=VideoEpisode(tmp_path,30)
    for _ in range(len(values['actions'])):
        stage.append({name:np.zeros((480,640,3),np.uint8) for name in ('side_cam','wrist_cam')})
    videos={key:{**spec,'file':str(stage.folder/spec['file'])} for key,spec in stage.finish(len(values['actions'])).items()}
    def fail(*args,**kwargs):raise OSError('write failed')
    target=tmp_path/'failed.hdf5';writer=DatasetWriter(target,args,contract)
    monkeypatch.setattr(dataset,'_write',fail)
    with pytest.raises(OSError,match='write failed'):writer.commit(values,attrs,videos,video_staging=stage)
    assert writer.count==writer.total==0 and not list(writer.data)
    writer.close(False);stage.abort()
    assert not target.exists() and not list(tmp_path.glob('failed.videos/*'))
    assert all(dataset.file_sha256(spec['file'])==original_hashes[name] for name,spec in original.items())


def test_independent_reference_random_streams_preserve_global_rng():
    from soarm101_lab.workflow.mimic import RandomStream
    import torch
    before=np.random.get_state();torch_before=torch.random.get_rng_state().clone()
    def sample(seed):
        stream=RandomStream(seed);out=[]
        for _ in range(20):
            with stream.active():out.append((int(np.random.randint(100000)),float(torch.rand(()))))
        return out
    with ThreadPoolExecutor(4) as executor:actual=list(executor.map(sample,[11,12,11,13]))
    assert actual[0]==actual[2] and actual[0]!=actual[1]
    assert np.array_equal(before[1],np.random.get_state()[1])
    assert torch.equal(torch_before,torch.random.get_rng_state())


def test_mujoco_import_does_not_load_isaac():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-c',"from soarm101_lab.workflow import datagen;import sys;assert not any(n.split('.')[0] in ('isaaclab','omni','pxr') for n in sys.modules)"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_pool_reuse_is_read_only_and_respects_selected_episodes(tmp_path,episode):
    import hashlib,pickle
    from soarm101_lab.workflow.mimic import configuration,source_pool,MimicGenerator
    values,attrs,videos,args,contract=episode
    path=tmp_path/'pool_reuse.hdf5'
    with DatasetWriter(path,args,contract) as writer:
        for _ in range(3):writer.commit(values,attrs,videos)
    env=SimpleNamespace(cfg=configuration('cube','cube'),device='cpu',actions_to_gripper_actions=lambda actions:{'tool0':actions[:,-1:]})
    env.cfg.subtask_configs['tool0'][0].subtask_term_offset_range=(0,0)
    pool=source_pool(env,path,['demo_0','demo_2'])
    assert pool.num_datagen_infos==2
    before=hashlib.sha256(pickle.dumps((pool.datagen_infos,pool.subtask_boundaries))).hexdigest()
    for seed in (11,12,11):
        reused=MimicGenerator(env,pool,seed)
        fresh=MimicGenerator(env,source_pool(env,path,['demo_0','demo_2']),seed)
        actual=reused.randomize_subtask_boundaries();expected=fresh.randomize_subtask_boundaries()
        for key in actual:np.testing.assert_array_equal(actual[key],expected[key])
    assert hashlib.sha256(pickle.dumps((pool.datagen_infos,pool.subtask_boundaries))).hexdigest()==before


@pytest.mark.parametrize('grasp,place,accepted',[(False,True,False),(True,False,False),(True,True,True)])
@pytest.mark.parametrize('manual',[False,True])
def test_common_annotation_requires_both_measured_task_signals(tmp_path,episode,grasp,place,accepted,manual,monkeypatch):
    """Synthetic signal fixture checks the gate; it does not claim physical success."""
    from soarm101_lab.workflow.annotation import run
    values,attrs,videos,args,contract=episode
    source=tmp_path/'source.hdf5'
    with DatasetWriter(source,args,contract) as writer:writer.commit(values,attrs,videos)
    reader=EpisodeDataset(source);calls=[]
    class Buffer:
        video=None;metrics={}
        def abort(self):pass
        def finish(self,attrs,*,success):
            assert success == (grasp and place)
            result=copy.deepcopy(values);result['obs'].setdefault('events',{})
            return result,dict(attrs,annotation=json.dumps({'grasp':2,'place':4})),videos
    from soarm101_lab.workflow.adapters import SimulationAdapter
    class Adapter:
        accept_followup=SimulationAdapter.accept_followup
        workspace={'task':{'dynamic':['cube']}};base_world=np.eye(4);step_index=0
        annotation_auto=True;buffer=Buffer();tracker=SimpleNamespace(completed=False);metrics={};last_images={}
        def close(self):pytest.fail('Annotation must not close its caller-owned primary adapter')
        def restore(self,*args):self.tracker.completed=False;self.step_index=0
        def state(self):return {'joint_pos':np.zeros(6),'object_poses':{'cube':np.eye(4)}}
        def step(self,*args):self.step_index+=1
        def update_grasp(self):self.tracker.completed=grasp
        def record_frame(self,*args):pass
        def place_success(self):return place
        def check_success(self):return self.tracker.completed and self.place_success()
        def failure_category(self):return 'grasp_failure' if not self.tracker.completed else 'place_failure'
    adapter=Adapter();adapter.backend=SimpleNamespace(step=adapter.step,render_metrics={})
    adapter.annotation_auto=not manual;adapter.realtime=True
    bridge=None
    if manual:
        class Bridge:
            callbacks={};details={};steps=0;config={};progress_provider=None
            def before_step(self):
                if adapter.annotation_position[0]==2:self.callbacks['mark']()
            def status(self,*args):pass
            def poll(self):pass
        bridge=Bridge();bridge.paused=False;bridge.single_step=False
    def commit(result,*args,**kwargs):
        if manual:np.testing.assert_array_equal(result['obs']['events']['grasp'][:,0],[0,0,1,1,1])
        calls.append(True);return 'demo_0'
    writer=SimpleNamespace(commit=commit)
    from soarm101_lab.workflow import unified
    monkeypatch.setattr(unified.time,'sleep',lambda *args:pytest.fail('Annotation must not pace by control Hz'))
    result=run(adapter,reader,tmp_path/'replayed',bridge,writer,1)
    assert result['passed']==accepted and result['accepted']==int(accepted)
    assert bool(calls)==accepted


@pytest.mark.parametrize('stage',['annotate','datagen'])
def test_valid_common_input_reaches_engine_without_legacy_import(stage,tmp_path,episode,monkeypatch):
    from soarm101_lab.workflow import unified,mujoco
    from soarm101_lab.workflow.operations import execute_request
    from soarm101_lab.real2sim.workspaces import package
    values,attrs,videos,args,contract=episode
    identity={'sha256':'immutable workspace'}
    args={**args,'workspace':identity,'scene_configuration':{'task_definition':{},'active_object_ids':['cube'],'reset':{}}}
    path=tmp_path/'common.hdf5'
    with DatasetWriter(path,args,contract) as writer:writer.commit(values,attrs,videos)
    monkeypatch.setattr(package,'load',lambda *args:{})
    monkeypatch.setattr(package,'metadata',lambda *args:identity)
    monkeypatch.setattr(unified,'migrate',lambda *args:pytest.fail('Common input must not be migrated'))
    def engine(*args,**kwargs):raise RuntimeError('Validated common input reached the physics boundary')
    from soarm101_lab.workflow import worker_request
    monkeypatch.setattr(worker_request,'episode_context',lambda *a,**kw:{'reader':EpisodeDataset(path),
        'workspace':{'root':tmp_path/'immutable'},'scene':{**args['scene_configuration'],
            'task_definition':{'language_instruction':'fixture'}},'tasks':[{}],'followup':None})
    monkeypatch.setattr(worker_request,'prepare_mujoco',lambda *args,**kwargs:{})
    monkeypatch.setattr(mujoco.simulation_environment,'create_mujoco',engine)
    request={'backend':'mujoco','stage':stage,'input':str(path),'workspace':'fixture',
        'output':str(tmp_path/stage),'options':{'generation_seed':1047}}
    with pytest.raises(RuntimeError,match='reached the physics boundary'):execute_request(request)
    assert json.loads((tmp_path/stage/'failure.json').read_text())['schema_compatible']


def test_datagen_cancel_before_trials_reports_stopped_without_false_success(tmp_path,episode,monkeypatch):
    from soarm101_lab.workflow import unified,mujoco
    from soarm101_lab.real2sim.workspaces import package
    values,attrs,videos,args,contract=episode;identity={'sha256':'immutable workspace'}
    args={**args,'workspace':identity,'scene_configuration':{'task_definition':{},'active_object_ids':['cube'],'reset':{}}}
    path=tmp_path/'source.hdf5'
    with DatasetWriter(path,args,contract) as writer:writer.commit(values,attrs,videos)
    monkeypatch.setattr(package,'load',lambda *args:{})
    monkeypatch.setattr(package,'metadata',lambda *args:identity)
    def cancel(*args,**kwargs):raise KeyboardInterrupt()
    from soarm101_lab.workflow import worker_request
    monkeypatch.setattr(worker_request,'episode_context',lambda *a,**kw:{'reader':EpisodeDataset(path), 'workspace':{'root':tmp_path/'immutable'},'scene':{**args['scene_configuration'],'task_definition':{'language_instruction':'fixture'}},'tasks':[{}],'followup':None})
    monkeypatch.setattr(worker_request,'prepare_mujoco',lambda *args,**kwargs:{})
    monkeypatch.setattr(mujoco.simulation_environment,'create_mujoco',cancel)
    result=unified.execute({'backend':'mujoco','stage':'datagen','input':str(path),'workspace':'fixture',
        'output':str(tmp_path/'run'),'options':{'generation_seed':1047,'num_successful_demos':100}})
    assert result['stopped'] and result['successes']==0 and not result['passed'] and result['dataset'] is None
    assert not (tmp_path/'run/failure.json').exists()


@pytest.mark.parametrize('backend,source_backend',[
    ('isaac','mujoco'),('mujoco','isaac'),('isaac','isaac'),('mujoco','mujoco')])
def test_datagen_starts_full_pool_without_automatic_source_replay(backend,source_backend,tmp_path,episode,monkeypatch):
    """14-episode contract fixture, stubbed engines: no physical preflight may run."""
    import sys
    from soarm101_lab.workflow import unified,mujoco,adapters,datagen,presentation
    from soarm101_lab.real2sim.workspaces import package
    values,attrs,videos,args,contract=episode;identity={'sha256':'immutable workspace'}
    args={**args,'backend':source_backend,'workspace':identity,
        'scene_configuration':{'task_definition':{},'active_object_ids':['cube'],'reset':{}}}
    source=tmp_path/'source.hdf5'
    with DatasetWriter(source,args,contract) as writer:
        for _ in range(14):writer.commit(values,attrs,videos)
    monkeypatch.setattr(package,'load',lambda *args:{})
    monkeypatch.setattr(package,'metadata',lambda *args:identity)
    from soarm101_lab.workflow import annotation
    monkeypatch.setattr(annotation,'run',lambda *args,**kw:pytest.fail('DataGen must not annotate the Source pool'))
    native=SimpleNamespace(backend=object(),close=lambda:None)
    adapter=SimpleNamespace(buffer=SimpleNamespace(abort=lambda:None),close=lambda:None,bind_followup=lambda runtime:None)
    from soarm101_lab.simulation import environment
    camera_requests=[]
    def create_isaac(*args,**kwargs):
        camera_requests.append(kwargs['overview_camera'])
        return native
    monkeypatch.setattr(environment,'create_isaac',create_isaac)
    from soarm101_lab.workflow import worker_request
    monkeypatch.setattr(worker_request,'episode_context',lambda *a,**kw:{'reader':EpisodeDataset(source), 'workspace':{'root':tmp_path/'immutable'},'scene':{**args['scene_configuration'],'task_definition':{'language_instruction':'fixture'}},'tasks':[{}],'followup':None})
    monkeypatch.setattr(worker_request,'prepare_mujoco',lambda *args,**kwargs:{})
    monkeypatch.setattr(mujoco.simulation_environment,'create_mujoco',lambda *args,**kwargs:native)
    monkeypatch.setattr(adapters,'IsaacAdapter',lambda *args:adapter)
    monkeypatch.setattr(adapters,'MujocoAdapter',lambda *args:adapter)
    monkeypatch.setitem(sys.modules,'isaaclab.app',SimpleNamespace(AppLauncher=lambda **kwargs:SimpleNamespace(app=native)))
    session=tmp_path/'session';session.mkdir()
    (session/'config.json').write_text(json.dumps({'stage':'datagen','mode':'operator'}))
    bridge=presentation.Presentation(session)
    monkeypatch.setattr(presentation,'current',lambda:bridge)
    monkeypatch.setattr(presentation,'attach',lambda *args:bridge)
    calls=[]
    def generate(request,reader,output,active_bridge):
        assert reader.backend==source_backend and len(reader.names)==14
        assert request['options']['num_envs']==4
        assert active_bridge.details['workflow_phase']=='preparing'
        assert 'source_total' not in active_bridge.details
        active_bridge.bind_generation(SimpleNamespace(progress=lambda:{'successes':0,'attempts':0}))
        calls.append(backend)
        return {'passed':True,'successes':2}
    monkeypatch.setattr(datagen,'run_isaac',lambda native,request,reader,output,bridge,boundary:
        generate(request,reader,output,bridge))
    monkeypatch.setattr(datagen,'run_mujoco',lambda request,reader,output,bridge,**kwargs:
        generate(request,reader,output,bridge))
    result=unified.execute({'backend':backend,'stage':'datagen','input':str(source),'workspace':'fixture',
        'output':str(tmp_path/'generated'),'options':{'generation_seed':1047,'num_envs':4,'num_successful_demos':2}})
    assert calls==[backend] and result['passed']
    assert camera_requests==([True] if backend=='isaac' else [])
    assert 'cross_replay_cache' not in result
    assert not (tmp_path/'generated/cross_replay.json').exists()
    status=json.loads((session/'status.json').read_text())
    assert status['workflow_phase']=='generation' and 'source_total' not in status




def test_standalone_manual_annotation_keyboard_uses_common_replay_state(monkeypatch):
    import sys
    from soarm101_lab.workflow.presentation import annotation_keyboard
    class Keyboard:
        def __init__(self,*args):self.callbacks={}
        def add_callback(self,key,callback):self.callbacks[key]=callback
        def reset(self):pass
    monkeypatch.setitem(sys.modules,'isaaclab.devices',SimpleNamespace(Se3Keyboard=Keyboard,Se3KeyboardCfg=lambda **kwargs:None))
    adapter=SimpleNamespace(annotation_marks=[],annotation_position=[7],annotation_skipped=[])
    keyboard=annotation_keyboard(adapter)
    assert adapter.annotation_paused
    keyboard.callbacks['N']();assert not adapter.annotation_paused
    keyboard.callbacks['B']();assert adapter.annotation_paused
    keyboard.callbacks['S']();assert adapter.annotation_marks==[7]
    keyboard.callbacks['Q']();assert adapter.annotation_skipped==[True]




def test_common_source_resume_keeps_previous_and_video_references(tmp_path,episode):
    from soarm101_lab.workflow.data_resume import combine
    from soarm101_lab.data.episodes import file_sha256
    values,attrs,videos,args,contract=episode
    old=tmp_path/'old.hdf5';new=tmp_path/'new.hdf5'
    for path in (old,new):
        with DatasetWriter(path,args,contract) as writer:writer.commit(values,attrs,videos)
    before=file_sha256(old)
    combine(old,new)
    assert file_sha256(old)==before
    assert validate_dataset(EpisodeDataset(new))['episodes']==2



def test_official_pool_order_and_trace_lineage_are_consistent(tmp_path,episode):
    import h5py
    from soarm101_lab.data.episodes import _write
    from soarm101_lab.workflow.datagen import _lineage
    from soarm101_lab.workflow.mimic_reference.dataset import HDF5DatasetFileHandler
    values,attrs,_,args,_=episode
    path=tmp_path/'pool.hdf5'
    with h5py.File(path,'w') as h:
        data=h.create_group('data');data.attrs['env_args']=json.dumps({**args,'dataset_schema':SCHEMA})
        for index in range(14):
            name=f'demo_{index}';group=data.create_group(name);_write(group,values)
            group.attrs['source_group']=name
            group.attrs['source_lineage']=json.dumps({'kind':'source_episode','source_id':'source:'+name,
                'source':{'groups':[name],'dataset_sha256':'original','episode_id':name}})
    reader=EpisodeDataset(path);handler=HDF5DatasetFileHandler();handler.open(path)
    assert reader.names[2]=='demo_2'
    assert handler.get_episode_names()[2]=='demo_10'
    lineage=_lineage(reader,SimpleNamespace(selection_trace=[{'source_index':2,'subtask':0,'boundary':[0,3]}]))
    assert lineage['segments'][0]['episode']=='demo_10'
    assert lineage['segments'][0]['source']['source_id']=='source:demo_10'


def test_legacy_internal_and_external_rgb_import_across_directories(tmp_path,episode,monkeypatch):
    import h5py
    from soarm101_lab.workflow.dataset import migrate
    from soarm101_lab.data.episodes import _write, file_sha256
    values,attrs,videos,args,contract=copy.deepcopy(episode)
    folder=tmp_path/'legacy';folder.mkdir();source=folder/'source.hdf5';destination=tmp_path/'export/common.hdf5'
    wrist=folder/'wrist.mp4';shutil.copyfile(videos['wrist_cam']['file'],wrist)
    values['obs']['policy']['side_cam']=np.zeros((5,480,640,3),np.uint8)
    with h5py.File(source,'w') as h:
        data=h.create_group('data');data.attrs['env_args']=json.dumps({**args,'backend':'isaac'})
        group=data.create_group('demo_0');_write(group,values)
        group.attrs['source_group']='original'
        group.attrs['external_videos']=json.dumps({'wrist_cam':{**videos['wrist_cam'],'file':'wrist.mp4'}})
    # Isolate storage migration; frame normalization has separate actual PhysX evidence.
    monkeypatch.setattr(EpisodeDataset,'_isaac_episode',lambda self,v,a,c:(v,a,c))
    before=file_sha256(source)
    migrate(source,destination)
    reader=EpisodeDataset(destination);_,_,copied=reader.episode('demo_0')
    assert validate_dataset(reader)['episodes']==1
    assert file_sha256(destination.parent/copied['wrist_cam']['file'])==file_sha256(wrist)
    assert file_sha256(source)==before


def test_resume_history_travels_with_hdf5_bundle(tmp_path,episode):
    from soarm101_lab.workflow.datagen import _resume
    values,attrs,videos,args,contract=episode
    args={**args,'generation_seed':9,'generation_provenance':{'source_sha256':'same-source-pool'}}
    first=tmp_path/'original/data.hdf5'
    history=[{'attempt':1,'env_id':0,'category':'success','episode':'demo_0'}]
    with DatasetWriter(first,args,contract) as writer:
        writer.commit(values,attrs,videos);writer.record_attempts(history)
    moved=tmp_path/'moved';shutil.move(first.parent,moved)
    assert not (moved/'attempts.json').exists()
    with DatasetWriter(tmp_path/'resumed.hdf5',args,contract) as writer:
        count,actual=_resume(writer,{'resume_data':str(moved/'data.hdf5'),'generation_seed':9})
        assert count==1 and actual==history and writer.count==1


def test_resume_rejects_solver_mix_before_copy(tmp_path,episode):
    from soarm101_lab.workflow.datagen import _resume
    from soarm101_lab.simulation.ik import solver_configuration
    values,attrs,videos,args,contract=episode
    args={**args,'generation_seed':9,'generation_provenance':{'source_sha256':'same'}}
    source=tmp_path/'source.hdf5'
    with DatasetWriter(source,args,contract) as writer:writer.commit(values,attrs,videos)
    new={**args,'generation_configuration':{'ik_solver':'pinocchio_dls','ik_configuration':solver_configuration('pinocchio_dls')}}
    with DatasetWriter(tmp_path/'new.hdf5',new,contract) as writer:
        with pytest.raises(ValueError,match='IK solver differs'):
            _resume(writer,{'resume_data':str(source),'generation_seed':9})
        assert writer.count==0


def test_resume_rejects_changed_ik_parameters(tmp_path,episode):
    from soarm101_lab.workflow.datagen import _resume
    from soarm101_lab.simulation.ik import solver_configuration
    values,attrs,videos,args,contract=episode
    config=solver_configuration('scipy_ls')
    args={**args,'generation_seed':9,'generation_provenance':{'source_sha256':'same'},
          'generation_configuration':{'ik_solver':'scipy_ls','ik_configuration':config}}
    source=tmp_path/'source.hdf5'
    with DatasetWriter(source,args,contract) as writer:writer.commit(values,attrs,videos)
    import copy
    changed=copy.deepcopy(args);changed['generation_configuration']['ik_configuration']['parameters']['max_nfev']=81
    with DatasetWriter(tmp_path/'new.hdf5',changed,contract) as writer:
        with pytest.raises(ValueError,match='IK configuration differs'):
            _resume(writer,{'resume_data':str(source),'generation_seed':9})
        assert writer.count==0


def test_resume_failure_policy_mismatch_and_old_strict_compatibility(tmp_path,episode):
    from soarm101_lab.workflow.datagen import _resume
    from soarm101_lab.simulation.ik import policy_configuration
    values,attrs,videos,args,contract=episode
    args={**args,'generation_seed':9,'generation_provenance':{'source_sha256':'same'}}
    source=tmp_path/'source.hdf5'
    with DatasetWriter(source,args,contract) as writer:writer.commit(values,attrs,videos)
    for policy in ('strict','hold','best_effort'):
        new={**args,'generation_configuration':{'ik_failure_policy':policy,'ik_policy_configuration':policy_configuration(policy)}}
        with DatasetWriter(tmp_path/(policy+'.hdf5'),new,contract) as writer:
            if policy=='strict':assert _resume(writer,{'resume_data':str(source),'generation_seed':9})[0]==1
            else:
                with pytest.raises(ValueError,match='failure policy differs'):
                    _resume(writer,{'resume_data':str(source),'generation_seed':9})
                assert writer.count==0


def test_invalid_ik_candidate_progress_has_no_fake_residual():
    from soarm101_lab.workflow.mimic import GenerationScheduler
    scheduler=GenerationScheduler(1,2,1)
    attempt=scheduler.claim(0)
    scheduler.finish(0,attempt,'ik_failure',{'steps':0,'ik':{
        'ik_invalid_candidate':True,'position_error_m':None,'rotation_error_rad':None,
        'failure_reason':'candidate_nonfinite'}},lambda:None)
    status=scheduler.progress()
    assert status['failures']==1 and status['successes']==0
    assert 'candidate_nonfinite' in status['outcome_reason'] and 'mm' not in status['outcome_reason']


def test_drop_diagnostic_distinguishes_release_and_sustained_loss():
    from types import SimpleNamespace
    from soarm101_lab.workflow.adapters import SimulationAdapter
    import numpy as np
    def adapter():
        a=object.__new__(SimulationAdapter);a.hz=30;a.previous=np.zeros(6);a.step_index=0
        a.tracker=SimpleNamespace(completed=True,details={})
        a.grasp_reference=None;a.grasp_command=None;a.release_commanded=False;a.loss_steps=0;a.drop_step=None
        a._update_failure_evidence(np.zeros(3),True,True)
        return a
    a=adapter()
    for i in range(5):
        a.step_index=i+1;a._update_failure_evidence(np.array([.02,0,0]),False,False)
    assert a.failure_category()=='place_failure'
    a.step_index=6;a._update_failure_evidence(np.array([.02,0,0]),False,False)
    assert a.failure_category()=='drop_failure' and a.failure_diagnostics()['drop_step']==6
    a=adapter();a.previous[5]=.1
    for i in range(10):a._update_failure_evidence(np.array([.02,0,0]),False,False)
    assert a.failure_category()=='place_failure' and a.release_commanded
    a.tracker.completed=False
    assert a.failure_category()=='grasp_failure'


def test_target_completion_time_stops_at_successful_commit(monkeypatch):
    from soarm101_lab.workflow import mimic
    clock=[100.];monkeypatch.setattr(mimic.time,'perf_counter',lambda:clock[0])
    scheduler=mimic.GenerationScheduler(1,10,1);attempt=scheduler.claim(0)
    clock[0]=165.;scheduler.finish(0,attempt,'success',{},lambda:'demo_0')
    clock[0]=200.
    assert scheduler.progress()['target_completed_seconds']==65.
    assert scheduler.progress()['wall_seconds']==100.
