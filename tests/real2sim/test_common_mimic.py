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

from soarm101_lab.workflow.dataset import (DatasetWriter,EpisodeDataset,SEMANTICS,SCHEMA,
    validate_dataset,validate_episode,recording_args)
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


def test_shared_subtask_source_cannot_leak_across_splits():
    from soarm101_lab.workflow.episodes import split_episodes
    def original(name):return {'kind':'source_episode','source_id':name,'source':{'groups':[]}}
    def entry(name,sources):return {'id':name,'group_id':name,'source_lineage':{'kind':'mimic_segments',
        'segments':[{'source':original(source)} for source in sources]}}
    entries=[entry('ab',['a','b']),entry('bc',['b','c']),entry('d',['d']),entry('e',['e'])]
    result,_=split_episodes(entries)
    split={e['id']:e['split'] for e in result}
    assert split['ab']==split['bc']
    with pytest.raises(ValueError,match='Source group leakage'):
        split_episodes(entries,{'ab':'train','bc':'val','d':'train','e':'val'})
    entries[0]['source_lineage']['segments'][0]['source']={'kind':'UNKNOWN'}
    with pytest.raises(ValueError,match='Unknown original Source'):split_episodes(entries)


def test_common_source_resume_keeps_previous_and_video_references(tmp_path,episode):
    from soarm101_lab.workflow.data_resume import combine
    from soarm101_lab.workflow.dataset import file_sha256
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
    from soarm101_lab.workflow.dataset import _write
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
    from soarm101_lab.workflow.dataset import _write,migrate,file_sha256
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
