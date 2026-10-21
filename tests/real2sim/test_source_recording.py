"""Source ownership, format publication/rollback and cleanup without devices."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from soarm101_lab.data import episodes as io
from soarm101_lab.data.recording import SourceEpisode
from soarm101_lab.utils.async_video_writer import VideoFrames
from test_minivla_integration import env


def descriptor():
    manifest=dict(schema_version=1,episodes=[],cameras=['side'],control_hz=30,joint_names=list(io.JOINTS),units=['rad']*5+['fraction'])
    provenance=dict(instruction='fixture',source_group_prefix='group',coordinates={'frame':'fixture'},
        scene={'dynamic_objects':['cube','other'],'pick':'cube','task_definition':{'task_id':'fixture'}},scene_configuration={})
    return manifest,provenance


def sample():
    return dict(state=np.arange(6,dtype=float),action=np.arange(6,dtype=float)+1,
        raw_state=np.arange(6,dtype=float)+2,raw_action=np.arange(6,dtype=float)+3,
        requested_action=np.arange(6,dtype=float)+4,eef_pose=np.eye(4),timestamp=np.float64(.1),
        object_poses={'cube':np.eye(4),'other':np.eye(4)})


def recorded(root):
    recorder=SourceEpisode(root,30,['side'],['cube','other'],'cube')
    recorder.begin({'native':{'positions':[1.,2.]}})
    for _ in range(3):recorder.append(sample(),{'side':np.full((24,32,3),150,np.uint8)})
    return recorder


def test_source_freezes_only_recorded_leaves_and_encoder_owns_rgb(tmp_path):
    r=SourceEpisode(tmp_path,30,['side'],['cube','other'],'cube');initial={'native':{'q':np.arange(6,dtype=float)}}
    r.begin(initial);initial['native']['q'][:]=99
    s=sample();expected=copy.deepcopy(s);pixels=np.full((24,32,3),180,np.uint8)
    try:
        r.append(s,{'side':pixels})
        for value in s.values():
            if isinstance(value,np.ndarray):value[:]=99
        for value in s['object_poses'].values():value[:]=99
        pixels[:]=0
        for key in ('state','action','raw_state','raw_action','requested_action','eef_pose'):
            np.testing.assert_array_equal(r.rows[0][key],expected[key])
        np.testing.assert_array_equal(r.initial['native']['q'],np.arange(6))
        assert r.rows[0]['object_pose'] is r.rows[0]['object_poses']['cube']
        assert not any(k in r.rows[0] for k in ('images','camera_images','env'))
        specs=r.finish()
        video=VideoFrames(r.video.folder/specs['side']['file'],1,30)
        try:assert video[0].mean()>150
        finally:video.close()
    finally:r.abort();r.abort()
    assert not list(tmp_path.glob('.video-*')) and not r.rows


@pytest.mark.parametrize('failure',['npy','manifest','rename'])
def test_legacy_failure_rolls_back_only_its_episode(tmp_path,monkeypatch,failure):
    manifest,pvd=descriptor();writer=io.LegacyEpisodeWriter(tmp_path/'dataset',manifest,pvd)
    first=recorded(writer.root);writer.save(first.rows,first.initial,first.video,{'success':True});first.abort()
    committed=(writer.root/'manifest.json').read_bytes();first_bytes=(writer.root/'episode_0000/raw_state.npy').read_bytes()
    r=recorded(writer.root);atomic=io.atomic;save=np.save;rename=Path.rename
    def fail(*args,**kwargs):raise OSError('injected failure')
    if failure=='npy':monkeypatch.setattr(np,'save',fail)
    elif failure=='manifest':
        def write(path,payload):
            if Path(path).name=='manifest.json':fail()
            atomic(path,payload)
        monkeypatch.setattr(io,'atomic',write)
    else:monkeypatch.setattr(Path,'rename',fail)
    try:
        with pytest.raises(OSError,match='injected'):writer.save(r.rows,r.initial,r.video,{'success':False})
        assert writer.count==1 and not (writer.root/'episode_0001').exists()
        assert (writer.root/'manifest.json').read_bytes()==committed
        assert (writer.root/'episode_0000/raw_state.npy').read_bytes()==first_bytes
        assert not list(writer.root.glob('.video-*'))
    finally:r.abort()
    monkeypatch.setattr(np,'save',save);monkeypatch.setattr(io,'atomic',atomic);monkeypatch.setattr(Path,'rename',rename)
    retry=recorded(writer.root)
    try:assert writer.save(retry.rows,retry.initial,retry.video,{'success':True})=='episode_0001'
    finally:retry.abort()


def test_manifest_is_last_and_source_metadata_is_detached(tmp_path,monkeypatch):
    manifest,pvd=descriptor();writer=io.LegacyEpisodeWriter(tmp_path/'dataset',manifest,pvd)
    pvd['scene']['pick']='changed';assert writer.provenance['scene']['pick']=='cube'
    original=io.atomic;events=[]
    def checked(path,payload):
        events.append(Path(path).name)
        if Path(path).name=='manifest.json':
            folder=writer.root/payload['episodes'][0]['path']
            assert all((folder/name).is_file() for name in ('side.mp4','raw_state.npy','replay.json','outcome.json'))
        return original(path,payload)
    monkeypatch.setattr(io,'atomic',checked)
    r=recorded(writer.root)
    try:writer.save(r.rows,r.initial,r.video,dict(success=True,validation_only=True))
    finally:r.abort()
    assert events[-2:]==['project_provenance.json','manifest.json']


def test_legacy_copy_failure_and_collision_preserve_original(tmp_path,monkeypatch):
    m,p=descriptor();old=io.LegacyEpisodeWriter(tmp_path/'original',m,p);r=recorded(old.root)
    old.save(r.rows,r.initial,r.video,{'success':True});r.abort()
    before=(old.root/'episode_0000/side.mp4').read_bytes();new=io.LegacyEpisodeWriter(tmp_path/'new',m,p)
    copytree=io.shutil.copytree
    def broken(src,dest,*args,**kwargs):
        Path(dest).mkdir();(Path(dest)/'partial').write_text('partial');raise OSError('copy failed')
    monkeypatch.setattr(io.shutil,'copytree',broken)
    with pytest.raises(OSError,match='copy failed'):new.copy_episodes(old.root,old.manifest)
    assert new.count==0 and not list(new.root.glob('.resume-*')) and not (new.root/'episode_0000').exists()
    assert json.loads((new.root/'manifest.json').read_text())['episodes']==[]
    assert (old.root/'episode_0000/side.mp4').read_bytes()==before
    monkeypatch.setattr(io.shutil,'copytree',copytree)
    new.copy_episodes(old.root,old.manifest);assert new.count==1
    r=recorded(new.root);collision=new.root/'episode_0001';collision.mkdir();(collision/'keep').write_text('keep')
    try:
        with pytest.raises(FileExistsError):new.save(r.rows,r.initial,r.video,{'success':True})
        assert (collision/'keep').read_text()=='keep'
    finally:r.abort()


def test_source_io_imports_no_stage_or_engine():
    code="""import sys
import soarm101_lab.data.episodes
import soarm101_lab.data.recording
assert not any(m.startswith(('soarm101_lab.workflow','PyQt6','torch','mujoco','isaaclab','omni','pxr')) for m in sys.modules)
"""
    subprocess.run([sys.executable,'-c',code],check=True)


def test_source_samples_do_not_repeat_observation_render_or_fk(env,monkeypatch,tmp_path):
    observation=env.observe();target=env.backend.home.copy();calls=[]
    # Common save supplies an explicit camera list; this Legacy fixture omits it.
    env.backend.workspace['task']['observation_cameras']=list(observation['camera_images'])
    target_pose=env.backend.target_pose;object_pose=env.backend.get_object_pose
    def forbidden(*args,**kwargs):raise AssertionError('Repeated observation/render')
    def fk(q):calls.append(('fk',q.copy()));return target_pose(q)
    def static(name):calls.append(('static',name));return object_pose(name)
    monkeypatch.setattr(env.backend,'get_state',forbidden);monkeypatch.setattr(env.backend,'get_rgb',forbidden)
    monkeypatch.setattr(env.backend,'target_pose',fk);monkeypatch.setattr(env.backend,'get_object_pose',static)
    r=SourceEpisode(tmp_path,env.hz,observation['camera_images'],env.backend.metadata['dynamic_objects'],env.backend.metadata['pick'])
    try:
        r.begin({'fixture':True})
        for _ in range(3):r.append(env.recording_sample(observation,target),observation['camera_images'])
        assert calls==[]
        geometry=env.recording_geometry(np.repeat(target[None],3,axis=0))
        assert geometry['target_poses'].shape==(3,4,4)
        assert [c[0] for c in calls]==['fk']*3+['static']*len(env.backend.metadata['target_objects'])
    finally:r.abort()


def test_legacy_rejects_foreign_staging_without_removing_it(tmp_path):
    manifest,pvd=descriptor();writer=io.LegacyEpisodeWriter(tmp_path/'dataset',manifest,pvd)
    r=recorded(tmp_path/'foreign')
    try:
        with pytest.raises(ValueError,match='private video staging'):writer.save(r.rows,r.initial,object(),{'success':True})
        with pytest.raises(ValueError,match='private video staging'):writer.save(r.rows,r.initial,r.video,{'success':True})
        assert r.video.folder.is_dir() and writer.count==0
    finally:r.abort()


def source_fixture(env,monkeypatch,common):
    from soarm101_lab.workflow import mujoco as worker
    from soarm101_lab.real2sim import hardware
    from soarm101_lab.so101_joint_mapping import DEFAULT_MAPPER,JOINTS
    closed=[]
    if common:
        env.backend.workspace['task']['observation_cameras']=['side','wrist']
        env.backend.get_rgb=lambda camera:np.zeros((480,640,3),np.uint8)
    class Leader:
        def __init__(self,*args,**kwargs):pass
        def connect(self):pass
        def activate(self):pass
        def read(self):return DEFAULT_MAPPER.urdf_to_calibrated(dict(zip(JOINTS,env.backend.home))),{}
        def close(self):closed.append(True)
    monkeypatch.setattr(hardware,'Leader',Leader)
    monkeypatch.setattr(worker,'_environment',lambda *args:env)
    monkeypatch.setattr(worker.time,'sleep',lambda *args:None)
    monkeypatch.setattr('soarm101_lab.utils.voice.log_say',lambda *args,**kwargs:None)
    return worker,closed


@pytest.mark.parametrize('common',[False,True])
def test_source_interrupt_preserves_only_completed_episode(env,monkeypatch,tmp_path,common):
    worker,closed=source_fixture(env,monkeypatch,common)
    class Bridge:
        pending=set();commands=iter([None,'discard',None,'save',None,'interrupt'])
        def poll(self):
            command=next(self.commands)
            if command=='interrupt':raise KeyboardInterrupt()
            if command:self.pending.add(command)
    report=worker.source({'options':dict(port='fixture',instruction='fixture',episodes=2,episode_time_s=10,seed=1047)},tmp_path,lambda *args:None,Bridge())
    assert closed==[True] and report['episode_count']==1 and report['stopped']
    assert not list(tmp_path.rglob('.video-*')) and not list(tmp_path.glob('*.tmp'))
    if common:
        from soarm101_lab.workflow.dataset import EpisodeDataset,validate_dataset
        reader=EpisodeDataset(report['dataset']);assert validate_dataset(reader)['episodes']==1
    else:assert len(json.loads((Path(report['dataset'])/'manifest.json').read_text())['episodes'])==1


def test_source_cleanup_closes_common_writer_and_leader_even_if_abort_fails(env,monkeypatch,tmp_path):
    worker,closed=source_fixture(env,monkeypatch,True)
    enabled=[False];writer_closed=[];abort=SourceEpisode.abort;close=io.DatasetWriter.close
    def fail_abort(self):
        abort(self)
        if enabled[0]:raise OSError('cleanup failed')
    def observed_close(self,publish=True):
        writer_closed.append(True);close(self,publish)
    monkeypatch.setattr(SourceEpisode,'abort',fail_abort);monkeypatch.setattr(io.DatasetWriter,'close',observed_close)
    class Bridge:
        pending=set()
        def poll(self):enabled[0]=True;self.pending.add('quit')
    with pytest.raises(OSError,match='cleanup failed'):
        worker.source({'options':dict(port='fixture',instruction='fixture',episodes=1,episode_time_s=10)},tmp_path,lambda *args:None,Bridge())
    assert closed==[True] and writer_closed==[True]
    assert not list(tmp_path.glob('*.tmp')) and not (tmp_path/'dataset.hdf5').exists()
