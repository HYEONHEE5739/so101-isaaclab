"""Reader ownership, bounded I/O and alignment without Workspace/Qt/engines."""
from collections import Counter
import subprocess
import sys
import h5py
import numpy as np
import pytest
from soarm101_lab.data.episodes import EpisodeReader


@pytest.fixture
def row(tmp_path):
    path=tmp_path/'episode.hdf5';n=20
    with h5py.File(path,'w') as f:
        g=f.create_group('data/demo_0');g['time']=np.arange(n)/30
        g['q']=np.arange(n*6,dtype=np.float32).reshape(n,6)/100
        g['action']=g['q'][...]+.01;g['ee']=np.zeros((n,3),np.float32)
        poses=np.repeat(np.eye(4)[None],n,axis=0);poses[:,0,3]=np.arange(n)/100
        g['object']=poses
        for camera in ('side','wrist','overview'):
            g[camera]=np.broadcast_to(np.arange(n,dtype=np.uint8)[:,None,None,None],(n,4,5,3))
        g['wrong_time']=np.arange(n)/30+.1
    return dict(data_ref=dict(path=str(path),episode='demo_0'),frames=n,dt=1/30,
        pose_transforms={'robot_base':np.eye(4)},display_frame='robot_base',
        timestamp_key='time',action_timestamp_key='time',q_key='q',action_key='action',
        ee_key='ee',ee_frame='robot_base',recorded_object_ids=['cube'],
        object_telemetry={'cube':dict(path='object',timestamp_path='time',frame='robot_base')},
        cameras={c:dict(path=c,timestamp_key='time') for c in ('side','wrist','overview')})


def test_lazy_open_one_handle_and_no_implicit_rgb(row,monkeypatch):
    opened=[];reads=Counter();real_file=h5py.File
    def open_file(*args,**kwargs):
        handle=real_file(*args,**kwargs);opened.append(handle);return handle
    monkeypatch.setattr(h5py,'File',open_file)
    with EpisodeReader(row) as reader:
        assert not opened
        original=reader.read
        def read(key,selection=slice(None)):
            reads[key]+=1;return original(key,selection)
        monkeypatch.setattr(reader,'read',read)
        for i in (0,3,10):
            frame=reader.frame(i)
            np.testing.assert_array_equal(frame['robot_q'],np.arange(i*6,i*6+6,dtype=np.float32)/100)
            assert frame['action_timestamp']==frame['timestamp']==pytest.approx(i/30)
            assert not frame['rgb_frames']
        assert len(opened)==1 and reads['time']==3
        assert not any(reads[c] for c in row['cameras'])
        handle=opened[0];assert handle.id.valid
    assert not handle.id.valid
    reader.close()
    with pytest.raises(RuntimeError,match='closed'):reader.frame(0)
    with pytest.raises(RuntimeError,match='closed'):reader.read('q')


def test_error_closes_handle_and_preserves_input(row):
    from pathlib import Path
    path=Path(row['data_ref']['path']);before=path.read_bytes();handle=None
    with pytest.raises(KeyError):
        with EpisodeReader(row) as reader:
            reader.read('q');handle=reader._file;reader.read('missing')
    assert not handle.id.valid and path.read_bytes()==before


@pytest.mark.parametrize('kind',['action','object','camera'])
def test_distinct_timestamp_mismatch_still_rejected(row,kind):
    if kind=='action':row['action_timestamp_key']='wrong_time'
    elif kind=='object':row['object_telemetry']['cube']['timestamp_path']='wrong_time'
    else:row['cameras']['side']['timestamp_key']='wrong_time'
    with EpisodeReader(row) as reader,pytest.raises(ValueError,match='timestamp|aligned'):
        reader.frame(4,'side')


def test_canonical_reads_detached_selection_and_rejects_escape(row,tmp_path):
    folder=tmp_path/'canonical';episode=folder/'episode_0';episode.mkdir(parents=True)
    array=np.arange(120,dtype=np.float32).reshape(20,6);np.save(episode/'q.npy',array)
    row['data_ref']=dict(path=str(folder),episode='episode_0')
    with EpisodeReader(row) as reader:
        values=reader.read('q.npy',slice(2,5));assert values.flags.owndata
        values[:]=99
        np.testing.assert_array_equal(reader.read('q.npy',slice(2,5)),array[2:5])
        with pytest.raises(ValueError,match='outside'):reader.read('../../outside.npy')


def test_video_ownership_eviction_cache_bounds_and_close(row,monkeypatch):
    from soarm101_lab.utils import async_video_writer as video
    instances=[]
    class Frames:
        def __init__(self):self.closed=False;self.decoded=0;instances.append(self)
        def __getitem__(self,index):
            assert not self.closed;self.decoded+=1
            return np.full((4,5,3),index,dtype=np.uint8)
        def close(self):self.closed=True
    monkeypatch.setattr(video,'external_camera',lambda *args:Frames())
    for spec in row['cameras'].values():spec['file']='placeholder.mp4'
    with EpisodeReader(row) as reader:
        one=reader.frame(0,'side');reader.frame(0,'side')
        assert instances[0].decoded==1 and one['rgb'].dtype==np.uint8
        for camera in ('wrist','overview'):reader.frame(0,camera)
        assert instances[0].closed and len(reader.video_readers)==2
        for i in range(1,15):reader.frame(i,'overview')
        assert len(reader.cache)<=8 and reader.cache_bytes<=16*2**20
    assert all(i.closed for i in instances) and not reader.cache and reader.cache_bytes==0


def test_common_reader_imports_no_stage_ui_or_simulator():
    code="""import sys
import soarm101_lab.data.episodes
assert not any(m.startswith(('soarm101_lab.workflow','PyQt6','torch','mujoco','isaaclab','omni','pxr')) for m in sys.modules)
"""
    subprocess.run([sys.executable,'-c',code],check=True)
