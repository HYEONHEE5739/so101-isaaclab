"""Passed snapshots, Common storage and resource lifetime; no device/engine run."""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import pytest
from soarm101_lab.data import episodes as episode_io
from soarm101_lab.data.recording import EpisodeBuffer
from test_common_mimic import episode


def test_explicit_frame_buffer_preserves_observed_command_and_clock(tmp_path,episode):
    source,attrs,_,args,contract=episode
    buffer=EpisodeBuffer(tmp_path,30,np.eye(4),['side','wrist'])
    initial=source['initial_state'];q=[];targets=[];goals=[]
    try:
        buffer.begin(initial)
        for i in range(5):
            measured=np.full(6,i*.01);measured[5]=i*.005
            command=np.full(6,i*.02);command[5]=0
            state=dict(joint_pos=measured.copy(),eef_pose=np.eye(4),object_poses={'cube':np.eye(4)})
            goal=np.eye(4);goal[0,3]=i*.02
            images={c:np.full((480,640,3),i*20,np.uint8) for c in ('side','wrist')}
            buffer.append(command,goal,state,images,grasp=i>=2,place=False)
            q.append(measured);targets.append(command.copy());goals.append(goal.copy())
            command[:]=99;goal[:]=99  # command/target buffers are reusable by simulation
        values,exported,videos=buffer.finish(attrs,success=True)
        np.testing.assert_array_equal(values['obs']['policy']['joint_pos'],q)
        np.testing.assert_array_equal(values['joint_targets'],targets)
        np.testing.assert_array_equal(values['obs']['datagen_info']['target_eef_pose']['tool0'],goals)
        np.testing.assert_array_equal(values['action_timestamp'],np.arange(5)/30)
        assert values['initial_state'] is initial
        assert values['obs']['events']['place'][:,0].tolist()==[0,0,0,0,1]
        assert not hasattr(buffer,'adapter')
        inodes={k:Path(v['file']).stat().st_ino for k,v in videos.items()}
        path=tmp_path/'result.hdf5'
        with episode_io.DatasetWriter(path,args,contract) as writer:
            writer.commit(values,exported,videos,video_staging=buffer.video)
        buffer.abort()
        from soarm101_lab.workflow.dataset import EpisodeDataset,validate_dataset
        reader=EpisodeDataset(path);assert validate_dataset(reader,require_annotations=True)['episodes']==1
        saved,_,stored=reader.episode('demo_0')
        np.testing.assert_array_equal(saved['obs']['policy']['joint_pos'],q)
        assert all((tmp_path/v['file']).stat().st_ino==inodes[k] for k,v in stored.items())
    finally:buffer.abort()


def test_buffer_abort_keeps_other_files_and_no_complete_dataset(tmp_path):
    other=tmp_path/'unrelated.txt';other.write_text('preserve')
    buffer=EpisodeBuffer(tmp_path,30,np.eye(4),['side','wrist']);buffer.begin({})
    buffer.append(np.zeros(6),np.eye(4),dict(joint_pos=np.zeros(6),eef_pose=np.eye(4),object_poses={}),
        {c:np.zeros((480,640,3),np.uint8) for c in ('side','wrist')},grasp=False,place=False)
    private=buffer.video.folder;buffer.abort();buffer.abort()
    assert not private.exists() and not buffer.rows and other.read_text()=='preserve'
    assert not list(tmp_path.glob('*.hdf5'))
    with pytest.raises(ValueError,match='No completed'):buffer.finish({},success=False)


@pytest.mark.parametrize('provided',[False,True])
def test_simulation_owns_precommand_capture_and_reuses_supplied_frames(provided):
    from soarm101_lab.workflow.adapters import SimulationAdapter
    calls=[];state={'joint_pos':np.zeros(6)};images={'side':np.zeros((4,5,3),np.uint8)}
    def observed():calls.append('state');return state
    def rgb():calls.append('images');return images
    def place():calls.append('place');return False
    def append(joints,target,actual,pixels,**events):
        assert actual is state and pixels is images
        assert events==dict(grasp=True,place=False,ik=None)
        calls.append('buffer')
    a=SimpleNamespace(state=observed,images=rgb,place_success=place,tracker=SimpleNamespace(completed=True),last_ik=None,
        buffer=SimpleNamespace(metrics=defaultdict(float),append=append))
    SimulationAdapter.record_frame(a,np.zeros(6),np.eye(4),state if provided else None,images if provided else None)
    calls.append('command')
    assert calls==(['place','buffer','command'] if provided else ['state','images','place','buffer','command'])
    assert a.last_images is images


def test_commit_borrows_unmodified_arrays_without_recursive_numeric_copy(tmp_path,episode,monkeypatch):
    values,attrs,videos,args,contract=episode
    info=values['obs']['datagen_info']
    for category in ('eef_pose','target_eef_pose','object_pose'):
        info[category]={k:v.astype(np.float32) for k,v in info[category].items()}
    original=values['joint_targets'].copy();checked=[];write=episode_io._write
    def inspect(group,payload):
        if 'obs' in payload:
            prepared=payload['obs']['datagen_info'];checked.append(True)
            assert prepared is not info
            assert np.shares_memory(prepared['eef_pose']['tool0'],info['eef_pose']['tool0'])
            assert prepared['subtask_term_signals'] is info['subtask_term_signals']
        return write(group,payload)
    monkeypatch.setattr(episode_io,'_write',inspect)
    with episode_io.DatasetWriter(tmp_path/'borrowed.hdf5',args,contract) as writer:writer.commit(values,attrs,videos)
    assert checked and values['obs']['datagen_info'] is info
    np.testing.assert_array_equal(values['joint_targets'],original)


def test_common_storage_imports_no_stage_or_engine():
    code="""import sys
import soarm101_lab.data.episodes
import soarm101_lab.data.recording
assert not any(m.startswith(('soarm101_lab.workflow','PyQt6','torch','mujoco','isaaclab','omni','pxr')) for m in sys.modules)
"""
    subprocess.run([sys.executable,'-c',code],check=True)
