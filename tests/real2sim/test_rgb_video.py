"""Real H264 encode/decode fixtures, no hardware or task-success claims."""
import json
from pathlib import Path
import numpy as np
import pytest
from soarm101_lab.utils.async_video_writer import AsyncVideoWriter,VideoFrames,VideoEpisode,external_camera


def image(i,h=48,w=64):
    a=np.zeros((h,w,3),np.uint8);a[:]=[220,20+i,40]
    a[:,(i*2)%w:((i*2)%w)+4]=[10,230,30]
    return a


def test_rgb_exact_seek_count_channel_and_buffer_ownership(tmp_path):
    path=tmp_path/'side.mp4';writer=AsyncVideoWriter(path,fps=30,width=64,height=48,queue_size=2)
    for i in range(45):
        value=image(i);writer.write(value);value[:]=0
    writer.close();video=VideoFrames(path,45,30)
    assert video.shape==(45,48,64,3)
    for i in [0,22,44,3,40,1,2,43,0]:
        actual=video[i];expected=image(i)
        assert np.abs(actual.astype(float)-expected).mean()<6
        assert actual[0,-1,0]>150 and actual[0,-1,2]<90
    video.close()
    with pytest.raises(ValueError,match='count'):VideoFrames(path,44,30)
    with pytest.raises(ValueError,match='FPS'):VideoFrames(path,45,25)


def test_abort_and_encoder_failure_never_deadlock(tmp_path):
    video=VideoEpisode(tmp_path,30);video.append({'side':image(0),'wrist':image(1)})
    folder=video.folder;video.abort();assert not folder.exists()
    writer=AsyncVideoWriter(tmp_path/'bad.mp4',codec='nonexistent_encoder',width=64,height=48,queue_size=1)
    try:
        with pytest.raises(RuntimeError,match='encoding failed'):
            for _ in range(100):writer.write(image(0))
            writer.close()
    finally:writer.abort()
    assert not (tmp_path/'bad.mp4').exists()


def test_canonical_video_writer_numeric_loader_curation_and_legacy(tmp_path):
    from test_minivla_integration import make_dataset
    from soarm101_lab.simulation.mujoco_backend import PolicyEnvironment,configure_rendering
    from soarm101_lab.workflow.curation import analyze,EpisodeReader
    from soarm101_lab.learning.minivla.data import load_manifest,episode_arrays
    configure_rendering('none')
    root=Path(__file__).resolve().parents[2]
    profile=json.loads((root/'tests/real2sim/fixtures/scene_profile.json').read_text())
    env=PolicyEnvironment(tmp_path/'scene',{'profile':profile},renderer='none',control_hz=30)
    backend=env.backend
    try:
        backend.get_rgb=lambda camera=None:np.zeros((48,64,3),np.uint8)
        writer=make_dataset(tmp_path/'dataset',env)
        folder=writer.root/'episode_0000'
        assert (folder/'side.mp4').exists() and not (folder/'side.npy').exists()
        assert all('camera_images' not in r for r in writer.rows)
        m=load_manifest(writer.root/'manifest.json');a=episode_arrays(m['episodes'][0],m['cameras'],m['control_hz'])
        assert isinstance(a['side'],VideoFrames)
        # Legacy compatibility uses the same numeric arrays with old RGB NPY.
        for camera in m['cameras']:
            v=VideoFrames(folder/(camera+'.mp4'))
            np.save(folder/(camera+'.npy'),np.stack([v[i] for i in range(len(v))]));v.close()
            (folder/(camera+'.mp4')).unlink()
        old=episode_arrays(m['episodes'][0],m['cameras'],m['control_hz'])
        assert isinstance(old['side'],np.memmap)
    finally:env.close()


def test_external_hdf5_curation_subset_resume_and_identity(tmp_path):
    import h5py
    from test_source_partitions import source_fixture
    from soarm101_lab.workflow.curation import analyze,EpisodeReader,export_subset
    from soarm101_lab.workflow.data_resume import combine
    from soarm101_lab.workflow.artifacts import fingerprint
    from soarm101_lab.workflow.episodes import source_catalog
    source=tmp_path/'source.hdf5';source_fixture(source,4)
    with h5py.File(source,'a') as h:
        args=json.loads(h['data'].attrs['env_args']);args['observation_timestamp_path']='obs/timestamp';h['data'].attrs['env_args']=json.dumps(args)
        for i,g in enumerate(h['data'].values()):
            video=VideoEpisode(tmp_path,30)
            for t in range(4):video.append({'side_cam':image(t+i),'wrist_cam':image(t+i+1)})
            specs=video.finish(4)
            g.attrs['external_videos']=json.dumps({k:{**v,'file':str((video.folder/v['file']).relative_to(tmp_path))} for k,v in specs.items()})
            for c in ['side','wrist']:del g['obs/policy/'+c+'_cam']
            g.create_dataset('obs/timestamp',data=np.arange(4,dtype=float)[:,None]/30)
    before=fingerprint(source);records,_=source_catalog(source)
    rows=analyze(source);reader=EpisodeReader(rows[0]);assert reader.frame(3,'side')['rgb'].shape==(48,64,3)
    subset=export_subset(source,tmp_path/'curated.hdf5',['demo_3','demo_1'])
    after,_=source_catalog(subset)
    assert {r['source_id'] for r in after.values()}=={records[n]['source_id'] for n in ['demo_3','demo_1']}
    combine(source,subset)
    with h5py.File(subset,'r') as h:
        assert len(h['data'])==6
        for g in h['data'].values():
            for spec in json.loads(g.attrs['external_videos']).values():assert external_camera(subset,spec)[3].shape==(48,64,3)
    assert fingerprint(source)==before
