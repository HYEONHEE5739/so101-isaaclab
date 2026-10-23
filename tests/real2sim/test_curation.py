import json
from pathlib import Path
import numpy as np
import h5py
import pytest
from soarm101_lab.workflow.curation import select_distribution, export_subset


def test_exact_deterministic_stratification():
    rows=[dict(name=f'demo_{i}',task_id='a' if i<8 else 'b',xy=[i%2,i%3],flags=[],jitter=i) for i in range(12)]
    chosen=select_distribution(rows,6)
    assert len(set(chosen))==6
    assert chosen==select_distribution(rows,6)
    assert set(select_distribution(rows,12))=={r['name'] for r in rows}
    with pytest.raises(ValueError):select_distribution(rows,13)
    with pytest.raises(ValueError):select_distribution([],1)


def test_export_preserves_source_and_full_episode(tmp_path):
    src=tmp_path/'input.hdf5';out=tmp_path/'curated.hdf5'
    with h5py.File(src,'w') as f:
        d=f.create_group('data');d.attrs['env_args']='{"task":"test"}';d.attrs['total']=6
        for i in range(3):
            g=d.create_group(f'demo_{i}');g.attrs['success']=True
            g.create_dataset('actions',data=np.ones((2,6))*i)
            g.create_dataset('obs/camera',data=np.zeros((2,4,4,3),dtype=np.uint8))
    from soarm101_lab.so101_dataset_contract import write_contract,read_contract
    write_contract(src,{'schema':'so101.coordinate-contract/1','action_space':'test'})
    before=src.read_bytes()
    export_subset(src,out,['demo_2','demo_0'])
    assert src.read_bytes()==before
    assert read_contract(src)==read_contract(out)
    with h5py.File(out) as f:
        assert len(f['data'])==2 and f['data'].attrs['total']==4
        assert f['data/demo_0/actions'][0,0]==0
        assert f['data/demo_0'].attrs['curation_source_episode']=='demo_0'
        assert 'obs/camera' in f['data/demo_0']
    with pytest.raises(ValueError):export_subset(src,out,['demo_0'])


def test_ui_select_exclude_and_restore(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow.curation_ui import CurationDialog
    app=QApplication.instance() or QApplication([])
    d=CurationDialog(tmp_path)
    d.rows=[dict(name='demo_0',task_id='a',xy=[0,0],flags=['長'],duration=1,stop_fraction=.2,jitter=1,gripper_jitter=1,
                 ee=[[0,0,0],[.01,0,.01]],actions=[[0]*6,[1]*6])]
    d.scatter.rows=d.rows;d.select(0);d.toggle()
    assert d.excluded=={'demo_0'}
    d.restore_all();assert not d.excluded
    d.close()


def test_robot_urdf_fk_and_visuals():
    from pathlib import Path
    from soarm101_lab.workflow.robot_preview import RobotPreview
    root=Path(__file__).resolve().parents[2]
    robot=RobotPreview(root/'assets/SO101/urdf/so101_isaaclab.urdf')
    q=np.zeros(6);a=robot.transforms(q)
    q[0]=.5;b=robot.transforms(q)
    assert np.allclose(a['base'],b['base'])
    assert not np.allclose(a['tool0'],b['tool0'])
    tri,colors=robot.triangles(q)
    assert tri.shape[1:]==(3,3) and len(tri)==len(colors)
    assert np.isfinite(tri).all()


@pytest.fixture
def motion_workspace(tmp_path):
    """Synthetic Builder package with a rotated/translated robot base; never modifies a user scene."""
    import yaml
    from test_builder_workspace import make_export
    from soarm101_lab.real2sim.workspaces import builder,package
    from soarm101_lab.workflow.robot_preview import rotation
    root=make_export(tmp_path);scene=yaml.safe_load((root/'scene.yaml').read_text())
    t=np.eye(4);t[:3,:3]=rotation([0,0,1],.4);t[:3,3]=[.05,-.02,.03]
    scene['frames']['robot_base']['transform']['value']=t.tolist()
    (root/'scene.yaml').write_text(yaml.safe_dump(scene));m=yaml.safe_load((root/'scene_manifest.yaml').read_text());m['files']['scene.yaml']=package.sha(root/'scene.yaml');(root/'scene_manifest.yaml').write_text(yaml.safe_dump(m))
    return package.load(builder.import_scene(root,registry=tmp_path/'workspaces'))


def motion_canonical(tmp_path,w,ids):
    from test_active_objects import record_fixture
    writer=record_fixture(tmp_path/'motion',w,ids)
    folder=writer.root/'episode_0000';poses=np.load(folder/'object_pose.npy');poses[:,0,3]=poses[0,0,3]+[0,.01,.02,.03];poses[:,2,3]=poses[0,2,3]+[0,.02,.03,.01];np.save(folder/'object_pose.npy',poses)
    (folder/'annotation.json').write_text(json.dumps({'schema':'so101.mujoco-subtasks/1','frame_count':4,'grasp':1,'place':3}))
    return writer


@pytest.mark.parametrize('ids',[['cube_red'],['cube_red','cube_blue','cube_green']])
def test_robot_object_coordinate_alignment_and_canonical_telemetry(tmp_path,motion_workspace,ids):
    from soarm101_lab.workflow.curation import analyze,load_episode
    from soarm101_lab.data.episodes import EpisodeReader
    from soarm101_lab.workflow.robot_preview import RobotPreview
    writer=motion_canonical(tmp_path,motion_workspace,ids);row=analyze(writer.root)[0]
    with EpisodeReader(row) as reader:
        loaded=load_episode(row)
        assert set(loaded['object_paths'])==set(ids) and all(p.shape==(4,4,4) for p in loaded['object_paths'].values())
        assert row['display_frame']=='robot_base' and row['pick_object_id']=='cube_red'
        assert row['active_object_count']==len(ids) and row['source_group']==writer.manifest['episodes'][0]['group_id']
        assert row['physical_success'] is None and row['recorded_success'] is True
        assert row['events']=={'grasp':1,'place':3} and row['lift_height_m']==pytest.approx(.03,abs=1e-6)
        assert row['movement_before_grasp_m']>0 and row['movement_after_grasp_m']>0
        raw=np.load(writer.root/'episode_0000/object_pose.npy');w=motion_workspace['profile']
        base_workspace=np.linalg.inv(w['robot_base']['T_world'])@w['workspace']['T_world']
        np.testing.assert_allclose(loaded['object_paths']['cube_red'],base_workspace@raw)
        model=RobotPreview(row['robot_urdf'])
        for i in range(4):
            frame=reader.frame(i,'side')
            np.testing.assert_allclose(frame['ee'],model.transforms(frame['robot_q'])['tool0'][:3,3],atol=1e-7)
            assert frame['rgb_timestamp']==pytest.approx(frame['timestamp'])
        assert set(row['object_approach'])==set(ids)
        assert set(writer.manifest['episodes'][0]['object_telemetry'])==set(ids)
        assert (writer.root/'episode_0000/object_poses.npy').exists()==(len(ids)>1)


def hdf5_motion(path,w,mode='pre',frames=12,large_rgb=False):
    from soarm101_lab.real2sim.workspaces.package import metadata
    from soarm101_lab.real2sim.workspaces.demo import coordinate_contract
    from soarm101_lab.so101_dataset_contract import write_contract
    from soarm101_lab.real2sim.workspaces.tasks import catalog, configured
    definition=next(t for t in catalog(w) if t['pick_object_id']=='cube_red');cfg=configured(w,definition,backend='isaac')['scene_configuration']
    args={'workspace':metadata(w),'workspace_path':str(w['root']),'control_dt':.1,'task_definitions':[definition],'scene_configuration':cfg,
          'ee_frame':'robot_base','root_quaternion_order':'wxyz'}
    if mode=='pre':
        args.update(observation_timestamp_path='obs/timestamp',object_telemetry={n:{'path':'obs/object_poses/'+n,'pose_format':'xyz_wxyz','frame':'world','timestamp_path':'obs/timestamp'} for n in w['task']['dynamic']})
    with h5py.File(path,'w') as h:
        d=h.create_group('data');d.attrs['env_args']=json.dumps(args);g=d.create_group('demo_0');g.attrs['success']=True;g.attrs['success_source']='operator_saved_unverified'
        g.create_dataset('actions',shape=(frames,7),dtype='f4');g.create_dataset('joint_targets',shape=(frames,6),dtype='f4');g.create_dataset('obs/policy/joint_pos',shape=(frames,6),dtype='f4');g.create_dataset('obs/policy/ee_state',shape=(frames,7),dtype='f4')
        for camera in ('side','wrist'):
            if large_rgb:g.create_dataset('obs/policy/'+camera+'_cam',shape=(frames,480,640,3),dtype='u1',chunks=(1,480,640,3),fillvalue=0)
            else:g.create_dataset('obs/policy/'+camera+'_cam',data=np.stack([np.full((32,32,3),i+(10 if camera=='wrist' else 0),np.uint8) for i in range(frames)]))
        if mode=='pre':g.create_dataset('obs/timestamp',data=np.arange(frames)[:,None]*.1)
        for j,n in enumerate(w['task']['dynamic']):
            pose=np.zeros((frames+1,7));pose[:,0]=np.arange(frames+1)*.002;pose[:,1]=j*.03;pose[:,2]=.01;pose[:,3]=1
            g.create_dataset('initial_state/rigid_object/'+n+'/root_pose',data=pose[:1])
            if mode=='pre':g.create_dataset('obs/object_poses/'+n,data=pose[:-1].astype('f4'))
            elif mode=='post':g.create_dataset('states/rigid_object/'+n+'/root_pose',data=pose[1:])
        if not large_rgb:
            signal=np.arange(frames)>=3;g.create_dataset('obs/datagen_info/subtask_term_signals/grasp',data=signal)
            g.create_dataset('obs/events/place',data=np.arange(frames)>=9)
    write_contract(path,coordinate_contract(w));return args


@pytest.mark.parametrize('mode',['pre','post','initial'])
def test_hdf5_pre_post_and_initial_only(tmp_path,motion_workspace,mode):
    from soarm101_lab.workflow.curation import analyze
    from soarm101_lab.data.episodes import EpisodeReader
    path=tmp_path/'motion.hdf5';hdf5_motion(path,motion_workspace,mode)
    with h5py.File(path,'a') as h:
        h['data/demo_0'].attrs['annotation']=json.dumps({'transport':6})
        if mode=='initial':
            for camera in ('side','wrist'):h.move('data/demo_0/obs/policy/'+camera+'_cam','data/demo_0/obs/camera/'+camera)
    row=analyze(path)[0]
    with EpisodeReader(row) as reader:
        one=reader.frame(4,'wrist')
        assert row['events']=={'grasp':3,'place':9,'transport':6} and row['physical_success'] is None
        assert one['rgb'][0,0,0]==14 and one['timestamp']==pytest.approx(.4)
        pair=reader.frame(4,('side','wrist'))
        assert pair['rgb_frames']['side'][0,0,0]==4 and pair['rgb_frames']['wrist'][0,0,0]==14
        assert pair['rgb_timestamps']=={'side':pytest.approx(.4),'wrist':pytest.approx(.4)}
        if mode!='initial':
            actual=one['object_poses']['cube_red'];base_world=np.linalg.inv(motion_workspace['profile']['robot_base']['T_world'])
            np.testing.assert_allclose(actual[:3,3],(base_world@np.array([.008,0,.01,1]))[:3],atol=1e-7)
            assert row['target_movement_m']==pytest.approx(.022,abs=1e-7)
        else:
            assert one['object_poses']['cube_red'] is None and one['initial_object_poses']['cube_red'] is not None
            assert row['target_movement_m'] is None and row['xy'] is not None


def test_old_canonical_distractor_trajectory_unavailable(tmp_path,motion_workspace):
    from soarm101_lab.workflow.curation import analyze,load_episode
    writer=motion_canonical(tmp_path,motion_workspace,motion_workspace['task']['dynamic'])
    m=json.loads((writer.root/'manifest.json').read_text());m['episodes'][0].pop('object_telemetry');(writer.root/'manifest.json').write_text(json.dumps(m))
    row=analyze(writer.root)[0];loaded=load_episode(row)
    assert row['active_object_count']==3 and loaded['object_paths']['cube_red'] is not None
    assert loaded['object_paths']['cube_blue'] is None and loaded['object_paths']['cube_green'] is None


@pytest.mark.parametrize('mode',['post','initial'])
def test_legacy_saved_root_registers_object_without_guessing_active_set(tmp_path,motion_workspace,mode):
    from soarm101_lab.workflow.curation import analyze
    from soarm101_lab.data.episodes import EpisodeReader
    path=tmp_path/'legacy.hdf5';hdf5_motion(path,motion_workspace,mode)
    with h5py.File(path,'a') as h:
        d=h['data'];args=json.loads(d.attrs['env_args'])
        for key in ('workspace','workspace_path','scene_configuration','task_definitions'):args.pop(key,None)
        args['grasp_object']='cube_red';d.attrs['env_args']=json.dumps(args)
        d['demo_0'].create_dataset('initial_state/articulation/robot/root_pose',data=[[.1,.2,.3,1,0,0,0]])
    row=analyze(path)[0]
    with EpisodeReader(row) as reader:
        frame=reader.frame(4)
        assert row['active_object_ids'] is None and row['active_object_count'] is None
        assert row['robot_urdf'] is None and row['recorded_object_ids']==['cube_red']
        if mode=='post':np.testing.assert_allclose(frame['object_poses']['cube_red'][:3,3],[-.092,-.2,-.29],atol=1e-7)
        else:
            assert frame['object_poses']['cube_red'] is None
            np.testing.assert_allclose(frame['initial_object_poses']['cube_red'][:3,3],[-.1,-.2,-.29],atol=1e-7)


def test_large_hdf5_lazy_rgb_and_small_cache(tmp_path,motion_workspace,monkeypatch):
    from soarm101_lab.workflow.curation import analyze,load_episode
    from soarm101_lab.data.episodes import EpisodeReader
    path=tmp_path/'large.hdf5';hdf5_motion(path,motion_workspace,'initial',frames=10000,large_rgb=True)
    reads=[];original=h5py.Dataset.__getitem__
    def audit(dataset,selection):
        if dataset.name.endswith('_cam'):
            assert isinstance(selection,(int,np.integer)),'Never read a complete RGB series'
            reads.append(selection)
        return original(dataset,selection)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',audit)
    rows=analyze(path);loaded=load_episode(rows[0],max_points=75)
    assert not reads and len(loaded['ee'])==75 and 'ee' not in rows[0]
    with EpisodeReader(rows[0]) as reader:
        for i in range(15):reader.frame(i,'side')
        assert len(reader.cache)<=8 and reader.cache_bytes<=16*2**20
        count=len(reads);reader.frame(14,'side');assert len(reads)==count
        assert path.stat().st_size<2**20 # logical RGB size ~18GiB; no materialized frame series


def test_composition_and_source_group_stratification():
    rows=[dict(name=f'{composition}_{i}',task_id='pick',active_object_ids=['one'] if composition=='single' else ['one','two','three'],
          source_group=f'g{i%3}',xy=[0,0],flags=[],jitter=i) for composition in ('single','triple') for i in range(9)]
    selected=select_distribution(rows,6);subset=[r for r in rows if r['name'] in selected]
    assert sum(len(r['active_object_ids'])==1 for r in subset)==3
    assert {r['source_group'] for r in subset}=={'g0','g1','g2'}
    assert selected==select_distribution(rows,6)


def wait_qt(predicate,timeout=10000):
    from PyQt6.QtTest import QTest
    import time
    deadline=time.monotonic()+timeout/1000
    while not predicate() and time.monotonic()<deadline:QTest.qWait(10)
    assert predicate(),'Qt worker did not complete'


def test_synchronized_ui_rgb_events_seek_and_responsive_worker(tmp_path,motion_workspace,monkeypatch):
    import time
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    from soarm101_lab.workflow.curation import analyze
    from soarm101_lab.workflow.curation_ui import CurationDialog
    app=QApplication.instance() or QApplication([])
    writer=motion_canonical(tmp_path,motion_workspace,motion_workspace['task']['dynamic']);rows=analyze(writer.root)
    d=CurationDialog(tmp_path/'workflow');d.error=lambda error:pytest.fail(str(error));d.loaded(str(writer.root),rows);d.show()
    wait_qt(lambda:d.reader is not None and d.frame_job is None and d.job is None)
    assert set(d.trajectory.row['object_paths'])==set(motion_workspace['task']['dynamic'])
    assert d.trajectory.robot is not None and not d.rgb.pixmap().isNull()
    assert set(d.event_buttons)=={'grasp','place'} and '물리 판정: 판정 근거 없음' in d.metrics.text()
    assert all(not label.pixmap().isNull() for label in d.rgb_views.values())
    assert not d.filter_panel.isVisible() and not d.analysis_panel.isVisible() and not d.advanced_panel.isVisible()
    d.preview_tabs.setCurrentIndex(1);app.processEvents()
    assert d.trajectory.width()==d.trajectory.height() and d.trajectory.width()>=320
    original=d.reader.frame;threads=[]
    def slow(index,camera):
        import threading
        threads.append(threading.get_ident());time.sleep(.07);return original(index,camera)
    monkeypatch.setattr(d.reader,'frame',slow)
    ticks=[];timer=QTimer();timer.setInterval(5);timer.timeout.connect(lambda:ticks.append(1));timer.start()
    d.timeline.setValue(1);d.timeline.setValue(3);d.timeline.setValue(2)
    wait_qt(lambda:d.frame_job is None and d.pending_frame is None)
    timer.stop()
    assert len(ticks)>5 and d.trajectory.frame==2 and d.actions.frame==2
    import threading
    assert all(i!=threading.get_ident() for i in threads)
    assert d.trajectory.payload['index']==2 and d.timeline.value()==2
    assert set(d.trajectory.payload['rgb_frames'])=={'side','wrist'}
    assert d.trajectory.payload['rgb_timestamps']['side']==d.trajectory.payload['rgb_timestamps']['wrist']
    d.rate.setCurrentIndex(3);d.timeline.setValue(0);wait_qt(lambda:d.frame_job is None)
    d.toggle_play();wait_qt(lambda:not d.play_timer.isActive() and d.frame_job is None)
    assert d.timeline.value()==d.timeline.maximum()
    d.filters['active_object_count'].setCurrentIndex(d.filters['active_object_count'].findData('3'))
    assert d.list.count()==1
    d.close()



def test_ui_reader_release_waits_for_inflight_frame(tmp_path,motion_workspace,monkeypatch):
    import threading
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow.curation import analyze
    from soarm101_lab.workflow.curation_ui import CurationDialog
    app=QApplication.instance() or QApplication([])
    writer=motion_canonical(tmp_path,motion_workspace,['cube_red'])
    rows=analyze(writer.root);d=CurationDialog(tmp_path/'workflow')
    d.error=lambda error:pytest.fail(str(error));d.loaded(str(writer.root),rows);d.show()
    wait_qt(lambda:d.reader is not None and d.frame_job is None and d.job is None)
    old=d.reader;entered=threading.Event();release=threading.Event();original=old.frame
    def slow(index,camera):
        entered.set()
        if not release.wait(5):raise RuntimeError('Frame release timed out')
        assert not old._closed
        return original(index,camera)
    monkeypatch.setattr(old,'frame',slow)
    try:
        d.show_frame(1);wait_qt(entered.is_set)
        d.select(0)
        assert not old._closed
        release.set()
        wait_qt(lambda:d.reader is not None and d.reader is not old and d.frame_job is None and d.job is None)
        assert old._closed and not old.video_readers and not old.cache
        assert d.trajectory.frame==0  # stale frame 1 was not installed
        current=d.reader;assert current.video_readers
        videos=list(current.video_readers.values())
        d.close();app.processEvents()
        assert current._closed and d.reader is None
        assert all(getattr(v,'_container',None) is None for v in videos)
    finally:
        release.set();wait_qt(lambda:d.frame_job is None and d.job is None);d.close()

def test_export_registry_convert_review_and_immutable_input(tmp_path,motion_workspace,monkeypatch):
    from soarm101_lab.utils.artifact_identity import fingerprint
    from soarm101_lab.workflow import episodes
    from soarm101_lab.workflow.operations import Operations
    from soarm101_lab.real2sim.workspaces import package
    writer=motion_canonical(tmp_path,motion_workspace,['cube_red','cube_blue']);before=fingerprint(writer.root)
    curated=export_subset(writer.root,tmp_path/'curated',['episode_0000'])
    plan=Operations(root=tmp_path/'workflow').plan('convert',str(curated),str(motion_workspace['root']),{'backend':'mujoco','repo_id':'local/review','storage_name':'review'},run_dir=tmp_path/'run')
    assert plan['options']['repo_id']=='local/review'
    def converter(argv,**kwargs):
        source=argv[argv.index('--input_file')+1];root=Path(argv[argv.index('--root')+1]);root.mkdir()
        # Legacy import and numeric mapping now belong to the single converter.
        assert Path(source)==curated.resolve()
        manifest=json.loads((Path(source)/'manifest.json').read_text())
        entry=manifest['episodes'][0];folder=Path(source)/entry['path']
        assert json.loads((folder/'annotation.json').read_text())['grasp']==1
        assert entry['group_id']==writer.manifest['episodes'][0]['group_id']
        assert entry['scene_configuration']['active_object_ids']==['cube_red','cube_blue']
        (root/'meta').mkdir();(root/'meta/info.json').write_text(json.dumps({'total_episodes':1}))
        (root/'workflow_provenance.json').write_text(json.dumps({'canonical_episodes':[entry]}))
    import subprocess
    monkeypatch.setattr(subprocess,'run',converter)
    output=tmp_path/'convert';output.mkdir();result=episodes.convert_dataset(curated,output,{'workspace':str(motion_workspace['root']),'repo_id':'local/review'})
    manifest=json.loads((Path(result['dataset'])/'workflow_provenance.json').read_text())
    assert not (output/'conversion_input.hdf5').exists()
    assert not list(output.rglob('*.npy'))
    assert manifest['canonical_episodes'][0]['object_telemetry']==writer.manifest['episodes'][0]['object_telemetry']
    assert fingerprint(writer.root)==before
    assert package.load(motion_workspace['root'])['manifest']==motion_workspace['manifest']


@pytest.mark.parametrize('kind',['object','action','rgb_future'])
def test_timestamp_mismatch_rejected(tmp_path,motion_workspace,kind):
    from soarm101_lab.workflow.curation import analyze
    from soarm101_lab.data.episodes import EpisodeReader
    writer=motion_canonical(tmp_path,motion_workspace,['cube_red']);folder=writer.root/'episode_0000'
    if kind=='object':
        np.save(folder/'wrong_time.npy',np.load(folder/'timestamp.npy')+.4)
        manifest=json.loads((writer.root/'manifest.json').read_text());manifest['episodes'][0]['object_telemetry']['cube_red']['timestamp_file']='wrong_time.npy';(writer.root/'manifest.json').write_text(json.dumps(manifest))
    elif kind=='action':np.save(folder/'action_timestamp.npy',np.load(folder/'timestamp.npy')+.1)
    else:
        np.save(folder/'side_timestamp.npy',np.load(folder/'timestamp.npy')+.001)
        with EpisodeReader(analyze(writer.root)[0]) as reader:
            with pytest.raises(ValueError,match='not aligned'):reader.frame(0,'side')
            return
    with pytest.raises(ValueError,match='timestamp mismatch'):analyze(writer.root)


def test_new_builder_object_geometry_is_metadata_driven(tmp_path):
    from test_active_objects import export_with_new_object,record_fixture
    from soarm101_lab.real2sim.workspaces import builder,package
    from soarm101_lab.workflow.curation import analyze,load_episode
    from soarm101_lab.workflow.robot_preview import object_triangles
    w=package.load(builder.import_scene(export_with_new_object(tmp_path),registry=tmp_path/'registry'))
    writer=record_fixture(tmp_path/'source',w,['cube_red','new_manipulable']);row=load_episode(analyze(writer.root)[0]);spec=row['object_specs']['new_manipulable']
    assert spec['dimensions_m']==[.03,.02,.02] and row['object_paths']['new_manipulable'] is not None
    tri=object_triangles(spec,np.eye(4));np.testing.assert_allclose(np.ptp(tri.reshape(-1,3),axis=0),[.03,.02,.02])


@pytest.mark.parametrize('storage',['canonical','hdf5'])
def test_distribution_boundary_matches_reset_and_robot_view(tmp_path,motion_workspace,storage):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow.curation import analyze
    from soarm101_lab.workflow.curation_ui import Plot
    from soarm101_lab.real2sim.workspaces.reset import WorkspaceRandomizer
    app=QApplication.instance() or QApplication([])
    if storage=='canonical':
        path=motion_canonical(tmp_path,motion_workspace,['cube_red']).root
    else:
        path=tmp_path/'motion.hdf5';hdf5_motion(path,motion_workspace)
    row=analyze(path)[0];region=row['placement_region'];profile=motion_workspace['profile'];reset=row['scene_configuration']['reset']
    support=profile['objects'][reset['support_object']];target=profile['objects'][row['pick_object_id']]
    center=np.array(support['T_workspace'])[:2,3]
    limit=np.array(support['dimensions_m'])[:2]/2-np.array(target['dimensions_m'])[:2]/2-reset['margin_m']
    np.testing.assert_allclose(np.min(region['allowed'],axis=0),center-limit)
    np.testing.assert_allclose(np.max(region['allowed'],axis=0),center+limit)
    sampler=WorkspaceRandomizer(motion_workspace)
    for _ in range(30):
        point=np.array(sampler.sample()['cube_positions'][row['pick_object_id']])[:2]
        assert np.all(point>=center-limit) and np.all(point<=center+limit)
    plot=Plot('scatter');plot.resize(500,350);plot.rows=[row];plot.context_rows=[row]
    robot,work=plot.project([region['robot_xy'],center])
    assert robot.y()>work.y()
    edges=np.diff(np.array([[p.x(),p.y()] for p in plot.project(region['support']+[region['support'][0]])]),axis=0)
    assert np.all(np.isclose(edges,0).any(axis=1)) # Horizontal/vertical edges, including a rotated robot base.
    assert 6<robot.x()<plot.width()-6 and 6<robot.y()<plot.height()-6
    old=plot.project([row['xy']])[0];plot.rows=[]
    new=plot.project([row['xy']])[0]
    assert (old.x(),old.y())==pytest.approx((new.x(),new.y()))
    assert plot.project([])==[]
    picture=plot.grab().toImage()
    assert picture.pixelColor(round(robot.x()),round(robot.y())).name()=='#a367db'
    plot.close()




def test_unified_picker_accepts_file_and_dataset_folder(tmp_path):
    from PyQt6.QtWidgets import QApplication,QDialog
    from soarm101_lab.workflow.curation_ui import DatasetPicker
    app=QApplication.instance() or QApplication([])
    folder=tmp_path/'canonical';folder.mkdir()
    for name in ('manifest.json','project_provenance.json'):(folder/name).write_text('{}')
    import h5py
    file=tmp_path/'source.hdf5'
    with h5py.File(file,'x') as h:h.create_group('data')
    for path in (folder,file):
        picker=DatasetPicker(None,str(tmp_path));picker.selectFile(str(path));picker.accept()
        assert picker.result()==QDialog.DialogCode.Accepted and picker.selectedFiles()==[str(path)]
        picker.close()




@pytest.mark.parametrize('storage',['canonical','hdf5'])
def test_annotation_markers_jump_to_exact_observation_and_clear(tmp_path,motion_workspace,storage):
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt,QPointF
    from PyQt6.QtTest import QTest
    from soarm101_lab.workflow.curation import analyze
    from soarm101_lab.workflow.curation_ui import CurationDialog
    from soarm101_lab.utils.artifact_identity import fingerprint
    app=QApplication.instance() or QApplication([])
    if storage=='canonical':
        path=motion_canonical(tmp_path,motion_workspace,['cube_red']).root
        times=np.array([12.,12.1,12.4,12.9]);events=dict(approach=0,grasp=1,transport=2,place=3)
        folder=path/'episode_0000'
        for name in ('timestamp','action_timestamp','side_timestamp','wrist_timestamp'):
            if (folder/(name+'.npy')).exists():np.save(folder/(name+'.npy'),times)
        (folder/'annotation.json').write_text(json.dumps({'schema':'so101.subtask-boundaries/1','frame_count':4,'boundaries':events}))
    else:
        path=tmp_path/'annotated.hdf5';hdf5_motion(path,motion_workspace)
        times=5+np.cumsum(np.linspace(.1,.2,12));events=dict(grasp=3,transport=6,place=9)
        with h5py.File(path,'a') as h:
            h['data/demo_0/obs/timestamp'][:,0]=times
            h['data/demo_0'].attrs['annotation']=json.dumps({'transport':6})
    before=fingerprint(path);rows=analyze(path)
    d=CurationDialog(tmp_path/'workflow');d.error=lambda error:pytest.fail(str(error));d.show();d.loaded(str(path),rows)
    wait_qt(lambda:d.job is None and d.frame_job is None and d.reader is not None)
    assert not d.analysis_panel.isVisible() and set(d.event_buttons)==set(events)
    assert [m['time'] for m in d.timeline.markers]==pytest.approx([times[i]-times[0] for i in events.values()])
    for name,index in events.items():
        button=d.event_buttons[name]
        assert f'{times[index]-times[0]:.2f}s' in button.text() and f'{index+1}F' in button.text()
        d.toggle_play();assert d.play_timer.isActive()
        button.click()
        wait_qt(lambda:d.frame_job is None and d.pending_frame is None)
        assert not d.play_timer.isActive()
        assert d.trajectory.frame==d.actions.frame==d.timeline.value()==index
        assert d.trajectory.payload['timestamp']==pytest.approx(times[index])
        assert set(d.trajectory.payload['rgb_frames'])=={'side','wrist'}
        assert button.isChecked() and '시점' in d.event_label.text()
    for width in (1100,1500):
        d.resize(width,920);app.processEvents()
        for name,index in events.items():
            point=d.timeline.marker_position(index)-QPointF(0,11)
            QTest.mouseClick(d.timeline,Qt.MouseButton.LeftButton,pos=point.toPoint())
            wait_qt(lambda:d.frame_job is None and d.pending_frame is None)
            assert d.trajectory.frame==d.actions.frame==index
    # Loading another episode must remove the previous episode's annotations.
    plain={**rows[0],'events':{}}
    d.rows=[plain];d.select(0);assert not d.event_buttons and not d.timeline.markers
    wait_qt(lambda:d.job is None and d.frame_job is None and d.reader is not None)
    assert not d.event_buttons and not d.timeline.markers and '주석 없음' in d.event_label.text()
    assert fingerprint(path)==before
    d.close()


@pytest.mark.parametrize('inverted',[False,True])
def test_annotation_marker_uses_native_slider_geometry(inverted):
    from PyQt6.QtWidgets import QApplication,QStyle,QStyleOptionSlider
    from soarm101_lab.workflow.curation_ui import EventTimeline
    app=QApplication.instance() or QApplication([])
    slider=EventTimeline();slider.resize(600,40);slider.setRange(0,100);slider.setInvertedAppearance(inverted)
    slider.set_events({'approach':0,'grasp':25,'place':100},dt=.2)
    for marker in slider.markers:
        slider.setValue(marker['frame']);option=QStyleOptionSlider();slider.initStyleOption(option)
        handle=slider.style().subControlRect(QStyle.ComplexControl.CC_Slider,option,QStyle.SubControl.SC_SliderHandle,slider)
        assert abs(slider.marker_position(marker['frame']).x()-handle.center().x())<=1
    assert [m['time'] for m in slider.markers]==[0,5,20]
    slider.close()


def test_review_reasons_include_original_values_and_comparison():
    from soarm101_lab.workflow.curation import flag_outliers
    from soarm101_lab.workflow.curation_ui import CurationDialog
    rows=[dict(name=str(i),task_id='pick',active_object_ids=['cube'],duration=2.,stop_fraction=.1,command_stop_fraction=.1,jitter=1.,flags=[]) for i in range(12)]
    rows[-1].update(duration=20.,stop_fraction=.9,command_stop_fraction=.8,jitter=15.)
    rows[-2]['duration']=.01
    rows.append(dict(rows[-1],name='other',task_id='other',flags=[]))
    flag_outliers(rows)
    assert len(rows[-2]['flag_details'])==4 and not rows[-1]['flags']
    assert rows[-2]['flag_details'][1]==dict(label='EE 정지 비율 높음',metric='stop_fraction',value=.9,median=.1,
                                          threshold=pytest.approx(.13),direction='high',group_size=12)
    reason=CurationDialog.review_text(rows[-2])
    assert '90.0%' in reason and '13.0%' in reason and '12개' in reason and '불량 확정' in reason
    assert '미만' in CurationDialog.review_text(rows[-3])
    assert not CurationDialog.review_text(rows[0])


def test_scatter_split_colors_keep_flag_outline_and_exclusion():
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow.curation_ui import Plot
    app=QApplication.instance() or QApplication([])
    plot=Plot('scatter');plot.resize(500,350)
    plot.rows=[dict(name='train',xy=[0,0],split='train',flags=[]),dict(name='val',xy=[1,0],split='val',flags=['check']),
               dict(name='excluded',xy=[0,1],split='val',flags=[])]
    plot.context_rows=plot.rows;plot.excluded={'excluded'};plot.selected=1
    picture=plot.grab().toImage();points=plot.project([r['xy'] for r in plot.rows])
    assert [picture.pixelColor(round(p.x()),round(p.y())).name() for p in points]==['#55d6b0','#69b9ff','#66717d']
    assert any(picture.pixelColor(round(points[1].x())+dx,round(points[1].y())+dy).name()=='#efb467'
               for dx in range(-6,7) for dy in range(-6,7))
    plot.close()


def test_legacy_copy_failure_rolls_back_manifest(tmp_path,motion_workspace,monkeypatch):
    import shutil
    from soarm101_lab.utils.artifact_identity import fingerprint
    writer=motion_canonical(tmp_path,motion_workspace,['cube_red']);before=fingerprint(writer.root)
    target=tmp_path/'failed';copy=shutil.copytree
    def fail(source,dest,*args,**kwargs):
        copy(source,dest,*args,**kwargs)
        raise OSError('copy interrupted')
    monkeypatch.setattr(shutil,'copytree',fail)
    with pytest.raises(OSError,match='copy interrupted'):export_subset(writer.root,target,['episode_0000'])
    assert not target.exists() and not list(tmp_path.glob('.failed*'))
    assert fingerprint(writer.root)==before


@pytest.mark.parametrize('nested',[False,True])
def test_curation_resolves_single_hdf5_folder_before_analysis(tmp_path,monkeypatch,nested):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import curation_ui,dataset
    app=QApplication.instance() or QApplication([])
    folder=tmp_path/'input';payload=folder/'origin' if nested else folder
    payload.mkdir(parents=True);path=payload/'dataset.hdf5';path.write_bytes(b'fixture')
    seen=[]
    monkeypatch.setattr(curation_ui,'analyze',lambda value:seen.append(value) or [])
    monkeypatch.setattr(dataset,'episode_input_kind',lambda value:'source')
    dialog=curation_ui.CurationDialog(tmp_path/'workflow')
    dialog.run_job=lambda fn,done:done(fn())
    dialog.loaded=lambda *args:seen.append(args)
    dialog.load_data(str(folder))
    assert seen==[str(path), (str(path),[],'source')]
    dialog.close()
