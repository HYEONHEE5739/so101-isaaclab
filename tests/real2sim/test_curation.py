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
        assert f['data/demo_0/actions'][0,0]==2
        assert f['data/demo_0'].attrs['curation_source_episode']=='demo_2'
        assert 'obs/camera' in f['data/demo_0']
    with pytest.raises(ValueError):export_subset(src,out,['demo_0'])


def test_ui_select_exclude_and_restore(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow.curation_ui import CurationDialog
    from soarm101_lab.workflow.artifacts import Registry
    app=QApplication.instance() or QApplication([])
    d=CurationDialog(Registry(tmp_path))
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
    from soarm101_lab.workflow.curation import analyze,load_episode,EpisodeReader
    from soarm101_lab.workflow.robot_preview import RobotPreview
    writer=motion_canonical(tmp_path,motion_workspace,ids);row=analyze(writer.root)[0];reader=EpisodeReader(row);loaded=load_episode(row)
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
    from soarm101_lab.workflow.tasks import catalog,configured
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
    from soarm101_lab.workflow.curation import analyze,EpisodeReader
    path=tmp_path/'motion.hdf5';hdf5_motion(path,motion_workspace,mode)
    with h5py.File(path,'a') as h:
        h['data/demo_0'].attrs['annotation']=json.dumps({'transport':6})
        if mode=='initial':
            for camera in ('side','wrist'):h.move('data/demo_0/obs/policy/'+camera+'_cam','data/demo_0/obs/camera/'+camera)
    row=analyze(path)[0];reader=EpisodeReader(row);one=reader.frame(4,'wrist')
    assert row['events']=={'grasp':3,'place':9,'transport':6} and row['physical_success'] is None
    assert one['rgb'][0,0,0]==14 and one['timestamp']==pytest.approx(.4)
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
    from soarm101_lab.workflow.curation import analyze,EpisodeReader
    path=tmp_path/'legacy.hdf5';hdf5_motion(path,motion_workspace,mode)
    with h5py.File(path,'a') as h:
        d=h['data'];args=json.loads(d.attrs['env_args'])
        for key in ('workspace','workspace_path','scene_configuration','task_definitions'):args.pop(key,None)
        args['grasp_object']='cube_red';d.attrs['env_args']=json.dumps(args)
        d['demo_0'].create_dataset('initial_state/articulation/robot/root_pose',data=[[.1,.2,.3,1,0,0,0]])
    row=analyze(path)[0];reader=EpisodeReader(row);frame=reader.frame(4)
    assert row['active_object_ids'] is None and row['active_object_count'] is None
    assert row['robot_urdf'] is None and row['recorded_object_ids']==['cube_red']
    if mode=='post':np.testing.assert_allclose(frame['object_poses']['cube_red'][:3,3],[-.092,-.2,-.29],atol=1e-7)
    else:
        assert frame['object_poses']['cube_red'] is None
        np.testing.assert_allclose(frame['initial_object_poses']['cube_red'][:3,3],[-.1,-.2,-.29],atol=1e-7)


def test_large_hdf5_lazy_rgb_and_small_cache(tmp_path,motion_workspace,monkeypatch):
    from soarm101_lab.workflow.curation import analyze,load_episode,EpisodeReader
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
    reader=EpisodeReader(rows[0])
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
    from soarm101_lab.workflow.artifacts import Registry
    app=QApplication.instance() or QApplication([])
    writer=motion_canonical(tmp_path,motion_workspace,motion_workspace['task']['dynamic']);rows=analyze(writer.root)
    d=CurationDialog(Registry(tmp_path/'registry'));d.error=lambda error:pytest.fail(str(error));d.loaded(str(writer.root),rows);d.show()
    wait_qt(lambda:d.reader is not None and d.frame_job is None and d.job is None)
    assert set(d.trajectory.row['object_paths'])==set(motion_workspace['task']['dynamic'])
    assert d.trajectory.robot is not None and not d.rgb.pixmap().isNull()
    assert 'grasp' in d.event_label.text() and 'Physical UNKNOWN' in d.metrics.text()
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
    d.camera.setCurrentIndex(d.camera.findData('wrist'));wait_qt(lambda:d.frame_job is None)
    d.rate.setCurrentIndex(3);d.timeline.setValue(0);wait_qt(lambda:d.frame_job is None)
    d.toggle_play();wait_qt(lambda:not d.play_timer.isActive() and d.frame_job is None)
    assert d.timeline.value()==d.timeline.maximum()
    d.filters['active_object_count'].setCurrentIndex(d.filters['active_object_count'].findData('3'))
    assert d.list.count()==1
    d.close()


def test_export_registry_convert_review_and_immutable_input(tmp_path,motion_workspace,monkeypatch):
    from soarm101_lab.workflow.artifacts import Registry,fingerprint
    from soarm101_lab.workflow import episodes
    from soarm101_lab.workflow.operations import Operations
    from soarm101_lab.real2sim.workspaces import package
    writer=motion_canonical(tmp_path,motion_workspace,['cube_red','cube_blue']);before=fingerprint(writer.root)
    curated=export_subset(writer.root,tmp_path/'curated',['episode_0000']);r=Registry(tmp_path/'registry')
    artifact=r.register('annotated',curated)
    plan=Operations(r).plan('convert',artifact,str(motion_workspace['root']),{'backend':'mujoco','repo_id':'local/review'},run_dir=tmp_path/'run')
    assert plan['options']['repo_id']=='local/review'
    def converter(argv,**kwargs):
        source=argv[argv.index('--input_file')+1];root=Path(argv[argv.index('--root')+1]);root.mkdir()
        with h5py.File(source) as h:
            g=h['data/demo_0'];assert json.loads(g.attrs['annotation'])['grasp']==1
            assert g.attrs['source_group']==writer.manifest['episodes'][0]['group_id']
            assert json.loads(g.attrs['scene_configuration'])['active_object_ids']==['cube_red','cube_blue']
        (root/'workflow_provenance.json').write_text('{}')
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
    from soarm101_lab.workflow.curation import analyze,EpisodeReader
    writer=motion_canonical(tmp_path,motion_workspace,['cube_red']);folder=writer.root/'episode_0000'
    if kind=='object':
        np.save(folder/'wrong_time.npy',np.load(folder/'timestamp.npy')+.4)
        manifest=json.loads((writer.root/'manifest.json').read_text());manifest['episodes'][0]['object_telemetry']['cube_red']['timestamp_file']='wrong_time.npy';(writer.root/'manifest.json').write_text(json.dumps(manifest))
    elif kind=='action':np.save(folder/'action_timestamp.npy',np.load(folder/'timestamp.npy')+.1)
    else:
        np.save(folder/'side_timestamp.npy',np.load(folder/'timestamp.npy')+.001)
        reader=EpisodeReader(analyze(writer.root)[0])
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
