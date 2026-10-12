"""Snapshot contract, atomic publication, shared runtime and independent Builder integration."""
import copy
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from soarm101_lab.real2sim.capture_dataset import SnapshotDataset, prepare_snapshot, digest_file
from soarm101_lab.real2sim.core import JOINTS, CONVENTION, Stability, to_sim
from soarm101_lab.so101_joint_mapping import DEFAULT_MAPPER

BUILDER = Path(os.environ.get('REAL2SIM_BUILDER_SOURCE',
    str(Path(__file__).resolve().parents[3] / 'real2sim-builder')))
sys.path.insert(0, str(BUILDER))


def fixture_input(tmp_path):
    urdf=tmp_path/'robot.urdf'
    text='<robot name="fixture"><link name="base"/>'
    for i,name in enumerate(JOINTS):
        parent='base' if i==0 else 'link'+str(i)
        text+=f'<link name="link{i+1}"><visual><geometry><box size=".02 .02 .04"/></geometry></visual></link><joint name="{name}" type="revolute"><parent link="{parent}"/><child link="link{i+1}"/><origin xyz="0 0 .05"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>'
    urdf.write_text(text+'</robot>')
    profile={'robot_base':{'T_world':np.eye(4).tolist()},'objects':{},'cameras':{}}
    for name,parent in [('side','world'),('wrist','link6')]:
        profile['cameras'][name]={'parent':parent,'resolution':[640,480],'K':[[500,0,320],[0,500,240],[0,0,1]],'distortion':[0]*5,'T_parent_camera':np.eye(4).tolist(),'provenance':{'K':'calibrated'}}
    image=np.tile((np.indices((480,640)).sum(0)%2*255).astype('uint8')[:,:,None],(1,1,3))
    return profile,urdf,{'real_side':image,'real_wrist':image.copy()}



def observation(profile,index=0):
    end=time.monotonic_ns()-10_000_000
    samples=[];stability=Stability()
    for i in range(8):
        t=end-(7-i)*100_000_000
        native=dict(zip(JOINTS,[index*15+i*.02,0.,0.,0.,0.,i*.1]))
        samples.append({'timestamp_ns':t,'native':native});stability.add(t,native)
    stamp=lambda t:{'monotonic_ns':t,'unix_ns':0}
    feedback={'start':stamp(end-1_000_000),'end':stamp(end)}
    command={**native,'shoulder_pan':native['shoulder_pan']+10}
    return {'joint_names':list(JOINTS),'joint_convention':CONVENTION,'sim_joint_mapping':DEFAULT_MAPPER.metadata(),
        'active_profile':copy.deepcopy(profile),'physical_follower_measured_joint_state':native,
        'sent_target':command,'measured_samples':samples,'real_stability':stability.assess(),
        'timestamps':{'follower':feedback,'send':feedback,'real_cameras':{
            'real_'+name:{'read_start':stamp(end-skew-2_000_000),'read_end':stamp(end-skew),'sequence':index+1}
            for name,skew in [('side',25_000_000),('wrist',65_000_000)]}}}


def dataset(tmp_path,poses=(0,1,2),states=None):
    profile,urdf,images=fixture_input(tmp_path)
    store=SnapshotDataset(tmp_path/'snapshots','Fixture',profile,urdf)
    for i,pose in enumerate(poses):store.save(observation(profile,pose),images,states[i] if states else 'static_001')
    return store,profile,images


def test_passive_precision_snapshot_saves_out_of_model_angle_with_warning(tmp_path):
    import math
    from soarm101_lab.real2sim.precision import FakeCalibrationDevice,fit_joint,review_joint,PrecisionMapper,native_from_raw
    profile,urdf,images=fixture_input(tmp_path)
    spec=FakeCalibrationDevice('follower').motor_spec('shoulder_pan')
    refs=[dict(joint='shoulder_pan',raw=r,angle_rad=math.radians(q),split='train',calibration_hash='fake')
          for r,q in [(1500,0),(2500,90)]]
    entry=review_joint(fit_joint(refs,spec,'shoulder_pan',[-1.91986,1.91986]))
    mapper=PrecisionMapper({'schema':'so101.precision-calibration/1','joints':{'shoulder_pan':entry}})
    state=observation(profile)
    state.update(sim_joint_mapping=mapper.metadata(),sent_target=None,precision_observation_only=True)
    native={n:0. for n in JOINTS};native['shoulder_pan']=native_from_raw(2800,spec)
    state['physical_follower_measured_joint_state']=native
    for row in state['measured_samples']:row['native']=copy.deepcopy(native)
    measured,_,warnings=prepare_snapshot(state,images,profile)
    assert measured['q_measured'][0]==pytest.approx(math.radians(117))
    assert measured['q_command'] is None and measured['joint_limit_warnings']['shoulder_pan']
    assert any('outside URDF' in warning for warning in warnings)
    store=SnapshotDataset(tmp_path/'passive','Fixture',profile,urdf)
    saved=store.save(state,images)
    shot=json.loads((Path(saved['path'])/'state.json').read_text())
    assert shot['q_measured'][0]==pytest.approx(math.radians(117))
    assert shot['precision_observation_only'] and shot['joint_limit_warnings']
    state['sent_target']=native
    with pytest.raises(ValueError,match='motor command'):prepare_snapshot(state,images,profile)


def test_one_snapshot_full_resolution_shared_assets_alignment(tmp_path):
    from app.core.capture import load_snapshots
    store,profile,images=dataset(tmp_path,poses=(0,))
    m,rows,events=load_snapshots(store.root)
    assert len(rows)==1 and not events
    common={p:digest_file(store.root/p) for p in m['files']}
    state=json.loads((store.root/m['snapshots'][0]['path']/'state.json').read_text())
    assert state['q_measured']!=state['q_command']
    assert state['q_native']==observation(profile)['physical_follower_measured_joint_state']
    assert rows[0]['images'][0]['resolution']==[640,480]
    assert rows[0]['images'][0]['timestamp_ns']!=rows[0]['images'][1]['timestamp_ns']
    assert rows[0]['images'][0]['q_measured']!=rows[0]['images'][1]['q_measured']
    store.save(observation(profile,1),images)
    assert store.count==2 and len(load_snapshots(store.root)[1])==2
    assert all(digest_file(store.root/p)==h for p,h in common.items())
    assert not list((store.root/'snapshots').rglob('robot.urdf'))
    assert not (store.root/'events.json').exists()
    resumed=SnapshotDataset.resume(store.root,profile,tmp_path/'robot.urdf')
    resumed.save(observation(profile,2),images)
    assert resumed.count==3
    with pytest.raises(ValueError,match='Calibration differs'):
        SnapshotDataset.resume(store.root,{**profile,'extra':'changed'},tmp_path/'robot.urdf')


@pytest.mark.parametrize('kind',['missing','resolution','stale','skew','moving','mapping','feedback','gap','nonfinite'])
def test_reject_bad_observation_without_publish(tmp_path,kind):
    profile,urdf,images=fixture_input(tmp_path);state=observation(profile)
    store=SnapshotDataset(tmp_path/'new','Invalid',profile,urdf)
    if kind=='missing':images.pop('real_wrist')
    elif kind=='resolution':images['real_side']=images['real_side'][:100]
    elif kind=='stale':state['timestamps']['follower']['end']['monotonic_ns']-=1_000_000_000
    elif kind=='skew':state['timestamps']['real_cameras']['real_side']['read_end']['monotonic_ns']-=150_000_000
    elif kind=='moving':state['real_stability']['stable']=False
    elif kind=='mapping':state['sim_joint_mapping']['sha256']='bad'
    elif kind=='feedback':state['measured_samples']=[]
    elif kind=='gap':state['measured_samples'][0]['timestamp_ns']=state['measured_samples'][1]['timestamp_ns']
    else:state['physical_follower_measured_joint_state']['elbow_flex']=float('nan')
    with pytest.raises(ValueError):store.save(state,images)
    assert not (store.root/'capture_manifest.yaml').exists() and store.count==0


def test_blur_is_saved_warning_but_not_fitting_input(tmp_path):
    from app.core.capture import import_snapshots
    from app.core.project import Project
    store,profile,images=dataset(tmp_path,poses=(0,))
    store.save(observation(profile,1),{k:np.zeros_like(v) for k,v in images.items()})
    assert store.last['warnings']
    p=Project.create(tmp_path/'projects','Blur')
    report=import_snapshots(p,store.root)
    assert report['imported']==4 and report['eligible']==2


def test_failed_index_write_retry_and_unindexed_crash_recovery(tmp_path,monkeypatch):
    from soarm101_lab.real2sim import capture_dataset as c
    from app.core.capture import load_snapshots
    store,profile,images=dataset(tmp_path,poses=(0,));before=(store.root/'capture_manifest.yaml').read_bytes()
    atomic=c.atomic
    def fail(path,data):
        if Path(path).name=='capture_manifest.yaml':raise OSError('simulated disk failure')
        return atomic(path,data)
    monkeypatch.setattr(c,'atomic',fail)
    with pytest.raises(OSError,match='disk failure'):store.save(observation(profile,1),images)
    assert (store.root/'capture_manifest.yaml').read_bytes()==before
    assert store.count==1 and len(load_snapshots(store.root)[1])==1
    monkeypatch.setattr(c,'atomic',atomic)
    orphan=store.root/'snapshots/snapshot_000002';orphan.mkdir();(orphan/'diagnostic.txt').write_text('crash remnant')
    store.save(observation(profile,1),images)
    assert store.last['id']=='snapshot_000003' and store.count==2
    assert (orphan/'diagnostic.txt').read_text()=='crash remnant'
    assert len(load_snapshots(store.root)[1])==2


def test_snapshot_hash_tampering_rejected(tmp_path):
    from app.core.capture import load_snapshots
    store,_,_=dataset(tmp_path,poses=(0,))
    (store.root/'snapshots/snapshot_000001/state.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):load_snapshots(store.root)


def test_import_append_no_events_pose_groups_and_static_states(tmp_path):
    from app.core.capture import import_snapshots,observation_scene
    from app.core.project import Project
    from app.core.scene import world_pose
    store,profile,images=dataset(tmp_path)
    originals={p:digest_file(p) for p in store.root.rglob('*') if p.is_file()}
    p=Project.create(tmp_path/'projects','Snapshots')
    report=import_snapshots(p,store.root,validation=['snapshot_000003'])
    assert report['eligible']==6
    a=observation_scene(p.scene,p.evidence['images'][0],p.root);b=observation_scene(p.scene,p.evidence['images'][2],p.root)
    assert not np.allclose(world_pose(a,a['cameras'][1]),world_pose(b,b['cameras'][1]))
    assert all(digest_file(path)==h for path,h in originals.items())
    first=copy.deepcopy(p.evidence['images'])
    store.save(observation(profile,3),images)
    assert import_snapshots(p,store.root,validation=['snapshot_000003'])['imported']==2
    assert p.evidence['images'][:6]==first
    with pytest.raises(ValueError,match='reassign'):import_snapshots(p,store.root)
    with pytest.raises(ValueError,match='No new'):import_snapshots(p,store.root,validation=['snapshot_000003'])
    store.save(observation(profile,4),images,'layout_002')
    with pytest.raises(ValueError,match='scene-state'):import_snapshots(p,store.root)
    with pytest.raises(ValueError,match='Different static'):import_snapshots(p,store.root,scene_state='layout_002')
    assert p.evidence['images'][:6]==first and len(p.evidence['images'])==8
    another=Project.create(tmp_path/'projects','Another layout')
    assert import_snapshots(another,store.root,scene_state='layout_002')['eligible']==2


def test_near_identical_pose_cannot_leak_train_to_holdout(tmp_path):
    from app.core.capture import import_snapshots
    from app.core.project import Project
    store,_,_=dataset(tmp_path,poses=(0,.01,2))
    p=Project.create(tmp_path/'projects','Leak')
    with pytest.raises(ValueError,match='near-identical'):
        import_snapshots(p,store.root,validation=['snapshot_000002'])
    assert not p.evidence['images']


@pytest.mark.parametrize('engine',['mujoco','blender'])
def test_snapshot_render_each_pose_and_dual_export(tmp_path,engine):
    from app.core.capture import import_snapshots,observation_scene
    from app.core.project import Project,read
    from app.validation.render import Renderer,for_observation
    from app.blender.backend import BlenderRenderer,capability
    from app.exporters.package import export_package,verify_package
    if engine=='blender' and not capability()['available']:pytest.skip('Installed Blender executable required')
    store,_,_=dataset(tmp_path,poses=(0,2));p=Project.create(tmp_path/'projects','Snapshot rendered')
    import_snapshots(p,store.root,validation=['snapshot_000002'])
    renderer=BlenderRenderer(p.scene,p.root,p.root/'scratch/base') if engine=='blender' else Renderer(p.scene,p.root,p.root/'scratch/base','software-egl')
    rendered=[]
    with renderer as model:
        for i,image in enumerate([p.evidence['images'][0],p.evidence['images'][2]]):
            with for_observation(p,p.scene,image,model,p.root/'scratch/posed'/str(i)) as posed:rendered.append(posed.rgb('side'))
    assert not np.array_equal(*rendered)
    package=export_package(p,{'passed':False,'input_signature':p.signature(),'physics_status':'PHYSICS_NOT_VALIDATED'},True)
    m=verify_package(package)
    assert all((package/name).is_file() for name in ('scene.yaml','scene.xml','scene.usd','scene_manifest.yaml','SCENE.md'))
    assert m['dataset_provenance'] and m['robot_model']['mapping_verified'] is False
    assert observation_scene(read(package/'scene.yaml'),p.evidence['images'][0],package)['objects']


def test_capture_command_never_accesses_robot_and_freezes_at_request(tmp_path):
    from soarm101_lab.real2sim.runtime import Runtime
    from concurrent.futures import ThreadPoolExecutor
    profile,urdf,images=fixture_input(tmp_path)
    class ForbiddenRobot:
        def __getattr__(self,name):raise AssertionError('Capture attempted hardware access: '+name)
    r=Runtime.__new__(Runtime);r.follower=ForbiddenRobot();r.cameras={k:object() for k in images}
    r.snapshot_requests=[];r.snapshot_store=None;r.pending=None;r.profile=profile;r.robot_model_path=urdf;r.hardware_identity=None
    r.snapshot_data=observation(profile);r.images=images;r.bus=Mock();r.writer=ThreadPoolExecutor(max_workers=1)
    q=copy.deepcopy(r.snapshot_data['physical_follower_measured_joint_state'])
    try:
        assert r.command({'command':'capture_snapshot','payload':{'output':str(tmp_path/'data')},'id':'capture'}) is None
        r.snapshot_data=observation(profile,3)  # Later robot pose must not replace the clicked observation.
        r.save_snapshot_requests();result=r.pending[0].result(timeout=10)
        assert result['count']==1
        saved=json.loads((Path(result['path'])/'state.json').read_text())
        assert saved['q_native']==q
        r.pending=None;r.save_snapshot_requests();assert r.snapshot_store.count==1
        for name in ('capture','dataset_record','render_dataset','export'):
            with pytest.raises(ValueError,match='Unsupported command'):
                r.command({'command':name,'payload':{},'id':'removed'})
        assert not (tmp_path/'captures').exists()
    finally:r.writer.shutdown()


def test_runtime_repeated_captures_share_session_and_common_files(tmp_path,monkeypatch):
    from soarm101_lab.real2sim.runtime import Runtime
    from soarm101_lab.real2sim.profile import digest as runtime_digest
    from soarm101_lab.real2sim.core import digest as snapshot_digest
    from soarm101_lab.real2sim import capture_dataset
    profile,urdf,images=fixture_input(tmp_path)
    # This difference caused the old runtime to allocate a new dataset every click.
    assert runtime_digest(profile)!=snapshot_digest(profile)
    r=Runtime(SimpleNamespace(workspace=str(tmp_path/'runtime'),backend='real'),None)
    r.profile=profile;r.robot_model_path=urdf;r.hardware_identity='fixture'
    r.follower=Mock();r.cameras={name:Mock() for name in images}
    r.snapshot_session=SnapshotDataset.new_session()
    copy_robot=Mock(wraps=capture_dataset.copy_robot)
    monkeypatch.setattr(capture_dataset,'copy_robot',copy_robot)
    roots=[];common=None
    try:
        for i in range(3):
            r.snapshot_data=observation(profile,i)
            r.snapshot_data.update(hardware_identity='fixture',profile_hash=runtime_digest(profile))
            r.images=images
            r.command({'command':'capture_snapshot','payload':{'output':str(tmp_path/'collected')},'id':str(i)})
            r.save_snapshot_requests()
            result=r.pending[0].result(timeout=10);r.pending=None
            roots.append(result['dataset'])
            m=json.loads((Path(result['dataset'])/'capture_manifest.yaml').read_text())
            current={path:((Path(result['dataset'])/path).stat().st_mtime_ns,digest_file(Path(result['dataset'])/path)) for path in m['files']}
            if common is None:common=current
            assert current==common
        assert len(set(roots))==1 and len(list((tmp_path/'collected').iterdir()))==1
        assert r.snapshot_store.count==3 and copy_robot.call_count==1
        assert Path(roots[0]).name==r.snapshot_session['id']
        r.stop();assert r.snapshot_store.count==3  # Stop Teleop doesn't end the session.
        r.disconnect_hardware()
        for f in r.snapshot_finalizers:f.result(timeout=10)
        r.collect_snapshot_finalizers()
        assert r.snapshot_store is None and r.snapshot_session is None
        manifest=json.loads((Path(roots[0])/'capture_manifest.yaml').read_text())
        assert len(manifest['snapshots'])==3
        assert manifest['session']['ended_at_unix_ns']>=manifest['session']['started_at_unix_ns']
        # Closed sessions remain readable by the independent Builder.
        from app.core.capture import load_snapshots
        assert len(load_snapshots(roots[0])[1])==3
    finally:r.disconnect_hardware();r.writer.shutdown();r.bus.close()


def test_runtime_rejects_mid_session_output_change_without_new_model(tmp_path):
    from soarm101_lab.real2sim.runtime import Runtime
    profile,urdf,images=fixture_input(tmp_path)
    r=Runtime(SimpleNamespace(workspace=str(tmp_path/'runtime'),backend='real'),None)
    r.profile=profile;r.robot_model_path=urdf
    r.follower=Mock();r.cameras={name:Mock() for name in images}
    try:
        for i,output in enumerate(('first','changed')):
            r.snapshot_data=observation(profile,i);r.images=images
            r.command({'command':'capture_snapshot','payload':{'output':str(tmp_path/output)},'id':str(i)})
            r.save_snapshot_requests()
            if i==0:r.pending[0].result(timeout=10);r.pending=None
        assert r.pending is None and not (tmp_path/'changed').exists()
        assert 'Disconnect' in r.error and r.snapshot_store.count==1
    finally:r.disconnect_hardware();r.writer.shutdown();r.bus.close()


def test_disconnect_finalizes_after_inflight_save_without_blocking_devices(tmp_path):
    from soarm101_lab.real2sim.runtime import Runtime
    from threading import Event
    store,profile,images=dataset(tmp_path,poses=(0,))
    r=Runtime(SimpleNamespace(workspace=str(tmp_path/'runtime'),backend='real'),None)
    r.snapshot_store=store;r.snapshot_session=store.session
    follower=Mock();r.follower=follower
    gate=Event();entered=Event();state=observation(profile,1);state['captured_host_ns']=time.monotonic_ns()
    def save():
        entered.set();assert gate.wait(10)
        return store.save(state,images)
    future=r.writer.submit(save)
    try:
        assert entered.wait(10)
        r.disconnect_hardware()
        follower.close.assert_called_once()
        assert not future.done() and not r.snapshot_finalizers[0].done()
        gate.set();future.result(timeout=10)
        r.snapshot_finalizers[0].result(timeout=10);r.collect_snapshot_finalizers()
        m=json.loads((store.root/'capture_manifest.yaml').read_text())
        assert len(m['snapshots'])==2 and m['session']['ended_at_unix_ns'] is not None
        with pytest.raises(ValueError,match='session is closed'):
            SnapshotDataset.resume(store.root,profile,tmp_path/'robot.urdf').save(observation(profile,2),images)
    finally:gate.set();r.writer.shutdown();r.bus.close()


def test_real_only_connect_teleop_stop_disconnect_uses_existing_devices(tmp_path,monkeypatch):
    from soarm101_lab.real2sim import runtime
    from soarm101_lab.real2sim.ui import ROOT
    from soarm101_lab.real2sim.profile import load
    r=runtime.Runtime(SimpleNamespace(workspace=str(tmp_path),backend='real'),None)
    profile_path=ROOT/'tests/real2sim/fixtures/scene_profile.json'
    profile=load(profile_path);state=observation(profile);native=state['physical_follower_measured_joint_state'];interval=state['timestamps']['follower']
    leader=Mock();follower=Mock()
    for device in (leader,follower):device.metadata={'calibration':{'sha256':'fixture'}};device.read.return_value=(native,interval)
    follower.send.return_value=(native,interval)
    monkeypatch.setattr(runtime,'Leader',lambda *a,**kw:leader);monkeypatch.setattr(runtime,'Follower',lambda *a,**kw:follower)
    opened=[]
    class Camera:
        error=None
        def __init__(self,path,w,h,fps):
            self.metadata={'resolved_device':path,'requested_size':[w,h]};self.w=w;self.h=h;opened.append(self)
        def start(self):pass
        def close(self):pass
        def nearest(self,t):return {'rgb':np.zeros((self.h,self.w,3),np.uint8),**state['timestamps']['real_cameras']['real_side']}
    monkeypatch.setattr(runtime,'CameraReader',Camera)
    r.build_environment=Mock(side_effect=AssertionError('real-only must not start an engine'))
    try:
        r.connect({'profile':str(profile_path),'hardware':{'leader_port':'/fake/leader','follower_port':'/fake/follower','follower_id':'fixture','leader_calibration':'fake.json','side_camera':'0','wrist_camera':'1'}})
        first_session=r.snapshot_session['id']
        assert first_session.startswith('snapshot_') and r.snapshot_store is None
        assert r.env is None and len(opened)==2
        r.build_environment.assert_not_called();follower.send.assert_not_called();follower.activate.assert_not_called()
        r.start();r.tick_real();follower.send.assert_called_once_with(native)
        r.stop();follower.read.return_value=(native,{'start':interval['start'],'end':{'monotonic_ns':interval['end']['monotonic_ns']+1,'unix_ns':0}})
        r.tick_real();assert follower.send.call_count==1
        r.apply(profile);assert r.profile==profile
        r.disconnect_hardware();follower.close.assert_called_once();leader.close.assert_called_once()
        assert r.follower is None and not r.running
        assert r.snapshot_session is None
        r.connect({'profile':str(profile_path),'hardware':{'leader_port':'/fake/leader','follower_port':'/fake/follower','follower_id':'fixture','leader_calibration':'fake.json','side_camera':'0','wrist_camera':'1'}})
        assert r.snapshot_session['id']!=first_session and r.snapshot_store is None
        assert follower.send.call_count==1  # Reconnect does not start teleop.
        r.disconnect_hardware()
    finally:r.writer.shutdown();r.bus.close()


def test_tab_changes_share_streams_and_do_not_connect_hardware(tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from PIL import Image
    from soarm101_lab.real2sim.ui import Window
    app=QApplication.instance() or QApplication([]);w=Window(tmp_path)
    try:
        w.runtime_launcher.connect=Mock(side_effect=AssertionError('Tab change tried to Connect'))
        w.runtime_enabled=True;w.runtime_session='test'
        status={'session':'test','backend':'real','real_connected':True,'real_camera_connected':True,'availability':{'robot_state':True,'real_side':'frame received','real_wrist':'frame received'}}
        monkeypatch.setattr(w.bus,'status',lambda:status);monkeypatch.setattr(w.bus,'responsive',lambda s,**kwargs:True)
        for name in ('real_side','real_wrist'):Image.new('RGB',(640,480),'green').save(w.bus.root/(name+'.jpg'))
        for index in (0,1,2,0):w.tabs.setCurrentIndex(index);w.refresh();app.processEvents()
        assert not w.capture_views['real_side'].pixmap().isNull()
        assert w.capture_views['real_side'].pixmap().cacheKey()!=0 and not w.views['real_side'].pixmap().isNull()
        assert w.capture_button.isEnabled()
        status['snapshot_dataset']={'root':'/data/snapshots/snapshot_20261008_213000_test','count':3}
        w.refresh()
        assert status['snapshot_dataset']['root'] in w.capture_session_label.text()
        assert not w.capture_output.isEnabled() and not w.capture_output_browse.isEnabled()
        assert not w.capture_name.isEnabled()
        for forbidden in ('editor','reference_view','capture_event','capture_purpose','capture_split'):
            assert not hasattr(w,forbidden)
        w.runtime_enabled=False;w.refresh()
        assert w.capture_output.isEnabled() and w.capture_output_browse.isEnabled()
        assert w.capture_name.isEnabled()
    finally:w.close();app.processEvents()


def test_real_worker_imports_no_simulation_engine(tmp_path):
    import subprocess,sys
    code='''
import importlib.abc, sys
class NoEngines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('isaaclab','omni','pxr','mujoco'):
            raise AssertionError('Unexpected engine import: '+fullname)
sys.meta_path.insert(0,NoEngines())
from soarm101_lab.real2sim.runtime import Runtime
from types import SimpleNamespace
r=Runtime(SimpleNamespace(workspace=sys.argv[1],backend='real'),None)
assert r.env is None
from pathlib import Path
assert not (Path(sys.argv[1])/'captures').exists()
r.disconnect_hardware();r.writer.shutdown();r.bus.close()
'''
    subprocess.run([sys.executable,'-B','-c',code,str(tmp_path)],check=True,timeout=30)


def test_snapshot_cli_import_and_freshness(tmp_path):
    from app.cli import main
    from app.core.project import Project
    store,_,_=dataset(tmp_path);p=Project.create(tmp_path/'projects','CLI snapshot')
    assert main(['import-snapshots',str(p.root),str(store.root),'--validation','snapshot_000003'])==0
    assert len(p.evidence['images'])==6
    before=p.signature()
    copied=p.root/p.evidence['images'][0]['snapshot_metadata']
    state=copied.parent/'state.json';state.write_text(state.read_text()+'\n')
    assert p.signature()!=before


def test_phone_reference_is_context_not_snapshot_calibration(tmp_path):
    from PIL import Image
    from app.cli import main
    from app.core.project import Project
    from app.reconstruction.pipeline import evaluate_project
    from app.calibration.preprocess import undistort_evidence
    store,_,_=dataset(tmp_path);p=Project.create(tmp_path/'projects','Reference plus snapshots')
    photo=tmp_path/'phone.jpg';Image.new('RGB',(731,529),'red').save(photo)
    main(['add',str(p.root),str(photo),'--kind','reference'])
    main(['import-snapshots',str(p.root),str(store.root),'--validation','snapshot_000003'])
    assert p.evidence['images'][0]['eligible'] is False
    assert p.evidence['images'][0]['role']=='reference_context'
    undistort_evidence(p)
    report=evaluate_project(p,'mujoco','software-egl')
    assert len(report['images'])==6 and not report['passed']
    assert not any(i['image']==p.evidence['images'][0]['path'] for i in report['images'])


def test_shared_calibration_and_robot_are_not_silently_reused(tmp_path):
    store,profile,images=dataset(tmp_path,poses=(0,))
    state=observation(profile,1);state['hardware_identity']='different_device_calibration'
    with pytest.raises(ValueError,match='Device calibration changed'):store.save(state,images)
    assert store.count==1
    (tmp_path/'robot.urdf').write_text((tmp_path/'robot.urdf').read_text()+'\n')
    with pytest.raises(ValueError,match='Robot model changed'):store.save(observation(profile,1),images)
    assert store.count==1


def test_real_worker_cli_idle_disconnect_without_engines_or_devices(tmp_path):
    import subprocess,sys
    from soarm101_lab.real2sim.storage import Bus
    from soarm101_lab.real2sim.ui import ROOT
    code='''
import importlib.abc,sys,runpy
class NoEngines(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('isaaclab','omni','pxr','mujoco'):
            raise AssertionError('Unexpected engine import: '+fullname)
sys.meta_path.insert(0,NoEngines())
script,workspace=sys.argv[1:]
sys.argv=[script,'--workspace',workspace,'--backend','real','--headless']
runpy.run_path(script,run_name='__main__')
'''
    log=(tmp_path/'worker.log').open('w')
    process=subprocess.Popen([sys.executable,'-B','-c',code,str(ROOT/'scripts/real2sim/real2sim_sim.py'),str(tmp_path)],stdout=log,stderr=subprocess.STDOUT)
    bus=Bus(tmp_path/'ipc');deadline=time.monotonic()+10
    try:
        while not (bus.owner_alive() and bus.status().get('backend')=='real'):
            if process.poll() is not None:raise AssertionError((tmp_path/'worker.log').read_text())
            if time.monotonic()>deadline:raise AssertionError('Real-only worker startup timed out')
            time.sleep(.02)
        assert bus.status()['state']=='IDLE'
        assert not bus.status()['real_connected'] and not bus.status()['sim_connected']
        assert not (tmp_path/'captures').exists()
        identifier=bus.send('disconnect')
        assert process.wait(timeout=10)==0
        assert json.loads((bus.root/('reply_'+identifier+'.json')).read_text())['ok']
        assert bus.status()['state']=='CLOSED'
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=10)
        log.close()


def test_snapshot_multi_pose_fk_wrist_fit_and_rollback(tmp_path):
    from app.core.project import Project,write
    from app.core.capture import import_snapshots,observation_scene
    from app.calibration.camera import project_point
    from app.calibration.multipose import fit
    from app.core.scene import world_pose
    p=Project.create(tmp_path/'projects','Multi pose')
    store,_,_=dataset(tmp_path)
    import_snapshots(p,store.root,validation=['snapshot_000003'])
    images=p.evidence['images'];truth=p.scene
    a=observation_scene(truth,images[0],p.root);b=observation_scene(truth,images[2],p.root)
    assert not np.allclose(world_pose(a,a['cameras'][1]),world_pose(b,b['cameras'][1]))
    evidence=p.evidence
    for image in images:
        if image['camera']!='wrist':continue
        posed=observation_scene(truth,image,p.root);camera=next(c for c in posed['cameras'] if c['id']=='wrist')
        for point in [[-.1,-.1,1],[.1,-.1,1],[-.1,.1,1],[.1,.1,1],[-.1,.1,1.2],[.1,.1,1.2]]:
            row={'observation':image['id'],'camera':'wrist','point_m':point,'uv':project_point(posed,camera,{'point_m':point})[0].tolist()}
            evidence['landmarks'].append(row)
    write(p.root/'input/evidence.yaml',evidence)
    changed=copy.deepcopy(truth);changed['cameras'][1]['pose']['value'][0][3]+=.025
    fitted,report=fit(p,changed,'camera:wrist')
    assert report['status']=='IMPROVED',report
    assert report['holdout_after']['rmse_px']<.01
    assert np.allclose(fitted['cameras'][1]['pose']['value'],truth['cameras'][1]['pose']['value'],atol=1e-4)
    assert images==p.evidence['images']  # measured state never rewritten by fitting
    # Exercise the complete captured-evidence candidate gate and report rollback.
    from app.core.candidates import Candidate,rollback
    from app.core.project import read
    p.save_scene(changed)
    previous={'passed':False,'input_signature':p.signature(),'test_marker':'baseline report'}
    write(p.root/'working/validation/report.json',previous)
    c=Candidate.create(p,'mount_recovery')
    assert c.fit('camera:wrist')['status']=='IMPROVED'
    evaluated=c.evaluate('mujoco',validation=True)
    assert evaluated['nonregression']['passed'],evaluated['nonregression']
    c.promote()
    assert read(p.root/'working/validation/report.json')['input_signature']==p.signature()
    rollback(p)
    assert p.scene==changed and read(p.root/'working/validation/report.json')==previous
    with pytest.raises(ValueError,match='locked'):fit(p,truth,'intrinsics:side')
    evidence['landmarks']=evidence['landmarks'][:6];write(p.root/'input/evidence.yaml',evidence)
    assert fit(p,changed,'camera:wrist')[1]['status']=='NEEDS_MORE_DATA'


def paired_observation(profile, index=0):
    state = observation(profile, index)
    state.update(backend='mujoco', workspace_context={'workspace_id':'fixture','version':1,
        'content_sha256':'a'*64,'builder':{'manifest_sha256':'b'*64},'scene_sha256':'c'*64},
        sim_measured_joint_state_rad=to_sim(state['physical_follower_measured_joint_state']),
        quality={'eligible_for_calibration':True,'reasons':[]},
        sim_cameras={n:{'K':c['K'],'resolution':c['resolution'],'T_world_optical':np.eye(4).tolist()} for n,c in profile['cameras'].items()})
    state['timestamps']['sim']=state['timestamps']['follower']
    return state


def test_pair_atomic_save_import_and_three_way_diagnostics(tmp_path):
    from app.core.project import Project
    from app.core.capture import import_snapshots, import_paired
    from app.validation.metrics import paired_diagnostics
    profile,urdf,images=fixture_input(tmp_path)
    initial=SnapshotDataset(tmp_path/'original','Initial',profile,urdf)
    initial.save(observation(profile,0),images)
    project=Project.create(tmp_path/'projects','Pair refinement')
    import_snapshots(project,initial.root)
    paired=SnapshotDataset(tmp_path/'pairs','Paired',profile,urdf)
    images.update(sim_side=np.zeros_like(images['real_side']),sim_wrist=np.zeros_like(images['real_wrist']))
    result=paired.save(paired_observation(profile,2),images,paired=True)
    folder=Path(result['path'])
    meta=json.loads((folder/'metadata.json').read_text())
    assert {'pair.json','sim_side.png','sim_wrist.png'} <= meta['files'].keys()
    import_paired(project,paired.root)
    obs=next(i for i in project.evidence['images'] if i.get('runtime_baseline'))
    assert obs['eligible'] and obs['runtime_baseline']['workspace_context']['version']==1
    real=images['real_'+obs['camera']]
    metrics=paired_diagnostics(project,obs,real,real,tmp_path/'diagnostics')
    assert metrics['real_runtime']['mae'] > 0
    assert metrics['candidate_runtime']['mae'] == metrics['real_runtime']['mae']
    baseline=project.root/obs['runtime_baseline']['path'];baseline.write_bytes(b'changed')
    with pytest.raises(ValueError,match='modified'):paired_diagnostics(project,obs,real,real,tmp_path/'bad')


def test_pair_mismatch_stored_but_excluded_and_missing_rgb_rejected(tmp_path):
    from app.core.capture import load_snapshots
    profile,urdf,images=fixture_input(tmp_path)
    store=SnapshotDataset(tmp_path/'pairs','Pair',profile,urdf)
    state=paired_observation(profile)
    with pytest.raises(ValueError,match='RGB'):store.save(state,images,paired=True)
    assert not (store.root/'COMPLETE').exists()
    images.update(sim_side=images['real_side'].copy(),sim_wrist=images['real_wrist'].copy())
    state['sim_measured_joint_state_rad'][JOINTS[0]]+=.2
    result=store.save(state,images,paired=True)
    _,rows,_=load_snapshots(store.root)
    assert rows[0]['quality']['eligible_for_calibration'] is False
    assert result['warnings']


def test_pair_explicit_quality_override_preserves_evidence(tmp_path):
    from app.core.project import Project
    from app.core.capture import import_snapshots, import_paired
    profile, urdf, images = fixture_input(tmp_path)
    initial = SnapshotDataset(tmp_path/'initial', 'Initial', profile, urdf)
    initial.save(observation(profile, 0), images)
    paired = SnapshotDataset(tmp_path/'pairs', 'Pairs', profile, urdf)
    state = paired_observation(profile, 2)
    state['sim_measured_joint_state_rad'][JOINTS[0]] += .2
    images.update(sim_side=images['real_side'].copy(), sim_wrist=images['real_wrist'].copy())
    result = paired.save(state, images, paired=True)
    state_path = Path(result['path'])/'state.json'
    original = state_path.read_bytes()
    for override in (None, 'User requests calibration mismatch diagnosis'):
        project = Project.create(tmp_path/'projects', 'Override' if override else 'Default')
        import_snapshots(project, initial.root)
        import_paired(project, paired.root, include_flagged_reason=override)
        rows = [i for i in project.evidence['images'] if i.get('runtime_baseline')]
        assert len(rows) == 2
        assert all(i['eligible'] == bool(override) for i in rows)
        assert all(i['capture_quality_reasons'] for i in rows)
        assert all(i['quality_override'] == override for i in rows)
        assert state_path.read_bytes() == original
    state_path.write_bytes(original + b' ')
    project = Project.create(tmp_path/'projects', 'Invalid hash')
    import_snapshots(project, initial.root)
    with pytest.raises(ValueError):
        import_paired(project, paired.root, include_flagged_reason='Do not bypass integrity')


def test_pair_command_freezes_four_frames_without_hardware_commands(tmp_path, monkeypatch):
    from soarm101_lab.real2sim.runtime import Runtime
    from soarm101_lab.real2sim.workspaces import package
    profile,urdf,images=fixture_input(tmp_path)
    class ForbiddenRobot:
        def __getattr__(self,name):raise AssertionError('Pair attempted hardware command: '+name)
    root=tmp_path/'workspace';root.mkdir();(root/'scene.yaml').write_text('scene');(root/'manifest.json').write_text('manifest')
    monkeypatch.setattr(package,'load',lambda path:{'root':root,'manifest':{'workspace_id':'fixture','version':1,
        'content_sha256':'a'*64,'builder':{'scene':'scene.yaml','manifest_sha256':'b'*64}}})
    r=Runtime.__new__(Runtime);r.follower=ForbiddenRobot();r.env=object();r.live_workspace=root
    r.workspace_context={'workspace_id':'fixture','version':1,'scene_sha256':digest_file(root/'scene.yaml')}
    r.cameras={k:object() for k in images};r.snapshot_requests=[]
    r.snapshot_data=paired_observation(profile)
    r.images={**images,'sim_side':images['real_side'].copy(),'sim_wrist':images['real_wrist'].copy()}
    r.command({'command':'capture_pair','payload':{},'id':'pair'})
    frozen=r.snapshot_requests[0]
    assert len(frozen['_snapshot_images'])==4
    r.images['sim_side'][:]=0
    assert frozen['_snapshot_images']['sim_side'].any()
    assert frozen['_snapshot_state']['workspace_context']['scene_sha256']==digest_file(root/'scene.yaml')


def test_pair_estimated_intrinsic_reconciliation_is_explicit(tmp_path):
    from app.core.project import Project
    from app.core.capture import import_snapshots, import_paired
    profile,urdf,images=fixture_input(tmp_path)
    for camera in profile['cameras'].values():camera['provenance']['K']='provisional'
    first=SnapshotDataset(tmp_path/'first','Old estimate',profile,urdf)
    first.save(observation(profile),images)
    p=Project.create(tmp_path/'projects','Reconcile');import_snapshots(p,first.root)
    changed=copy.deepcopy(profile);changed['cameras']['wrist']['K'][0][0]+=2
    s=p.scene;next(c for c in s['cameras'] if c['id']=='wrist')['K']['value']=changed['cameras']['wrist']['K'];p.save_scene(s)
    second=SnapshotDataset(tmp_path/'second','Runtime estimate',changed,urdf)
    images.update(sim_side=images['real_side'].copy(),sim_wrist=images['real_wrist'].copy())
    second.save(paired_observation(changed,2),images,paired=True)
    with pytest.raises(ValueError,match='reconciliation'):import_paired(p,second.root)
    import_paired(p,second.root,reconciliation='Runtime K matches canonical estimated K; original preserved')
    assert p.evidence['capture_sessions'][-1]['calibration_reconciliation']
    assert json.loads((first.root/'capture_manifest.yaml').read_text())['cameras'][1]['K'][0][0]==500
