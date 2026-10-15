"""Scene composition contracts and real MuJoCo model checks (no device motion)."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from soarm101_lab.workflow.tasks import active_objects,configured,catalog,recorded_scene,object_metrics,evaluation_distribution
from soarm101_lab.real2sim.workspaces.reset import WorkspaceRandomizer
from soarm101_lab.real2sim.workspaces import package
from test_builder_workspace import make_export


def export_with_new_object(tmp_path):
    import yaml
    import xml.etree.ElementTree as ET
    root=make_export(tmp_path)
    path=root/'scene.yaml';scene=yaml.safe_load(path.read_text())
    obj=copy.deepcopy(next(o for o in scene['objects'] if o['id']=='cube_red'))
    obj.update(id='new_manipulable');obj['dimensions']['value']=[.03,.02,.02]
    obj['pose']['value'][0][3]=.055;obj['pose']['value'][1][3]=.055
    scene['objects'].append(obj);path.write_text(yaml.safe_dump(scene))
    xml=ET.parse(root/'scene.xml');body=ET.SubElement(xml.getroot().find('worldbody'),'body',name=obj['id'])
    for kind in ('visual','collision'):ET.SubElement(body,'geom',name=obj['id']+'_'+kind+'_0',type='box',size='.015 .01 .01')
    xml.write(root/'scene.xml')
    usd=root/'scene.usd';usd.write_text(usd.read_text().rstrip()[:-1]+'\ndef Material "material_new_manipulable" {}\ndef Xform "new_manipulable" {\ndef Cube "visual_0" {}\ndef Cube "collision_0" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}\n}\n}\n')
    path=root/'scene_manifest.yaml';manifest=yaml.safe_load(path.read_text())
    manifest['files']={p.relative_to(root).as_posix():package.sha(p) for p in root.rglob('*') if p.is_file() and p!=path}
    path.write_text(yaml.safe_dump(manifest));return root


@pytest.fixture
def workspace(tmp_path):
    from soarm101_lab.real2sim.workspaces.builder import import_scene
    path=import_scene(make_export(tmp_path),registry=tmp_path/'workspaces')
    return package.load(path)


def definition(w,pick='cube_red'):
    return next(t for t in catalog(w) if t['pick_object_id']==pick)


def test_source_uniform_xy_preserves_original_and_recorded_policy(workspace,tmp_path):
    from soarm101_lab.real2sim.workspaces.reset import source_workspace
    from soarm101_lab.real2sim.precision import simulation_configuration
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    original=copy.deepcopy(workspace)
    run=source_workspace(configured(workspace,definition(workspace)))
    assert workspace==original and run['task']['reset']['mode']=='uniform'
    assert run['task']['reset']['yaw_rad']==[0.,0.]
    assert configured(run)['scene_configuration']==run['scene_configuration']
    assert recorded_scene({'scene_configuration':run['scene_configuration']},workspace)==run['scene_configuration']
    old=configured(workspace)['scene_configuration']
    assert recorded_scene({'scene_configuration':old},workspace)['reset']==original['task']['reset']
    randomizer=WorkspaceRandomizer(run,seed=41);points=[]
    support=run['profile']['objects'][run['task']['reset']['support_object']]
    center=np.array(support['T_workspace'])[:2,3];half=np.array(support['dimensions_m'])[:2]/2
    for _ in range(500):
        sample=randomizer.sample()['cube_positions'];names=list(sample)
        for i,name in enumerate(names):
            xy=np.array(sample[name][:2]);size=np.array(run['profile']['objects'][name]['dimensions_m'])[:2]
            assert np.all(abs(xy-center)+size/2+run['task']['reset']['margin_m']<=half+1e-10)
            points.append(xy-center)
            for other in names[:i]:
                other_size=np.array(run['profile']['objects'][other]['dimensions_m'])[:2]
                assert np.any(abs(xy-np.array(sample[other][:2]))>(size+other_size)/2+run['task']['reset']['separation_m'])
    assert np.any(np.linalg.norm(points,axis=1)<.01) # The center gap of four-anchor sampling is populated.
    simulation=simulation_configuration(run,backend='mujoco',scene_configuration=run['scene_configuration'])
    with MujocoBackend(tmp_path/'uniform_model',workspace['root'],render_backend='none',task_definition=run['task_definition'],
                       simulation_configuration=simulation) as backend:
        assert backend.randomizer.sample()['sampling']['mode']=='uniform'
        assert backend.metadata['scene_configuration']['reset']==run['task']['reset']
        backend.reset(seed=41)
        assert np.isfinite(backend.data.qpos).all()
        for name in run['task']['dynamic']:
            pose=backend.get_state()['object_poses'][name]
            np.testing.assert_allclose(np.array(pose)[:3,:3],np.eye(3),atol=1e-7)


@pytest.mark.parametrize('ids',[['cube_red'],['cube_red','cube_blue'],['cube_red','cube_green','cube_blue'],['cube_green','cube_red']])
def test_runtime_filters_without_mutating_workspace(workspace,ids):
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    w=workspace;before=copy.deepcopy(w);t=definition(w)
    run=configured(w,t,ids,'mujoco');active=run['task']['dynamic']
    assert active==[n for n in w['task']['dynamic'] if n in ids]
    assert w==before
    with MujocoBackend(w['root'].parent.parent/('runtime_'+str(len(ids))+'_'+ids[0]),w['root'],task_definition=t,active_object_ids=ids,render_backend='none') as e:
        for name in w['task']['dynamic']:
            present=name in ids
            assert (e.mj.mj_name2id(e.model,e.mj.mjtObj.mjOBJ_BODY,name)>=0)==present
            assert any('builder_'+name+'_' in (e.model.geom(i).name or '') for i in range(e.model.ngeom))==present
            assert (name in e.object_indices)==present
        assert set(e.get_state()['object_poses'])==set(ids)|set(w['task']['targets'])
        assert 'builder_camera_case_visual_0' in [e.model.geom(i).name for i in range(e.model.ngeom)]
        for _ in range(20):e.step(e.home)
        assert np.isfinite(e.data.qpos).all()
        assert set(e.randomizer.sample()['cube_positions'])==set(ids)
    package.load(w['root'])  # immutable package checksum still valid


def test_validation_and_new_dynamic_object(workspace):
    for ids in ([],['missing'],['table'],['cube_red','cube_red']):
        with pytest.raises(ValueError):active_objects(workspace,ids)
    with pytest.raises(ValueError,match='Pick target'):configured(workspace,definition(workspace),['cube_blue'])
    w=copy.deepcopy(workspace);name='new_manipulable';w['task']['dynamic'].append(name)
    w['profile']['objects'][name]=copy.deepcopy(w['profile']['objects']['cube_red']);w['profile']['objects'][name]['dimensions_m']=[.03,.02,.02]
    run=configured(w,definition(w),['cube_red',name]);assert name in run['task']['dynamic']
    assert all(t['pick_object_id']!=name for t in catalog(run))
    w['task']['dynamic']+=['extra_a','extra_b']
    for n in ['extra_a','extra_b']:w['profile']['objects'][n]=copy.deepcopy(w['profile']['objects']['cube_red'])
    with pytest.raises(ValueError,match='four'):active_objects(w,w['task']['dynamic'])


def test_reset_determinism_and_no_overlap(workspace):
    for ids in [['cube_red'],['cube_red','cube_blue'],workspace['task']['dynamic']]:
        w=configured(workspace,definition(workspace),ids);a=WorkspaceRandomizer(w,73);b=WorkspaceRandomizer(w,73)
        for _ in range(15):
            x=a.sample();assert x==b.sample();assert set(x['cube_positions'])==set(ids)
            for i,n in enumerate(ids):
                for other in ids[:i]:
                    size=np.array(w['profile']['objects'][n]['dimensions_m'])[:2]+np.array(w['profile']['objects'][other]['dimensions_m'])[:2]
                    assert np.any(np.abs(np.array(x['cube_positions'][n])[:2]-x['cube_positions'][other][:2])>size/2)


def test_recorded_identity_and_eval_composition_shift(workspace):
    s=configured(workspace,definition(workspace),['cube_red'],'mujoco')['scene_configuration']
    a={'provenance':{'scene_configuration':s}}
    assert recorded_scene(a,workspace)==s
    change=evaluation_distribution(a,workspace,['cube_red','cube_blue'])
    assert change['scene_composition_distribution_shift'] is True
    bad=copy.deepcopy(a);bad['provenance']['scene_configuration']['active_object_ids']=['cube_blue']
    with pytest.raises(ValueError,match='hash'):recorded_scene(bad,workspace)
    with pytest.raises(ValueError,match='UNKNOWN'):recorded_scene({},workspace)


def test_object_metrics_single_na_all_distractors():
    positions={'one':np.array([0.,0,0]),'two':np.array([.1,0,0]),'three':np.array([.2,0,0])}
    row=object_metrics(['one'],'one',{'one':positions['one']},np.zeros(3),{})
    assert row['target_selection_accuracy'] is None and row['wrong_object_approach'] is None
    row=object_metrics(list(positions),'one',positions,np.array([.1,0,0]),{})
    assert row['selected_object_id']=='two' and row['wrong_object_approach'] is True
    assert set(row['object_distances_m'])==set(positions) and not row['full_task_success']


def test_new_builder_object_is_real_dynamic_distractor(tmp_path):
    from soarm101_lab.real2sim.workspaces.builder import import_scene
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    w=package.load(import_scene(export_with_new_object(tmp_path),registry=tmp_path/'workspaces'))
    with MujocoBackend(tmp_path/'sim',w['root'],render_backend='none',active_object_ids=['cube_red','new_manipulable']) as e:
        assert set(e.object_indices)=={'cube_red','new_manipulable'}
        assert e.model.body('new_manipulable').mass>0
        assert all(t['pick_object_id']!='new_manipulable' for t in catalog(e.workspace))
        for _ in range(20):e.step(e.home)
        assert np.isfinite(e.data.qpos).all()
    t=definition(w);t['pick_object_id']='new_manipulable'
    with pytest.raises(ValueError,match='semantic'):configured(w,t,['new_manipulable'])


def test_ui_filters_objects_and_requires_explicit_task(workspace):
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from soarm101_lab.workflow.task_ui import TaskSelector
    app=QApplication.instance() or QApplication([])
    selector=TaskSelector();selector.configure(workspace,'source')
    assert len(selector.object_selection())==3
    selector.configure(workspace,'source',active_object_ids=['cube_red','cube_green'])
    assert selector.pick.count()==2 and 'cube_blue' not in [selector.pick.itemText(i) for i in range(selector.pick.count())]
    selector.pick.setCurrentText('cube_green')
    selector.configure(workspace,'source',active_object_ids=['cube_red'])
    with pytest.raises(ValueError,match='conflicts'):selector.selection()
    selector.preset.setCurrentIndex(0);assert selector.selection()[0]['pick_object_id']=='cube_red'
    selector.configure(workspace,'source',active_object_ids=[])
    with pytest.raises(ValueError,match='least one'):selector.object_selection()
    selector.configure(workspace,'replay',[definition(workspace)],['cube_red'])
    assert not selector.objects.isEnabled() and selector.object_selection()==['cube_red']
    selector.setEnabled(False);assert not selector.objects.isEnabled()
    selector.close()


def record_fixture(path,w,ids,episodes=1):
    from soarm101_lab.simulation.mujoco_backend import PolicyEnvironment
    from soarm101_lab.workflow.episodes import EpisodeWriter
    with PolicyEnvironment(path.parent/(path.name+'_scene'),w['root'],'none',full_workspace=True,
                           task_definition=definition(w),active_object_ids=ids) as env:
        env.backend.get_rgb=lambda camera=None:np.zeros((24,32,3),np.uint8)
        writer=EpisodeWriter(path,env,definition(w)['language_instruction'])
        for i in range(episodes):
            env.reset(i);writer.begin()
            for _ in range(4):
                writer.append(env.observe(),env.backend.home);env.backend.step(env.backend.home)
            writer.save(True) # Synthetic labels for metadata tests only, no grasp-success claim.
    return writer


def test_source_replay_annotation_datagen_inherit(workspace,tmp_path,monkeypatch):
    from soarm101_lab.workflow import mujoco as worker,episodes
    from soarm101_lab.workflow.artifacts import fingerprint
    writer=record_fixture(tmp_path/'source',workspace,['cube_red'])
    before=fingerprint(writer.root);seen=[]
    original=worker._environment
    def factory(request,output,*args):
        env=original(request,output,*args)
        assert env.backend.workspace['task']['dynamic']==['cube_red']
        seen.append(env.backend.metadata['scene_configuration'])
        env.backend.get_rgb=lambda camera=None:np.zeros((24,32,3),np.uint8)
        return env
    monkeypatch.setattr(worker,'_environment',factory)
    request={'workspace':str(workspace['root']),'input':str(writer.root),'renderer':'none',
             'options':{'active_object_ids':workspace['task']['dynamic'],'realtime':False,'auto':True}}
    report=worker.replay(request,tmp_path/'replay',lambda *args:None,None)
    assert report['passed'] and report['max_joint_error_rad']<1e-10
    (tmp_path/'annotation').mkdir()
    report=episodes.annotate(request,tmp_path/'annotation',lambda *args:None,None,factory,worker._preview)
    assert not report['passed'] # Holding home is not a valid manipulation.
    entry=writer.manifest['episodes'][0]
    (writer.root/entry['path']/'annotation.json').write_text(json.dumps({'schema':'so101.mujoco-subtasks/1','grasp':1,'place':3,'frame_count':4}))
    original_after_label=fingerprint(writer.root)
    request['options'].update(instruction=writer.instruction,num_successful_demos=1,max_attempts=1)
    generated=episodes.datagen(request,tmp_path/'generated',lambda *args:None,None,factory,worker._preview)
    assert generated['attempts']==1 and generated['successes']==0
    assert all(s['active_object_ids']==['cube_red'] for s in seen)
    assert fingerprint(writer.root)==original_after_label and before!=original_after_label
    assert json.loads((Path(generated['dataset'])/'project_provenance.json').read_text())['scene_configuration']['active_object_ids']==['cube_red']


def test_resume_and_curated_mixed_compositions(workspace,tmp_path):
    import shutil
    from soarm101_lab.simulation.mujoco_backend import PolicyEnvironment
    from soarm101_lab.workflow.episodes import EpisodeWriter,prepare_dataset
    from soarm101_lab.workflow.curation import export_subset
    from soarm101_lab.workflow.artifacts import inspect_artifact
    a=record_fixture(tmp_path/'single',workspace,['cube_red'],2)
    b=record_fixture(tmp_path/'double',workspace,['cube_red','cube_blue'])
    with PolicyEnvironment(tmp_path/'resume_scene',workspace['root'],'none',task_definition=definition(workspace),active_object_ids=['cube_red','cube_blue']) as env:
        resumed=EpisodeWriter(tmp_path/'resume',env,a.instruction)
        with pytest.raises(ValueError,match='active'):resumed.resume(a.root)
    mixed=tmp_path/'mixed';shutil.copytree(a.root,mixed)
    entry=copy.deepcopy(b.manifest['episodes'][0]);shutil.copytree(b.root/entry['path'],mixed/'episode_0002')
    entry.update(id='episode_0002',path='episode_0002');manifest=copy.deepcopy(a.manifest);manifest['episodes'].append(entry)
    (mixed/'manifest.json').write_text(json.dumps(manifest))
    prepared=tmp_path/'prepared';prepare_dataset(mixed,prepared)
    prov=json.loads((prepared/'project_provenance.json').read_text())
    assert len(prov['scene_configurations'])==2
    assert inspect_artifact('dataset',prepared)['provenance']['scene_configurations']==prov['scene_configurations']
    assert [e['scene_configuration']['active_object_ids'] for e in json.loads((prepared/'manifest.json').read_text())['episodes']]==[['cube_red'],['cube_red'],['cube_red','cube_blue']]
    selected=export_subset(prepared,tmp_path/'curated',['episode_0002'])
    assert json.loads((selected/'project_provenance.json').read_text())['scene_configurations']==[entry['scene_configuration']]
    from soarm101_lab.learning.minivla.config import MiniVLAConfig
    cfg=MiniVLAConfig();assert cfg.horizon==40 and cfg.layers==4


def test_hdf5_legacy_evidence_unknown_and_resume(workspace,tmp_path):
    import h5py
    from soarm101_lab.workflow.tasks import bind_scene_options
    from soarm101_lab.workflow.data_resume import validate as resume
    from soarm101_lab.real2sim.workspaces.demo import coordinate_contract
    s=configured(workspace,definition(workspace),['cube_red'],'isaac')['scene_configuration']
    path=tmp_path/'old.hdf5'
    args={'workspace':package.metadata(workspace),'so101_coordinates':coordinate_contract(workspace),'task_definitions':[definition(workspace)]}
    with h5py.File(path,'w') as h:
        d=h.create_group('data');d.attrs['env_args']=json.dumps(args)
        g=d.create_group('demo_0');g.create_dataset('actions',data=np.zeros((2,7)));g.create_dataset('joint_targets',data=np.zeros((2,6)));g.create_dataset('obs/policy/joint_pos',data=np.zeros((2,6)))
        g.create_group('initial_state/rigid_object/cube_red')
    a={'path':str(path),'env_args':args}
    assert bind_scene_options('datagen',a,workspace,{})['active_object_ids']==['cube_red']
    with pytest.raises(ValueError,match='inherit'):bind_scene_options('annotate',a,workspace,{'active_object_ids':['cube_blue']})
    assert resume('source',path,package.metadata(workspace),[definition(workspace)],scene_configuration=s,workspace_definition=workspace)==1
    bad=configured(workspace,definition(workspace),['cube_red','cube_blue'],'isaac')['scene_configuration']
    with pytest.raises(ValueError,match='active'):resume('source',path,package.metadata(workspace),[definition(workspace)],scene_configuration=bad,workspace_definition=workspace)
    with h5py.File(path,'a') as h:del h['data/demo_0/initial_state']
    with pytest.raises(ValueError,match='UNKNOWN'):recorded_scene(a,workspace)
