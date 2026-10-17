import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from test_workspaces import published
from soarm101_lab.representation import real_action, real_observation, require_compatible, describe
from soarm101_lab.so101_joint_mapping import DEFAULT_MAPPER
from soarm101_lab.real2sim.workspaces.package import load, metadata
from soarm101_lab.real2sim.workspaces.demo import coordinate_contract
from soarm101_lab.real2sim.workspaces.physics import compile_physics
from soarm101_lab.real2sim.workspaces.reset import WorkspaceRandomizer
from soarm101_lab.workflow.artifacts import Registry, browse_directory
from soarm101_lab.workflow.operations import Operations
from soarm101_lab.workflow.hub import publication_metadata, publish


def test_hardware_roundtrip_and_gripper():
    native = {n + '.pos': v for n, v in zip(DEFAULT_MAPPER.metadata()['joint_order'], [20, -65, 30, 40, -80, 61])}
    q = real_observation(native)
    np.testing.assert_allclose(q, DEFAULT_MAPPER.calibrated_to_urdf(list(native.values())), atol=1e-7)
    command = real_action(q)
    np.testing.assert_allclose(list(command.values()), list(native.values()), atol=5e-6)
    for percent in (0, 25, 50, 100):
        native['gripper.pos'] = percent
        assert real_action(real_observation(native))['gripper.pos'] == pytest.approx(percent, abs=1e-5)


def test_source_uniform_reset_resume_and_isaac_dataset_binding(published,tmp_path):
    from test_source_partitions import source_fixture
    from soarm101_lab.real2sim.workspaces.reset import source_workspace
    from soarm101_lab.real2sim.workspaces.demo import verify_dataset
    from soarm101_lab.workflow.tasks import configured
    original=load(published[3]);new=source_workspace(configured(original))
    legacy=configured(original,reset={**original['task']['reset'],'mode':'uniform','yaw_rad':[0.,0.],'xy_jitter_m':0.})
    for name,w in [('old',configured(original)),('legacy_uniform',legacy),('new',new)]:
        path=tmp_path/(name+'.hdf5');source_fixture(path,4,w)
        bound,_=verify_dataset(path,published[3])
        resumed=source_workspace(original,path)
        assert bound['task']['reset']==w['task']['reset']==resumed['task']['reset']
    assert original['task']['reset']['mode']=='four_anchors' and new['task']['reset']['mode']=='uniform'
    assert new['task']['reset']['margin_m']==0. and legacy['task']['reset']['margin_m']==original['task']['reset']['margin_m']


def test_contract_semantics_and_mapping_rejection(published):
    c = coordinate_contract(load(published[3]))
    assert describe(c)['id'] == 'MODEL_JOINT_RAD_V1'
    assert describe(c)['action']['units'] == ['m'] * 3 + ['rad'] * 4
    bad = copy.deepcopy(c); bad['mapping']['arm_sign'][0] = -1
    with pytest.raises(ValueError, match='mapping'): require_compatible(bad, c)
    with pytest.raises(ValueError, match='semantic'): require_compatible(c, c, 'joint_degree')
    with pytest.raises(ValueError): describe({'provenance_status': 'legacy_unknown'})


def test_physics_roles(published):
    w = load(published[3]); physics = compile_physics(w['profile'], w['task'])['objects']
    assert physics['table']['collision'] and not physics['table']['rigid_body']
    assert physics['marker']['role'] == 'VISUAL_ONLY' and not physics['marker']['collision']
    assert physics['cube_red']['resettable'] and physics['cube_red']['mass_kg'] > 0
    assert physics['cup_a']['collision_approximation'] == 'open_segmented_walls_and_bottom'
    assert physics['cup_a']['kinematic']
    assert physics['robot']['source'] == 'model_asset'
    task = copy.deepcopy(w['task']); task['roles'] = {'cube_red': 'VISUAL_ONLY'}
    with pytest.raises(ValueError): compile_physics(w['profile'], task)


def test_compiled_static_collision(published):
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(published[3] / 'scene/static.usda'))
    assert stage.GetPrimAtPath('/Real2Sim/Workspace/table/collision_0').HasAPI(UsdPhysics.CollisionAPI)
    assert stage.GetPrimAtPath('/Real2Sim/Workspace/marker/collision_0').GetAttribute('physics:collisionEnabled').Get() is False
    assert (published[3] / 'physics.json').is_file()


def test_four_anchor_bounds_repeatability(published):
    w = copy.deepcopy(load(published[3])); w['task']['reset'].update(mode='four_anchors', xy_jitter_m=.02)
    a, b = WorkspaceRandomizer(w), WorkspaceRandomizer(w)
    for i in range(300):
        sample = a.sample(); assert sample == b.sample()
        assert sample['sampling']['sample_index'] == i + 1
        for p in sample['cube_positions'].values(): assert max(abs(p[0]), abs(p[1])) <= .053


def test_registry_dependencies_tamper_and_picker(published, tmp_path):
    registry = Registry(tmp_path / 'registry'); a = registry.register('workspace', published[3])
    report = tmp_path / 'replay.json'; report.write_text('{"passed":true}')
    b = registry.register('replay', report, [a['id']])
    assert b['workspace'] == a['workspace'] and b['parents'] == [a['id']]
    assert registry.latest({'replay'})['id'] == b['id']
    assert browse_directory('replay', registry=registry) == str(tmp_path)
    report.write_text('{"passed":false}')
    with pytest.raises(ValueError, match='changed'): registry.verify(b)


def test_operation_plan_preserves_cli_no_shell(published, tmp_path):
    r = Registry(tmp_path / 'registry'); a = r.register('workspace', published[3]); op = Operations(r)
    assert op.readiness('source', a, '', {})[0] == 'NOT_READY'
    opts = {'port': '/dev/test;echo not-a-shell', 'output': str(tmp_path / 'demo with space.hdf5')}
    assert op.readiness('source', a, '', opts)[0] == 'READY'
    plan = op.plan('source', a, '', opts, run_dir=tmp_path)
    assert plan['argv'][2:4] == ['scripts/real2sim/workspace.py', 'source-demo']
    assert opts['port'] in plan['argv'] and opts['output'] in plan['argv']
    assert '--recalibrate' not in plan['argv']


def test_failed_run_never_registers(published, tmp_path, monkeypatch):
    r = Registry(tmp_path / 'registry'); a = r.register('workspace', published[3]); op = Operations(r)
    monkeypatch.setenv('PYTHONPATH', '/gui/venv/lib/python3.11/site-packages')
    monkeypatch.setenv('PYTHONHOME', '/gui/venv')
    environments = []
    class Process:
        def __init__(self, *args, **kwargs): environments.append(kwargs['env'])
        def wait(self, **kwargs): return 3
    monkeypatch.setattr('soarm101_lab.workflow.operations.subprocess.Popen', Process)
    monkeypatch.setattr('soarm101_lab.workflow.operations.subprocess.run', lambda *a, **k: SimpleNamespace(stdout='test'))
    with pytest.raises(RuntimeError): op.run('source', a, options={'port': 'test'})
    states = list((r.root / 'runs').glob('*/state.json'))
    assert json.loads(states[0].read_text())['status'] == 'FAILED'
    assert len(r.all()) == 1
    from soarm101_lab.real2sim.workspaces.package import ROOT
    import os
    assert environments[0]['PYTHONPATH'].split(os.pathsep) == [str(ROOT / 'source/soarm101_lab')]
    assert 'PYTHONHOME' not in environments[0] and environments[0]['PYTHONNOUSERSITE'] == '1'


def test_cancelled_datagen_without_success_is_stopped_without_artifact(published,tmp_path,monkeypatch):
    r=Registry(tmp_path/'registry');a=r.register('workspace',published[3]);op=Operations(r)
    target=tmp_path/'output'
    plan={'output':str(target),'argv':['fake-worker'],'options':{'mode':'operator','backend':'mujoco','headless':True},
        'worker_report':True,'type':'datagen','result_key':'dataset'}
    monkeypatch.setattr(op,'plan',lambda *args,**kwargs:plan)
    class Process:
        def __init__(self,*args,**kwargs):
            target.mkdir();(target/'report.json').write_text(json.dumps({'passed':False,'stopped':True,'successes':0}))
        def wait(self,**kwargs):op.cancelled=True;return 1
    monkeypatch.setattr('soarm101_lab.workflow.operations.subprocess.Popen',Process)
    monkeypatch.setattr('soarm101_lab.workflow.operations.subprocess.run',lambda *a,**k:SimpleNamespace(stdout='test'))
    result=op.run('datagen',a,str(published[3]))
    assert result=={'status':'STOPPED','no_artifact':True} and len(r.all())==1
    state=json.loads(next((r.root/'runs').glob('*/state.json')).read_text())
    assert state['status']=='STOPPED' and 'error' not in state


def test_hub_metadata_no_live_upload(published, tmp_path):
    local = tmp_path / 'dataset'; local.mkdir(); (local / 'data.txt').write_text('test')
    artifact = {'id': 'test', 'type': 'dataset', 'path': str(local), 'workspace': metadata(load(published[3]))}
    opts = {'repo_id': 'test/data', 'private': True, 'revision': 'main'}
    calls = []
    class API:
        def repo_info(self, **kwargs): return SimpleNamespace(private=True)
        def upload_folder(self, **kwargs): calls.append(kwargs); return SimpleNamespace(oid='data-commit', commit_url='url1')
        def upload_file(self, **kwargs): calls.append(kwargs); return SimpleNamespace(oid='metadata-commit', commit_url='url2')
    before = list(local.iterdir())
    result = publish(artifact, opts, tmp_path / 'remote.json', api=API())
    assert result['commit'] == 'metadata-commit'
    assert calls[1]['parent_commit'] == 'data-commit'
    assert list(local.iterdir()) == before  # registered input remains immutable
    assert publication_metadata(artifact, opts)['workspace'] == artifact['workspace']


def test_source_artifact_contract_and_readiness(published, tmp_path):
    import h5py
    w = load(published[3]); demo = tmp_path / 'source.hdf5'
    with h5py.File(demo, 'w') as h:
        data = h.create_group('data')
        data.attrs['env_args'] = json.dumps({'workspace': metadata(w), 'so101_coordinates': coordinate_contract(w),
                                                     'grasp_object': 'cube_red', 'place_bin': 'cup_a'})
        episode = data.create_group('demo_0')
        for name in w['task']['dynamic']:episode.create_group('initial_state/rigid_object/'+name)
        episode.create_dataset('actions', data=np.zeros((2, 7)))
        episode.create_dataset('joint_targets', data=np.zeros((2, 6)))
        episode.create_dataset('obs/policy/joint_pos', data=np.zeros((2, 6)))
    r = Registry(tmp_path / 'registry'); source = r.register('source', demo)
    assert source['representation']['id'] == 'MODEL_JOINT_RAD_V1'
    assert Operations(r).readiness('annotate', source, str(published[3]))[0] == 'READY'
    bad = copy.deepcopy(coordinate_contract(w)); bad['mapping']['gripper_zero_offset_rad'] = .1
    with h5py.File(demo, 'a') as h:
        h['data'].attrs['env_args'] = json.dumps({'workspace': metadata(w), 'so101_coordinates': bad})
    bad_source = r.register('source', demo)
    assert Operations(r).readiness('annotate', bad_source, str(published[3]))[0] == 'NOT_READY'


def test_workflow_panel_shared_backend_and_no_autorun(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    app = QApplication.instance() or QApplication([])
    registry = Registry(tmp_path / 'registry')
    monkeypatch.setattr(ui, 'Registry', lambda: registry)
    calls = []
    panel = ui.WorkflowPanel(lambda *args: calls.append(args))
    panel.select_stage('train')
    assert not calls
    assert panel.service.registry is registry
    assert panel.current_screen.monitor.status.text().startswith('NOT_READY')
    assert panel.current_screen.stage_options()['architecture'] == 'smolvla'
    panel.close()


def test_source_and_datagen_replay_layout_no_remapping():
    from soarm101_lab.real2sim.workspaces.demo import replay_fields
    q = np.array([[.1, .2, .3, .4, .5, .6]])
    images = {'side_cam': np.zeros((1, 2, 2, 3)), 'wrist_cam': np.zeros((1, 2, 2, 3))}
    source = {'joint_targets': q, 'obs': {'policy': images}, 'states': {'joint_pos': q}}
    generated = {'joint_targets': q, 'obs': images,
                 'states': {'articulation': {'robot': {'joint_position': q}},
                            'rigid_object': {'cube': {'root_pose': np.zeros((1, 7))}}}}
    assert replay_fields(source)[1] is q
    assert replay_fields(generated)[1] is q
    assert 'cube' in replay_fields(generated)[2]
    with pytest.raises(ValueError): replay_fields({**source, 'joint_targets': np.zeros((2, 6))})


def test_source_timing_options(published, tmp_path):
    registry = Registry(tmp_path / 'registry')
    artifact = registry.register('workspace', published[3])
    operations = Operations(registry)
    options = {'port': '/dev/test', 'episode_time_s': 12.5, 'reset_time_s': 0}
    argv = operations.plan('source', artifact, '', options, run_dir=tmp_path)['argv']
    assert float(argv[argv.index('--episode_time_s') + 1]) == 12.5
    assert float(argv[argv.index('--reset_time_s') + 1]) == 0
    for key, value in [('episode_time_s', 0), ('reset_time_s', -1), ('episode_time_s', float('nan'))]:
        with pytest.raises(ValueError):
            operations.plan('source', artifact, '', {**options, key: value}, run_dir=tmp_path)


def test_overview_framing_is_read_only(published):
    from soarm101_lab.workflow.presentation import framing
    workspace = load(published[3])
    original = copy.deepcopy(workspace['profile'])
    eye, quaternion = framing(workspace)
    assert framing(workspace) == (eye, quaternion)
    assert np.isfinite(eye).all()
    assert np.linalg.norm(quaternion) == pytest.approx(1)
    assert workspace['profile'] == original


def test_readable_run_names():
    import re
    from soarm101_lab.workflow.operations import run_folder_name
    artifact = {'tasks': [{'task_id': 'cube_red_to_cup_a'}]}
    name = run_folder_name('annotate', artifact)
    assert re.fullmatch(r'\d{8}_\d{6}_annotation_cube_red_to_cup_a_[0-9a-f]{8}', name)
    assert run_folder_name('annotate', artifact) != name
    assert '_cube_blue_to_cup_b_' in run_folder_name('source', artifact, options={'tasks': [{'task_id': 'cube_blue_to_cup_b'}]})
    assert '_all_tasks_' in run_folder_name('sim_eval', artifact, options={'task_scope': 'all'})
    assert '_multi_2_tasks_' in run_folder_name('train', {'tasks': [{'task_id': 'a'}, {'task_id': 'b'}]})
    assert '_no_task_' in run_folder_name('hf_policy', {})
    assert '/' not in run_folder_name('source', {}, options={'tasks': [{'task_id': '../../bad/name'}]})


def test_success_target_name_and_legacy_alias():
    from soarm101_lab.workflow.operations import successful_demo_target
    assert successful_demo_target({}) == 10
    assert successful_demo_target({'num_successful_demos': 30}) == 30
    assert successful_demo_target({'trials': 20}) == 20
    with pytest.raises(ValueError):
        successful_demo_target({'trials': 20, 'num_successful_demos': 30})
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError): successful_demo_target({'num_successful_demos': value})


def test_datagen_seed_changes_layout_without_mutating_workspace(published, monkeypatch):
    from soarm101_lab.real2sim.workspaces.reset import datagen_seed
    w = load(published[3]); before = copy.deepcopy(w)
    import secrets
    values = iter([w['task']['reset']['seed'], 12345, 67890])
    monkeypatch.setattr(secrets, 'randbits', lambda bits: next(values))
    seed = datagen_seed(w['task']['reset']['seed'])
    assert seed == 12345
    assert datagen_seed(w['task']['reset']['seed']) == 67890
    a, b = WorkspaceRandomizer(w, seed), WorkspaceRandomizer(w, seed)
    original = WorkspaceRandomizer(w)
    assert a.sample() == b.sample()
    different = a.sample()
    assert different['cube_positions'] != original.sample()['cube_positions']
    assert different['sampling']['seed'] == seed
    assert w == before
    with pytest.raises(ValueError): datagen_seed(0, 0)
    with pytest.raises(ValueError): datagen_seed(0, -1)
    assert datagen_seed(0, 12345) == 12345


@pytest.mark.parametrize('fail_registration', [False, True])
def test_annotation_replaces_source_only_after_validation(published, tmp_path, monkeypatch, fail_registration):
    import h5py
    import shutil
    from soarm101_lab.so101_dataset_contract import write_contract, sidecar
    w = load(published[3]); source_path = tmp_path / 'source.hdf5'
    with h5py.File(source_path, 'w') as f:
        data = f.create_group('data')
        data.attrs['env_args'] = json.dumps({'workspace': metadata(w), 'so101_coordinates': coordinate_contract(w)})
        g = data.create_group('demo_0')
        g.create_dataset('actions', data=np.zeros((2, 7)))
        g.create_dataset('joint_targets', data=np.zeros((2, 6)))
        g.create_dataset('obs/policy/joint_pos', data=np.zeros((2, 6)))
    write_contract(source_path, coordinate_contract(w))
    registry = Registry(tmp_path / 'registry')
    source = registry.register('source', source_path)
    assert registry.register('source', source_path)['id'] == source['id']
    result_path = tmp_path / 'annotated.hdf5'
    shutil.copy2(source_path, result_path)
    write_contract(result_path, {**coordinate_contract(w), 'source_dataset': str(source_path)})
    before = source_path.read_bytes(), sidecar(source_path).read_bytes()
    with pytest.raises(ValueError, match='Missing Mimic annotation'):
        registry.promote_annotation(source, result_path)
    with h5py.File(result_path, 'a') as f:
        f['data/demo_0'].create_dataset('obs/datagen_info/subtask_term_signals/grasp', data=[False, True])
    if fail_registration:
        monkeypatch.setattr(registry, 'register', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('registration failed')))
        with pytest.raises(RuntimeError, match='registration failed'):
            registry.promote_annotation(source, result_path)
        assert before == (source_path.read_bytes(), sidecar(source_path).read_bytes())
        assert result_path.exists() and registry.get(source['id'])['status'] == 'SUCCEEDED'
        registry.verify(source)
    else:
        result = registry.promote_annotation(source, result_path)
        assert result['path'] == str(source_path) and result['type'] == 'annotated'
        assert registry.get(source['id'])['status'] == 'SUPERSEDED'
        assert registry.get(source['id'])['superseded_by'] == result['id']
        registry.verify(result)
        assert not result_path.exists()
    assert not list(tmp_path.glob('.*annotation_backup*'))


@pytest.mark.parametrize('canonical', [False, True])
def test_partition_source_annotation_cannot_replace_original(published, tmp_path, canonical):
    from test_source_partitions import source_fixture
    from soarm101_lab.workflow.episodes import export_source_splits
    from soarm101_lab.workflow.artifacts import fingerprint
    source_path=tmp_path/'source.hdf5';source_fixture(source_path,4,load(published[3]))
    plan=export_source_splits(source_path,tmp_path/'split')
    path=Path(plan['outputs']['train'])
    if canonical:
        path=tmp_path/'canonical';folder=path/'episode';folder.mkdir(parents=True)
        for name in ('state','action'):np.save(folder/(name+'.npy'),np.zeros((3,6),np.float32))
        (folder/'outcome.json').write_text('{}')
        (path/'manifest.json').write_text(json.dumps({'episodes':[{'id':'episode','path':'episode'}]}))
        (path/'project_provenance.json').write_text(json.dumps({'workspace':metadata(load(published[3])),
            'coordinates':coordinate_contract(load(published[3])),'source_partition':plan['partitions']['train']}))
    registry=Registry(tmp_path/'registry');source=registry.register('source',path)
    before=fingerprint(path)
    with pytest.raises(ValueError,match='Source Partition is immutable'):
        registry.promote_annotation(source,tmp_path/'annotation_not_opened.hdf5')
    assert fingerprint(path)==before and registry.get(source['id'])['status']=='SUCCEEDED'




def test_workspace_live_profile_updates_static_and_dynamic_layout(published, monkeypatch):
    import sys
    from unittest.mock import Mock
    from pxr import Usd, UsdGeom, UsdLux
    from soarm101_lab.real2sim.scene import apply_profile
    w = load(published[3]); p = copy.deepcopy(w['profile'])
    p['workspace']['T_world'][0][3] += .05
    stage = Usd.Stage.CreateInMemory()
    UsdLux.DomeLight.Define(stage, '/World/Light')
    stage.DefinePrim('/World/envs/env_0/WorkspaceStatic').GetReferences().AddReference(str(published[3] / 'scene/static.usda'))
    monkeypatch.setitem(sys.modules, 'isaaclab.sim', SimpleNamespace(get_current_stage=lambda: stage))
    class Scene(dict):
        def update(self, dt): pass
    scene = Scene(robot=Mock())
    for name in [*w['task']['dynamic'], *w['task']['targets']]: scene[name] = Mock()
    for view in ('side', 'wrist'):
        path = '/World/' + view
        UsdGeom.Camera.Define(stage, path)
        width, height = p['cameras'][view]['resolution']
        scene['camera_'+view+'view'] = SimpleNamespace(
            cfg=SimpleNamespace(width=width, height=height),
            _view=SimpleNamespace(prim_paths=[path]), set_intrinsic_matrices=Mock())
    env = SimpleNamespace(scene=scene, device='cpu', sim=Mock())
    apply_profile(env, p, workspace=w)
    workspace = UsdGeom.Xformable(stage.GetPrimAtPath('/World/envs/env_0/WorkspaceStatic/Workspace'))
    assert np.asarray(workspace.GetLocalTransformation())[3, 0] == pytest.approx(.05)
    for name in [*w['task']['dynamic'], *w['task']['targets']]:
        visual = w['task']['targets'].get(name, name)
        expected = np.asarray(p['workspace']['T_world']) @ np.asarray(p['objects'][visual]['T_workspace'])
        actual = scene[name].write_root_pose_to_sim.call_args.args[0].numpy()[0, :3]
        np.testing.assert_allclose(actual, expected[:3, 3], atol=1e-6)


def test_retired_profile_publication_is_not_an_executable_stage():
    from soarm101_lab.workflow.operations import STAGES, execute_request
    from soarm101_lab.real2sim.workspaces import package
    assert not {'accept', 'publish'} & set(STAGES)
    assert not hasattr(package, 'publish')
    with pytest.raises(ValueError, match='accept'):
        execute_request({'stage':'accept', 'input':{}, 'options':{}, 'output':'unused'})
