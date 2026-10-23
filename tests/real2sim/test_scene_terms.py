"""Scene-only recording and explicit Terms use the existing engine and Worker."""
import copy
from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
import pytest
from test_builder_workspace import exported
from test_common_mimic import episode
from soarm101_lab.real2sim.workspaces import builder,package
from soarm101_lab.real2sim.workspaces.tasks import configured,generate_template,catalog
from soarm101_lab.workflow.task_scripts import read_script,script_inputs,resolve_script,read_followup,followup_context,load_followup,validate_processing


@pytest.fixture
def workspace(exported,tmp_path):
    # Synthetic export places the robot at support centre. Move ONLY this new
    # fixture's robot away so random resets can exercise valid free support.
    import yaml
    path=exported/'scene.yaml';scene=yaml.safe_load(path.read_text())
    transform=scene['frames']['robot_base']['transform']['value'];transform[0][3]=.4;transform[1][3]=.4
    path.write_text(yaml.safe_dump(scene))
    manifest=exported/'scene_manifest.yaml';data=yaml.safe_load(manifest.read_text());data['files']['scene.yaml']=package.sha(path);manifest.write_text(yaml.safe_dump(data))
    return package.load(builder.import_scene(exported,registry=tmp_path/'workspaces'))


def scene_env(w,names):
    from soarm101_lab.simulation.environment import create_mujoco
    return create_mujoco(configured(w,active_object_ids=[n for n in names if n in w['task']['dynamic']],
        scene_objects=list(names),scene_only=True,object_configuration=names,backend='mujoco'),renderer='none')


def source_script(tmp_path,w,objects,extra=None,code=''):
    config=dict(stage='source',workspace=str(w['root']),scene={'objects':objects},options=dict(backend='mujoco',port='virtual',storage_name='new',num_envs=1))
    config.update(extra or {});path=tmp_path/'user.py';path.write_text('SCRIPT = '+repr(config)+'\n'+code);return path


def test_generator_and_minimal_source_have_no_hidden_task(workspace,tmp_path):
    w=workspace;path=generate_template(w,tmp_path/'source.py',script_stage='source')
    config,_,_=read_script(path,'source');assert isinstance(config['scene']['objects'],dict) and 'source' not in config
    _,_,opts=script_inputs(path,'source');opts=resolve_script(w,opts)
    assert opts['tasks']==[] and opts['task_definition'] is None and opts['scene_only']
    with scene_env(w,{'cube_red':{}}) as env:
        b=env.backend
        assert b.metadata['dynamic_objects']==['cube_red'] and b.metadata['target_objects']==[] and b.is_success() is None
        np.testing.assert_allclose(b.get_object_pose('cube_red'),w['profile']['objects']['cube_red']['T_workspace'],atol=1e-10)
        assert env.recording_context()['scene']['task_definition'] is None


def test_pose_reset_only_moves_declared_body(workspace,tmp_path):
    from soarm101_lab.simulation.task_context import reset_objects
    def reset_scene(env,request,seed):
        env.reset(seed);reset_objects(env.backend,request['options']['script_scene']['objects'],request['_scene_resets'])
    from soarm101_lab.task_terms.reset import uniform,edge_uniform,metadata_pose
    w=workspace;objects={'cube_red':{},'cube_blue':{}}
    with scene_env(w,objects) as env:
        blue=np.asarray(w['profile']['objects']['cube_blue']['T_workspace'])
        request={'options':{'scene_only':True,'script_scene':{'objects':{'cube_red':{'reset':'red'},'cube_blue':{}}}},'_scene_resets':{'cube_red':uniform}}
        for fn in (uniform,edge_uniform):
            request['_scene_resets']['cube_red']=fn
            reset_scene(env,request,1047);a=env.backend.get_object_pose('cube_red')
            reset_scene(env,request,1047);np.testing.assert_array_equal(env.backend.get_object_pose('cube_red'),a)
            np.testing.assert_allclose(env.backend.get_object_pose('cube_blue'),blue,atol=1e-10)
        # Explicit quaternion omitted: preserve Builder orientation.
        support=w['profile']['objects'][w['task']['reset']['support_object']]
        xyz=np.array(support['T_workspace'])[:3,3];xyz[2]+=support['dimensions_m'][2]/2+.03
        request['options']['script_scene']['objects']['cube_red']={'pose':{'xyz':xyz.tolist()}}
        reset_scene(env,request,3);np.testing.assert_allclose(env.backend.get_object_pose('cube_red')[:3,3],xyz,atol=1e-10)
        before=env.backend.get_initial_state();request['options']['script_scene']['objects']['cube_red']={'pose':{'xyz':[99,0,0]}}
        with pytest.raises(ValueError,match='support'):reset_scene(env,request,3)
        # Bad reset is rejected and the reset scene is restored, not published.
        assert np.isfinite(env.backend.data.qpos).all()


@pytest.mark.parametrize('objects',[{'cube_red':{'pose':{'xyz':[0,0,float('nan')]}}},{'cube_red':{'pose':{'xyz':[0,0,0],'frame':'world'}}},{'cup_a':{'pose':{'xyz':[0,0,0]}}},{'cube_red':{'pose':{'xyz':[0,0,0]},'reset':'fn'}}])
def test_invalid_scene_contract_rejected_before_engine(workspace,objects):
    from soarm101_lab.simulation.task_context import validate_scene_objects
    with pytest.raises(ValueError):validate_scene_objects(workspace,objects)


def test_terms_static_identity_and_per_episode_state(workspace,tmp_path):
    w=workspace;objects={'cube_red':{}}
    annotation={'events':{'moved':{'factory':'factory'}},'signals':['moved'],'manual_event':'moved','accept':'accept'}
    marker=tmp_path/'executed'
    path=source_script(tmp_path,w,objects,{'annotation':annotation},f'''
from pathlib import Path
Path({str(marker)!r}).touch()
class Counter:
    def reset(self,ctx):self.n=0
    def __call__(self,ctx):self.n+=1;return self.n>1
def factory():return Counter()
def accept(ctx):return True
''')
    _,_,_=read_followup(path);assert not marker.exists()
    scene=configured(w,active_object_ids=['cube_red'],scene_objects=['cube_red'],scene_only=True,object_configuration=objects)['scene_configuration']
    reader=SimpleNamespace(args={},provenance={});contract=followup_context(reader,w,scene,{'backend':'mujoco','followup_python':str(path)},stage='annotate')
    runtime=load_followup(contract);assert marker.exists()
    from soarm101_lab.workflow.adapters import MujocoAdapter
    with scene_env(w,objects) as env:
        a=MujocoAdapter(env.backend,30,tmp_path,1,threading.Event());a.bind_followup(runtime);a.after_reset()
        term=a.processing_events['moved'];ctx=a.processing_context
        assert not term(ctx) and term(ctx);a.after_reset();assert a.processing_events['moved'] is not term and not a.processing_events['moved'](ctx)
        assert a.tracker is None and a.check_success()
        # Sharing the immutable runtime declaration does not share factory state.
        other=MujocoAdapter(env.backend,30,tmp_path,2,threading.Event())
        other.bind_followup(runtime);other.after_reset()
        assert other.processing_events['moved'] is not a.processing_events['moved']
        assert not other.processing_events['moved'](other.processing_context)
        def broken(ctx):raise RuntimeError('term failed')
        with pytest.raises(RuntimeError,match='term failed'):
            a.term_boolean(broken,a.state())
        assert a.processing_context._observation is None
        other.close();a.buffer.abort()
    path.write_text(path.read_text()+'# changed')
    with pytest.raises(ValueError,match='changed'):load_followup(contract)


def test_mimic_reference_config_and_unsupported_contracts(workspace):
    from soarm101_lab.workflow.mimic import configuration,subtask_configuration
    specs=[dict(name='pick',object='cube_red',eef='tool0',end_event='grasp',offset=[0,10]),dict(name='place',object='cup_a',eef='tool0',end_event=None)]
    old=configuration('cube_red','cup_a',1047);new=subtask_configuration(specs,1047)
    assert [vars(s) for s in old.subtask_configs['tool0']]==[vars(s) for s in new.subtask_configs['tool0']]
    config=dict(scene={'objects':{'cube_red':{},'cup_a':{}}},annotation={'events':{'grasp':'event'},'signals':['grasp'],'accept':'accept'},mimic={'subtasks':specs,'accept':'accept'})
    validate_processing(config,workspace)
    for key,value in [('eef','other_tool'),('end_event','missing'),('object','absent')]:
        broken=copy.deepcopy(config);broken['mimic']['subtasks'][0][key]=value
        with pytest.raises(ValueError):validate_processing(broken,workspace)


def test_empty_annotation_does_not_reclassify_source():
    from soarm101_lab.workflow.dataset import annotated_group
    class Group(dict):attrs={'annotation':'{}'}
    assert not annotated_group(Group())
    g=Group();g.attrs={'annotation':'{"schema":"so101.event-annotation/1","events":{},"signals":[]}'}
    assert annotated_group(g)


def test_failure_term_latches_and_does_not_advance_physics(workspace,tmp_path):
    from soarm101_lab.workflow.adapters import MujocoAdapter
    with scene_env(workspace,{'cube_red':{}}) as env:
        a=MujocoAdapter(env.backend,30,tmp_path,1047,threading.Event())
        from soarm101_lab.simulation.task_context import TaskContext
        a.processing_context=TaskContext(env.backend,adapter=a)
        a.processing_runtime={'contract':{'declaration':{}},'accept':lambda ctx:True,'failure':lambda ctx:'custom_failure'};a.processing_failure=None
        before=env.backend.data.qpos.copy();a._check_term_failure()
        assert not a.check_success() and a.failure_category()=='custom_failure'
        np.testing.assert_array_equal(before,env.backend.data.qpos)
        a.processing_runtime['failure']=lambda ctx:None
        assert not a.check_success()  # latched for the episode
        a.buffer.abort()


def test_scene_only_source_discard_interrupt_cleanup(workspace,tmp_path,monkeypatch):
    from test_source_recording import source_fixture
    from soarm101_lab.workflow.dataset import EpisodeDataset,validate_dataset
    env=scene_env(workspace,{'cube_red':{}})
    worker,closed=source_fixture(env,monkeypatch,True)
    def forbidden():raise AssertionError('Scene Source must not evaluate Task success')
    monkeypatch.setattr(env.backend,'is_success',forbidden)
    class Bridge:
        pending=set();commands=iter([None,'discard',None,'save',None,'interrupt'])
        def poll(self):
            command=next(self.commands)
            if command=='interrupt':raise KeyboardInterrupt()
            if command:self.pending.add(command)
    req={'options':dict(port='virtual',instruction='',episodes=2,episode_time_s=10,seed=1047,scene_only=True,script_scene={'objects':{'cube_red':{}}}),'_scene_resets':{}}
    result=worker.source(req,tmp_path/'recorded',lambda *a:None,Bridge())
    assert closed==[True] and env.backend.closed and result['stopped'] and result['episode_count']==1
    r=EpisodeDataset(result['dataset']);assert validate_dataset(r)['episodes']==1
    assert r.args['task_definitions']==[] and r.args['grasp_object'] is None
    assert 'success' not in r.episode_attributes('demo_0')
    assert not list((tmp_path/'recorded').glob('.video-*'))


def test_manual_annotation_uses_declared_event_name(tmp_path,monkeypatch):
    from test_playback import adapter_fixture,drive
    from soarm101_lab.workflow import annotation
    # No engine is mocked as real here: this checks manual signal ownership only.
    values={'actions':np.zeros((5,7)), 'joint_targets':np.zeros((5,6)),
        'obs':{'policy':{'joint_pos':np.zeros((5,6))},'datagen_info':{'object_pose':{'cube':np.repeat(np.eye(4)[None],5,0)},'target_eef_pose':{'tool0':np.repeat(np.eye(4)[None],5,0)}}},
        'initial_state':{'articulation':{'robot':{'joint_position':np.zeros((1,6)),'root_pose':np.array([[0,0,0,1,0,0,0]])}}}}
    a,_=adapter_fixture(values);a.annotation_auto=False;a.tracker=None;a.place_success=lambda:False;a.check_success=lambda:True;a.failure_category=lambda:'task_condition_not_met'
    a.record_frame=lambda *args:None;a.event_boundaries={'arrived':1};a.processing_events={}
    a.processing_runtime={'contract':{'declaration':{'annotation':{'manual_event':'arrived'}}}}
    a.buffer=SimpleNamespace(final_events=[],finish=lambda attrs,success:(copy.deepcopy({'obs':{'events':{},'datagen_info':{'subtask_term_signals':{}}}}),{**attrs,'annotation':'{"schema":"so101.event-annotation/1","events":{"arrived":1}}'},{}))
    bridge=SimpleNamespace(callbacks={},details={})
    bridge.before_step=lambda:bridge.callbacks['mark']() if a.annotation_position[0]==2 else None
    row,prepared=drive(annotation.annotate_episode(a,SimpleNamespace(backend='mujoco'), 'demo_0',values,{},bridge),a.step)
    assert prepared and row['manual_event']=='arrived' and row['manual_boundary']==2
    np.testing.assert_array_equal(prepared[0]['obs']['datagen_info']['subtask_term_signals']['arrived'][:,0],[0,0,1,1,1])


def test_resume_identity_allows_run_options_but_not_reset_code(workspace,tmp_path):
    path=source_script(tmp_path,workspace,{'cube_red':{'reset':'custom'}},code='def custom(ctx,name,rng): return ctx.object_pose(name)\n')
    config,first,_=read_script(path,'source')
    config['options'].update(episodes=9,resume_data='/explicit/previous.hdf5')
    path.write_text('SCRIPT = '+repr(config)+'\ndef custom(ctx,name,rng): return ctx.object_pose(name)\n')
    _,changed,_=read_script(path,'source')
    assert first['sha256']!=changed['sha256'] and first['source_code_sha256']==changed['source_code_sha256']
    path.write_text(path.read_text().replace('return ctx.object_pose(name)','return rng.normal(size=(4,4))'))
    assert read_script(path,'source')[1]['source_code_sha256']!=first['source_code_sha256']


def test_builder_without_task_catalog_can_record_scene(exported,tmp_path):
    import yaml
    path=exported/'scene.yaml';scene=yaml.safe_load(path.read_text());scene.pop('workflow')
    for obj in scene['objects']:
        if obj['role']=='manipulable_object':obj['physics']={'mass_kg':{'value':.02,'provenance':'estimated','evidence':[]}}
    path.write_text(yaml.safe_dump(scene))
    path=exported/'scene_manifest.yaml';manifest=yaml.safe_load(path.read_text());manifest['files']['scene.yaml']=package.sha(exported/'scene.yaml');path.write_text(yaml.safe_dump(manifest))
    w=package.load(builder.import_scene(exported,registry=tmp_path/'workspaces'));assert w['task']['mode']=='geometry_review'
    script=generate_template(w,tmp_path/'source.py',script_stage='source')
    _,_,opts=script_inputs(script,'source')
    from soarm101_lab.workflow.tasks import bind_scene_options
    opts=bind_scene_options('source',{},w,opts)
    assert opts['scene_only'] and opts['tasks']==[]
    with scene_env(w,{'cube_red':{}}) as env:
        assert env.backend.metadata['pick'] is None and env.backend.is_success() is None


def test_grasp_term_can_bind_without_placement_target(workspace,tmp_path):
    from soarm101_lab.workflow.adapters import MujocoAdapter
    annotation={'diagnostics':{'object_id':'cube_red'},'events':{'lifted':'event'},'signals':['lifted'],'accept':'event'}
    path=source_script(tmp_path,workspace,{'cube_red':{}},{'annotation':annotation},'from soarm101_lab.task_terms.diagnostics import grasp_completed as event\n')
    declaration,identity,_=read_followup(path)
    runtime=load_followup({'declaration':declaration,'python':identity,'schema':'so101.scene-terms/1'})
    with scene_env(workspace,{'cube_red':{}}) as env:
        a=MujocoAdapter(env.backend,30,tmp_path,1,threading.Event());a.bind_followup(runtime);a.after_reset()
        assert a.grasp_object=='cube_red' and a.place_target is None and a.tracker is not None and not a.check_success()
        a.tracker.completed=True;assert a.check_success() and a.place_success() is None
        a.buffer.abort()


def test_missing_custom_signal_cannot_publish_annotated_dataset(episode,tmp_path):
    import json
    from soarm101_lab.data.episodes import DatasetWriter
    values,attrs,videos,args,contract=copy.deepcopy(episode)
    n=len(values['actions']);values['obs']['datagen_info']['subtask_term_signals']={'arrived':np.zeros((n,1),np.uint8)}
    attrs['annotation']=json.dumps(dict(schema='so101.event-annotation/1',frame_count=n,events={},signals=['arrived']))
    target=tmp_path/'invalid.hdf5';writer=DatasetWriter(target,args,contract)
    try:
        with pytest.raises(ValueError,match='boundary missing'):writer.commit(values,attrs,videos)
    finally:writer.close(False)
    assert not target.exists()
