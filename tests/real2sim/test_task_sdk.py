"""Metadata generation and a borrowed native Backend; no robot or model training."""
import ast
import copy
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_builder_workspace import exported, fruit_export
from soarm101_lab.real2sim.workspaces import builder, package
from soarm101_lab.real2sim.workspaces.tasks import catalog, generate_template
from soarm101_lab.simulation.task_context import TaskContext


@pytest.mark.parametrize('fixture_name',['exported','fruit_export'])
def test_builder_template_runtime_and_trace(request,fixture_name,tmp_path):
    export=request.getfixturevalue(fixture_name)
    root=builder.import_scene(export,registry=tmp_path/'workspaces')
    w=package.load(root);before=copy.deepcopy(w)
    definition=catalog(w)[0]
    path=generate_template(w,tmp_path/'my_task.py',definition)
    ast.parse(path.read_text());task=runpy.run_path(str(path))
    assert task['OBJECT_IDS']==(definition['pick_object_id'],definition['target_object_id'])
    assert set(task['CAMERA_IDS'])==set(w['profile']['cameras'])
    assert w==before
    with pytest.raises(FileExistsError):generate_template(w,path,definition)
    with pytest.raises(ValueError,match='immutable'):generate_template(w,root/'my_task.py')
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    with MujocoBackend(w,task_definition=definition,render_backend='none') as env:
        ctx=TaskContext(env);assert ctx.workspace is env.workspace
        env.reset(seed=1047);saved=env.get_initial_state()
        baseline=[]
        for _ in range(8):
            state=env.step(env.home);baseline.append((env.data.qpos.copy(),env.data.qvel.copy(),env.is_success()))
        env.restore_state(saved);task['setup'](ctx)
        for q,v,success in baseline:
            poses=task['observe'](ctx)
            for name,pose in poses.items():
                np.testing.assert_array_equal(pose,env.get_object_pose(name))
                pose[:]=0
                assert np.any(ctx.object_pose(name))
            env.step(env.home)
            np.testing.assert_array_equal(env.data.qpos,q);np.testing.assert_array_equal(env.data.qvel,v)
            assert task['success'](ctx)==success
        with pytest.raises(ValueError,match='absent'):ctx.object_pose('not_in_scene')
        with pytest.raises(ValueError,match='camera'):ctx.require(cameras=['missing'])
        bad=copy.deepcopy(task['TASK_DEFINITION']);bad['pick_object_id']='missing'
        with pytest.raises(ValueError):generate_template(w,tmp_path/'bad.py',bad)
        assert env.renderer is None
    with pytest.raises(RuntimeError):ctx.object_pose(definition['pick_object_id'])


def test_generic_template_never_invents_success(tmp_path):
    w={'root':tmp_path/'workspace','profile':{'robot_asset':{'urdf':'actual.urdf'},'cameras':{'front':{}}},
       'task':{'dynamic':['item with spaces'], 'targets':{'destination': 'visual'}}}
    path=generate_template(w,tmp_path/'generic.py');task=runpy.run_path(str(path))
    assert task['OBJECT_IDS']==('item with spaces','destination')
    with pytest.raises(NotImplementedError):task['success'](None)


def test_import_has_no_engine_ui_workflow_dependency():
    code="from soarm101_lab.simulation.task_context import TaskContext;import sys;assert not any(n.split('.')[0] in ('torch','mujoco','isaaclab','omni','PyQt6') or n.startswith('soarm101_lab.workflow') for n in sys.modules)"
    subprocess.run([sys.executable,'-c',code],check=True)


def test_module_contract_rejected_before_hooks_execute():
    from soarm101_lab.real2sim.workspaces.tasks import validate_task_module
    def forbidden(ctx):raise AssertionError('Validation must not execute hooks')
    task=dict(TASK_ID='custom',INSTRUCTION='User-defined task',OBJECT_IDS=['item'],CAMERA_IDS=[],
              setup=forbidden,observe=forbidden,success=forbidden)
    validate_task_module(task)
    for change in ({'TASK_ID':''},{'INSTRUCTION':None},{'OBJECT_IDS':['item','item']},
                   {'CAMERA_IDS':['']},{'success':None},{'success':lambda:False},{'TASK_DEFINITION':'invalid'}):
        with pytest.raises(ValueError):validate_task_module({**task,**change})


def test_user_task_worker_independent_and_cleanup(exported,tmp_path,monkeypatch):
    """Execute the production CLI on a real engine; pose injection is test-only."""
    import json
    root=builder.import_scene(exported,registry=tmp_path/'workspaces')
    w=package.load(root)
    name=w['task']['dynamic'][0]
    repo=Path(__file__).resolve().parents[2]
    source=(repo/'docs/architecture/task_x_threshold.py').read_text().replace("('cube_red',)",repr((name,)))
    task_path=tmp_path/'user.py';task_path.write_text(source)
    assert 'block_crosses_workspace_x' not in {d['task_id'] for d in catalog(w)}
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    original_step=MujocoBackend.step
    instances=[]
    original_init=MujocoBackend.__init__
    def init(self,*a,**kw):
        original_init(self,*a,**kw);instances.append(self)
    def step(self,target):
        state=original_step(self,target)
        pose=self.get_object_pose(name);pose[0,3]=0.6
        self.set_object_pose(name,pose)
        assert not original_success(self)
        return state
    original_success=MujocoBackend.is_success
    monkeypatch.setattr(MujocoBackend,'__init__',init)
    monkeypatch.setattr(MujocoBackend,'step',step)
    # Existing get_state invokes the backend predicate internally; force it false
    # to prove the worker's reported task outcome comes from the user hook.
    monkeypatch.setattr(MujocoBackend,'is_success',lambda self: False)
    worker=runpy.run_path(str(repo/'scripts/mujoco/check_env.py'))
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['check_env','--workspace',str(root),'--task-python',str(task_path),
                                 '--output',str(output),'--steps','1','--seed','1047'])
    worker['main']()
    report=json.loads((output/'report.json').read_text())
    assert report['task_python']['initial']['success'] is False
    assert report['task_python']['final']['success'] is True
    assert 'placement_probe' not in report and report['environment_closed']
    assert instances[-1].closed
    # Missing declaration must fail even though setup is a no-op, and close the engine.
    task_path.write_text(source.replace(repr((name,)),"('absent',)"))
    monkeypatch.setattr(sys,'argv',['check_env','--workspace',str(root),'--task-python',str(task_path),
                                 '--output',str(tmp_path/'invalid'),'--steps','1'])
    with pytest.raises(ValueError,match='absent'):worker['main']()
    assert instances[-1].closed
    # Hook errors and interrupts still reach the same worker finally.
    for index,error in enumerate(('RuntimeError','KeyboardInterrupt')):
        task_path.write_text(source.replace('    pass',f'    raise {error}("hook failure")'))
        monkeypatch.setattr(sys,'argv',['check_env','--workspace',str(root),'--task-python',str(task_path),
                                     '--output',str(tmp_path/f'error{index}'),'--steps','1'])
        with pytest.raises((RuntimeError,KeyboardInterrupt)):worker['main']()
        assert instances[-1].closed


def test_runtime_bindings_and_explicit_scene(exported,tmp_path):
    root=builder.import_scene(exported,registry=tmp_path/'workspaces');w=package.load(root)
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    with MujocoBackend(w,render_backend='none') as env:
        ctx=TaskContext(env)
        task=dict(OBJECT_IDS=[env.metadata['pick']],CAMERA_IDS=[env.metadata['primary_camera']])
        q=env.data.qpos.copy();v=env.data.qvel.copy()
        ctx.validate_bindings(task);ctx.object_pose(task['OBJECT_IDS'][0]);ctx.joint_state()
        np.testing.assert_array_equal(env.data.qpos,q);np.testing.assert_array_equal(env.data.qvel,v)
        with pytest.raises(ValueError,match='identity'):
            ctx.validate_bindings({**task,'ROBOT_ASSET_SHA256':'0'*64})
        with pytest.raises(ValueError,match='asset reference'):
            ctx.validate_bindings({**task,'ROBOT_ASSET':{}})
        with pytest.raises(ValueError,match='explicitly'):
            ctx.validate_bindings({**task,'TASK_DEFINITION':{'task_id':'unconfigured'}})
        excluded=set(w['task']['dynamic'])-set(ctx.object_ids)
        for name in excluded:
            with pytest.raises(ValueError,match='absent'):ctx.validate_bindings({**task,'OBJECT_IDS':[name]})
        with pytest.raises(ValueError,match='camera'):
            ctx.validate_bindings({**task,'CAMERA_IDS':['missing']})


def test_workspace_path_and_loaded_scene_preserve_compilation(exported,tmp_path):
    root=builder.import_scene(exported,registry=tmp_path/'workspaces');w=package.load(root)
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    # Previous CLI passed a path; current CLI shares the validated Workspace dict.
    with MujocoBackend(root,render_backend='none') as before, MujocoBackend(w,render_backend='none') as after:
        assert before.metadata['model_sha256']==after.metadata['model_sha256']
        for name in ('body_pos','body_quat','geom_size','jnt_range','actuator_gainprm'):
            np.testing.assert_array_equal(getattr(before.model,name),getattr(after.model,name))
        before.reset(seed=1047);after.reset(seed=1047)
        for _ in range(8):
            before.step(before.home);after.step(after.home)
            np.testing.assert_array_equal(before.data.qpos,after.data.qpos)
            np.testing.assert_array_equal(before.data.qvel,after.data.qvel)
            assert before.is_success()==after.is_success()
