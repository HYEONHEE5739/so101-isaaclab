"""Script contracts use literal plans and the existing MuJoCo engine/terms."""
import ast
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from test_builder_workspace import exported,fruit_export
from soarm101_lab.real2sim.workspaces import builder,package
from soarm101_lab.real2sim.workspaces.tasks import catalog,generate_template
from soarm101_lab.workflow.task_scripts import read_script,script_inputs,load_execution_script,resolve_script


@pytest.mark.parametrize('fixture_name',['exported','fruit_export'])
def test_metadata_scripts_and_existing_placement(request,fixture_name,tmp_path):
    root=builder.import_scene(request.getfixturevalue(fixture_name),registry=tmp_path/'workspaces')
    w=package.load(root);d=catalog(w)[0]
    files={}
    for stage in ('source','sim_eval'):
        path=generate_template(w,tmp_path/(stage+'.py'),d,script_stage=stage,checkpoint='/explicit/checkpoint')
        config,identity,source=read_script(path,stage);ast.parse(source);files[stage]=path
        assert list(config['scene']['objects'])==w['task']['dynamic']+list(w['task']['targets'])
        assert 'bindings' not in config and 'tasks' not in config['options']
        assert 'randomization_spec' not in source.decode()
        with pytest.raises(FileExistsError):generate_template(w,path,d,script_stage=stage)
    _,workspace,opts=script_inputs(files['source'],'source')
    req={'stage':'source','workspace':workspace,'options':opts};load_execution_script(req)
    assert req['_script_evaluations']==[]  # Source does not require a success hook.
    _,workspace,opts=script_inputs(files['sim_eval'],'sim_eval')
    req={'stage':'rollout','workspace':workspace,'options':opts};load_execution_script(req)
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    from soarm101_lab.simulation.task_context import TaskContext
    with MujocoBackend(w,task_definition=d,render_backend='none') as backend:
        backend.reset(seed=1047);ctx=TaskContext(backend);ctx.validate_bindings(req['_script_bindings'])
        before=(backend.data.qpos.copy(),backend.data.qvel.copy())
        details=ctx.placement(d['pick_object_id'],d['target_object_id'],d['success_condition'])
        assert details==backend.success_details()
        assert req['_script_evaluations'][0]['success_fn'](ctx)==backend.is_success()
        np.testing.assert_array_equal(before[0],backend.data.qpos);np.testing.assert_array_equal(before[1],backend.data.qvel)
        with pytest.raises(ValueError):ctx.placement('missing',d['target_object_id'],d['success_condition'])


def test_static_plan_hash_and_duplicate_ids(tmp_path):
    path=Path('examples/tasks/eval_pick_place.py');config,_,source=read_script(path,'sim_eval')
    marker=tmp_path/'executed';local=tmp_path/'task.py'
    source=source.replace(b'../../outputs/',str(Path('outputs').resolve()).encode()+b'/')
    local.write_bytes(source+f"\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n".encode())
    _,workspace,opts=script_inputs(local,'sim_eval');assert not marker.exists()
    req={'stage':'rollout','workspace':workspace,'options':opts};load_execution_script(req);assert marker.exists()
    local.write_bytes(source+b'\n# modified\n')
    with pytest.raises(ValueError,match='changed'):load_execution_script(req)
    config['evaluations']*=2;local.write_text('SCRIPT = '+repr(config))
    with pytest.raises(ValueError,match='Duplicate'):read_script(local,'sim_eval')
    config['evaluations']=config['evaluations'][:1];config['options']['backend']='isaac';local.write_text('SCRIPT = '+repr(config))
    with pytest.raises(ValueError,match='MuJoCo'):read_script(local,'sim_eval')


def test_script_preflight_rejects_missing_bindings_and_reset(exported,tmp_path):
    from soarm101_lab.workflow.task_scripts import validate_script_scenes
    root=builder.import_scene(exported,registry=tmp_path/'workspaces');w=package.load(root);d=catalog(w)[0]
    path=generate_template(w,tmp_path/'eval.py',d,script_stage='sim_eval',checkpoint='/explicit/policy')
    _,_,opts=script_inputs(path,'sim_eval');opts=resolve_script(w,opts);validate_script_scenes(w,opts)
    broken=copy.deepcopy(opts);broken['script_evaluations'][0]['objects']=['missing']
    with pytest.raises(ValueError,match='absent'):validate_script_scenes(w,broken)
    broken=copy.deepcopy(opts);broken['script_evaluations'][0]['reset']={'invented':True}
    with pytest.raises(ValueError,match='reset'):validate_script_scenes(w,broken)
    # Native reset variant accepted without changing the immutable Workspace.
    original=copy.deepcopy(w['task']['reset'])
    opts['script_evaluations'][0]['reset']={**original,'mode':'uniform','yaw_rad':[0.,0.],'xy_jitter_m':0.}
    validate_script_scenes(w,opts);assert w['task']['reset']==original


def test_compact_scene_selection_and_explicit_conditions(fruit_export,tmp_path):
    """A non-catalog pair compiles through the real backend, without task-ID dispatch."""
    from soarm101_lab.real2sim.workspaces.tasks import task_binding,configured,placement_condition
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    from soarm101_lab.simulation.task_context import TaskContext
    root=builder.import_scene(fruit_export,registry=tmp_path/'workspaces');w=package.load(root)
    # Fixture is loaded/validated first; this copy models metadata with one published pair.
    w=copy.deepcopy(w);w['task']['tasks']=[dict(id='apple_only',pick='apple',place='basket',
        instruction='Place apple',success=w['task']['success'])]
    with pytest.raises(ValueError,match='unique'):placement_condition(w,'banana','basket')
    explicit=task_binding(w,'banana','basket',w['task']['success'])
    assert explicit['schema']=='so101.task-binding/1'
    scene=configured(w,explicit,['banana'],'mujoco',scene_objects=['banana','basket'])
    assert scene['task']['dynamic']==['banana'] and 'apple' not in scene['profile']['objects']
    with MujocoBackend(scene,task_definition=explicit,active_object_ids=['banana'],render_backend='none',full_workspace=True) as backend:
        backend.reset(seed=1047);ctx=TaskContext(backend)
        ctx.require(objects=['banana','basket'])
        with pytest.raises(ValueError,match='absent'):ctx.object_pose('apple')
        assert ctx.placement('banana','basket',w['task']['success'])==backend.success_details()
    with pytest.raises(ValueError,match='cup geometry'):
        placement_condition(w,'banana','basket',{'relation':'contained_and_settled','max_speed_m_s':.03})
    ambiguous=copy.deepcopy(w)
    ambiguous['task']['tasks'].append({**ambiguous['task']['tasks'][0],'id':'other',
        'success':{**w['task']['success'],'max_speed_m_s':.04}})
    with pytest.raises(ValueError,match='unique'):placement_condition(ambiguous,'apple','basket')


def test_compact_preflight_conflicts_and_scene_overrides(exported,tmp_path):
    from pprint import pformat
    from soarm101_lab.workflow.task_scripts import validate_script_scenes
    root=builder.import_scene(exported,registry=tmp_path/'workspaces');w=package.load(root);d=catalog(w)[0]
    path=generate_template(w,tmp_path/'initial.py',d,script_stage='sim_eval',checkpoint='/policy')
    config,_,source=read_script(path,'sim_eval')
    body=source.decode().split('from functools import partial')[1]
    def prepare(cfg):
        local=tmp_path/'edit.py';local.write_text('SCRIPT = '+pformat(cfg)+'\nfrom functools import partial'+body)
        _,_,opts=script_inputs(local,'sim_eval');return resolve_script(w,opts)
    cfg=copy.deepcopy(config);cfg['options']['tasks']=[d]
    with pytest.raises(ValueError,match='conflict'):prepare(cfg)
    cfg=copy.deepcopy(config);cfg['scene']['objects'].append('missing')
    with pytest.raises(ValueError,match='Object IDs'):prepare(cfg)
    cfg=copy.deepcopy(config);cfg['scene']['objects'].remove(d['target_object_id'])
    with pytest.raises(ValueError,match='target absent'):prepare(cfg)
    cfg=copy.deepcopy(config);cfg['evaluations']*=2
    with pytest.raises(ValueError,match='Duplicate'):prepare(cfg)
    cfg=copy.deepcopy(config);cfg['evaluations'][0]['reset']={'invented':1}
    with pytest.raises(ValueError,match='reset'):prepare(cfg)
    cfg=copy.deepcopy(config);cfg['evaluations'].append({**cfg['evaluations'][0],'id':'another_user_id',
        'scene':{'objects':config['scene']['objects']}})
    opts=prepare(cfg);validate_script_scenes(w,opts)
    assert [r['id'] for r in opts['script_evaluations']]==[d['task_id'],'another_user_id']
    assert opts['script_scenes'][0]==opts['script_scenes'][1]


def test_worker_sibling_term_and_source_identity(exported,tmp_path):
    from pprint import pformat
    root=builder.import_scene(exported,registry=tmp_path/'workspaces');w=package.load(root);d=catalog(w)[0]
    source=generate_template(w,tmp_path/'source.py',d,script_stage='source')
    cfg,_,_=read_script(source,'source');cfg['source']['id']='my_demo';cfg['source']['instruction']='My demonstration'
    # Explicit legacy input remains supported; new generator emits Scene-only Sources.
    cfg['scene']['objects']=list(cfg['scene']['objects']);cfg.pop('annotation');cfg.pop('mimic')
    cfg['source'].update(pick_object=d['pick_object_id'],target_object=d['target_object_id'])
    cfg['source']['condition']={**d['success_condition'],'max_speed_m_s':.02}
    source.write_text('SCRIPT = '+pformat(cfg))
    _,wp,opts=script_inputs(source,'source');req={'stage':'source','workspace':wp,'options':opts}
    load_execution_script(req)
    assert not req['_script_evaluations'] and req['_source_reference']['id']=='my_demo'
    assert req['options']['task_definition']['language_instruction']=='My demonstration'
    assert req['options']['task_definition']['schema']=='so101.task-binding/1'
    from soarm101_lab.real2sim.workspaces.tasks import validate
    validate(req['options']['task_definition'],w)
    local=tmp_path/'user_terms.py';local.write_text('def custom_success(ctx): return ctx.object_pose("'+d['pick_object_id']+'")[0,3] > 0\n')
    cfg.pop('source');cfg.update(stage='sim_eval',checkpoint='/policy',evaluations=[
        dict(id='custom',instruction='Cross x',success='custom_success',episodes=1)])
    cfg['scene']['diagnostics']={'object_id':d['pick_object_id'],'target_id':d['target_object_id']}
    task=tmp_path/'eval.py';task.write_text('SCRIPT = '+pformat(cfg)+'\nfrom user_terms import custom_success\n')
    _,wp,opts=script_inputs(task,'sim_eval');req={'stage':'rollout','workspace':wp,'options':opts}
    load_execution_script(req);assert callable(req['_script_evaluations'][0]['success_fn'])
    task.write_text(task.read_text()+'\n# edited')
    with pytest.raises(ValueError,match='changed after planning'):load_execution_script(req)
