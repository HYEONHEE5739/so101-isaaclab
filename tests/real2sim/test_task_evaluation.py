"""Task contracts reuse existing placement/grasp semantics; no physical thresholds copied."""
import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from test_builder_workspace import exported,fruit_export
from soarm101_lab.real2sim.workspaces import builder,package
from soarm101_lab.real2sim.workspaces.tasks import catalog,configured,generate_template,validate_task_scene,pick_place_contracts
from soarm101_lab.simulation.task_context import TaskContext
from soarm101_lab.workflow.worker_request import validate_task_request,load_task_python


def test_planning_never_executes_task_and_worker_hashes_exact_bytes(tmp_path):
    path=tmp_path/'task.py';marker=tmp_path/'executed'
    source=f"from pathlib import Path\nPath({str(marker)!r}).touch()\nTASK_ID='new'\nINSTRUCTION='test'\nOBJECT_IDS=()\nCAMERA_IDS=()\ndef setup(ctx):pass\ndef observe(ctx):return {{}}\ndef success(ctx):return False\n"
    path.write_text(source)
    opts={'task_python':str(path),'task_definition':{'explicit':True}}
    validate_task_request('sim_eval','mujoco',opts);assert not marker.exists()
    task,identity=load_task_python(opts);assert marker.exists()
    import hashlib
    assert identity['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert task['TASK_ID']=='new'
    for stage,backend,extra in [('annotate','mujoco',{}),('sim_eval','isaac',{}),('sim_eval','mujoco',{'num_envs':2}),('sim_eval','mujoco',{'task_definition':None})]:
        with pytest.raises(ValueError):validate_task_request(stage,backend,{**opts,**extra})
    with pytest.raises(ValueError,match='identity'):load_task_python({**opts,'task_sha256':'0'*64})


@pytest.mark.parametrize('fixture_name',['exported','fruit_export'])
def test_pick_place_existing_predicate_and_boundary_states(request,fixture_name,tmp_path):
    exported=request.getfixturevalue(fixture_name)
    root=builder.import_scene(exported,registry=tmp_path/'workspaces');w=package.load(root);d=catalog(w)[0]
    path=generate_template(w,tmp_path/'pick_place.py',d);task=runpy.run_path(str(path))
    assert task['TASK_CONTRACTS']==pick_place_contracts()
    scene=configured(w,d);validate_task_scene(task,scene)
    with pytest.raises(ValueError,match='Scene'):
        validate_task_scene({**task,'OBJECT_IDS':['missing']},scene)
    with pytest.raises(ValueError,match='contracts'):
        validate_task_scene({**task,'TASK_CONTRACTS':{'evaluation':'invented'}},scene)
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    with MujocoBackend(w,task_definition=d,render_backend='none') as env:
        env.reset(seed=1047);ctx=TaskContext(env);ctx.validate_bindings(task);task['setup'](ctx)
        original=env.get_object_pose(d['pick_object_id']);target=env.get_object_pose(d['target_object_id'])
        obj=w['profile']['objects'][d['pick_object_id']]
        cup=w['profile']['objects'][w['task']['targets'][d['target_object_id']]]
        placed=target.copy()
        if d['success_condition']['relation']=='inside_region_and_settled':
            lo,hi=np.asarray(d['success_condition']['bounds_m']);offset=(lo+hi)/2
            placed[:3,3]+=target[:3,:3]@offset
            edge=placed.copy();edge[:3,3]+=target[:3,:3]@np.array([hi[0]-offset[0],0,0])
        else:
            placed[:3,3]+=target[:3,:3]@np.array([0,0,-cup['dimensions_m'][2]/2+cup['wall_m']+obj['dimensions_m'][2]/2+.001])
            edge=placed.copy();edge[:3,3]+=target[:3,:3]@np.array([cup['dimensions_m'][0]/2,0,0])
        outcomes=[]
        for pose,speed in [(original,0),(edge,0),(placed,1),(placed,0)]:
            env.set_object_pose(d['pick_object_id'],pose)
            _,index=env.object_indices[d['pick_object_id']];env.data.qvel[index]=speed
            env.mj.mj_forward(env.model,env.data)
            before=env.success_details();q=env.data.qpos.copy();v=env.data.qvel.copy()
            assert ctx.placement_details(d)==before
            assert ctx.placement(d['pick_object_id'],d['target_object_id'])==before
            assert ctx.placement(d['pick_object_id'],d['target_object_id'],d['success_condition'])==before
            assert task['success'](ctx)==env.is_success()
            np.testing.assert_array_equal(q,env.data.qpos);np.testing.assert_array_equal(v,env.data.qvel)
            outcomes.append(bool(before['success']))
        assert outcomes==[False,False,False,True]


def test_existing_release_drop_and_mimic_contracts_are_separate():
    from types import SimpleNamespace
    from soarm101_lab.workflow.adapters import SimulationAdapter
    from soarm101_lab.workflow.mimic import configuration
    from soarm101_lab.real2sim.workspaces.grasp import HOLD_SECONDS
    a=SimpleNamespace(tracker=SimpleNamespace(completed=True),grasp_reference=None,grasp_command=None,
                      previous=np.zeros(6),release_commanded=False,loss_steps=0,drop_step=None,step_index=0,hz=20)
    SimulationAdapter._update_failure_evidence(a,np.zeros(3),True,True)
    for i in range(int(HOLD_SECONDS*a.hz)):
        a.step_index=i+1;SimulationAdapter._update_failure_evidence(a,np.ones(3),False,False)
    assert a.drop_step==int(HOLD_SECONDS*a.hz) and a.tracker.completed
    b=copy.deepcopy(a);b.drop_step=None;b.loss_steps=0;b.previous[5]=1
    SimulationAdapter._update_failure_evidence(b,np.ones(3),False,False)
    assert b.release_commanded and b.drop_step is None and b.loss_steps==0
    cfg=configuration('pick-id','target-id',1047)
    steps=cfg.subtask_configs['tool0']
    assert [s.object_ref for s in steps]==['pick-id','target-id']
    assert [s.subtask_term_signal for s in steps]==['grasp',None]
    assert [s.subtask_term_offset_range for s in steps]==[(0,10),(0,0)]
    for s in steps:
        assert s.selection_strategy=='nearest_neighbor_object' and s.selection_strategy_kwargs=={'nn_k':3}
        assert s.num_interpolation_steps==5 and s.num_fixed_steps==0 and s.action_noise==0
