"""Followup terms reuse the native Adapter and exact recorded Scene semantics."""
from types import SimpleNamespace

import numpy as np
import pytest
from test_builder_workspace import exported
from test_common_mimic import episode
from soarm101_lab.real2sim.workspaces import builder,package
from soarm101_lab.real2sim.workspaces.tasks import configured,catalog
from soarm101_lab.workflow.task_scripts import read_followup,followup_context,load_followup


@pytest.fixture
def context(exported,tmp_path):
    w=package.load(builder.import_scene(exported,registry=tmp_path/'workspaces'))
    scene=configured(w,catalog(w)[0],backend='mujoco')['scene_configuration'];d=scene['task_definition']
    ref={'id':'my_recording','pick_object':d['pick_object_id'],'target_object':d['target_object_id'],
         'instruction':d['language_instruction']}
    r=SimpleNamespace(args={},provenance={'source_script':{'path':'/never/execute/source.py','source_reference':ref}})
    return r,w,scene


def contract_file(tmp_path,scene,**extra):
    d=scene['task_definition'];path=tmp_path/'contract.py'
    config=dict(schema='so101.pick-place-followup/1',pick_object=d['pick_object_id'],
                target_object=d['target_object_id'],mimic='pick_place/1',**extra)
    path.write_text('FOLLOWUP = '+repr(config)+'\n')
    return path


def test_native_and_explicit_contract_never_executes_recorded_paths(context,tmp_path):
    r,w,scene=context;marker=tmp_path/'executed'
    source=tmp_path/'source.py';source.write_text(f'open({str(marker)!r},"w").write("bad")')
    r.provenance['source_script']['path']=str(source)
    native=followup_context(r,w,scene,{})
    assert native['source_reference']['id']=='my_recording' and load_followup(native) is None
    p=contract_file(tmp_path,scene)
    with p.open('a') as f:f.write(f'open({str(marker)!r},"w").write("worker")\n')
    c=followup_context(r,w,scene,{'backend':'mujoco','followup_python':str(p)})
    assert not marker.exists()
    load_followup(c);assert marker.read_text()=='worker'
    p.write_text(p.read_text()+'\n# changed\n')
    with pytest.raises(ValueError,match='changed'):load_followup(c)


@pytest.mark.parametrize('change',['pick','instruction','condition','mimic','identity','backend'])
def test_mismatched_source_or_unsupported_contract_rejected(context,tmp_path,change):
    r,w,scene=context;p=contract_file(tmp_path,scene)
    opts={'backend':'mujoco','followup_python':str(p)}
    ref=r.provenance['source_script']['source_reference']
    if change=='pick':ref['pick_object']='missing'
    if change=='instruction':ref['instruction']='different task'
    if change=='condition':ref['condition']={**scene['task_definition']['success_condition'],'max_speed_m_s':.02}
    if change=='mimic':p.write_text(p.read_text().replace('pick_place/1','approach_grasp_move_release/1'))
    if change=='identity':opts['followup_identity']={'sha256':'changed'}
    if change=='backend':opts['backend']='isaac'
    with pytest.raises(ValueError):followup_context(r,w,scene,opts)


def test_annotation_identity_required_for_mimic_and_resume(context,tmp_path):
    r,w,scene=context;p=contract_file(tmp_path,scene,annotation='accept')
    with p.open('a') as f:f.write('def accept(ctx): return True\n')
    opts={'backend':'mujoco','followup_python':str(p)}
    c=followup_context(r,w,scene,opts);r.args['task_followup']=c
    with pytest.raises(ValueError,match='same explicit'):followup_context(r,w,scene,{})
    assert followup_context(r,w,scene,opts)==c
    r.args['task_followup']={**c,'pick_object':'not_the_recorded_object'}
    with pytest.raises(ValueError,match='metadata differs'):followup_context(r,w,scene,opts)
    r.args['task_followup']=c
    p.write_text(p.read_text()+'# new code\n')
    with pytest.raises(ValueError,match='same explicit'):followup_context(r,w,scene,opts)


def test_terms_refine_native_gate_without_updating_tracker():
    from soarm101_lab.workflow.adapters import SimulationAdapter
    from soarm101_lab.task_terms.annotation import grasp_boundary,place_boundary
    a=SimpleNamespace(place_success=lambda:True,tracker=SimpleNamespace(completed=False))
    calls=[];a.followup_acceptance=(object(),lambda ctx:calls.append(ctx) or True)
    assert not SimulationAdapter.check_success(a) and calls==[]
    a.tracker.completed=True
    assert SimulationAdapter.check_success(a) and calls==[]
    assert SimulationAdapter.accept_followup(a) and len(calls)==1
    a.followup_acceptance=(object(),lambda ctx:False)
    assert SimulationAdapter.check_success(a) and not SimulationAdapter.accept_followup(a)
    a.followup_acceptance=(object(),lambda ctx:{'success':True})
    with pytest.raises(ValueError,match='Boolean'):SimulationAdapter.accept_followup(a)
    a.buffer=SimpleNamespace(rows=[{'grasp':False,'place':False},{'grasp':True,'place':False}])
    ctx=SimpleNamespace(adapter=a)
    assert grasp_boundary(ctx)==1 and place_boundary(ctx) is None


def test_runtime_binding_uses_existing_tracker_and_native_mimic(context,tmp_path):
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    from soarm101_lab.workflow.adapters import MujocoAdapter
    from soarm101_lab.workflow.mimic import configuration
    import threading
    r,w,scene=context;d=scene['task_definition'];p=contract_file(tmp_path,scene,annotation='accept')
    with p.open('a') as f:f.write('def accept(ctx): return bool(ctx.object_pose(FOLLOWUP["pick_object"])[2,3] > 0)\n')
    c=followup_context(r,w,scene,{'backend':'mujoco','followup_python':str(p)})
    runtime=load_followup(c)
    with MujocoBackend(w,task_definition=d,render_backend='none') as b:
        a=MujocoAdapter(b,30,tmp_path,1047,threading.Event());tracker=a.tracker
        before=(b.data.qpos.copy(),b.data.qvel.copy());a.bind_followup(runtime)
        assert a.tracker is tracker
        assert a.followup_acceptance[1](a.followup_acceptance[0]) in (True,False)
        expected=configuration(d['pick_object_id'],d['target_object_id'],1047)
        assert vars(a.cfg.datagen_config)==vars(expected.datagen_config)
        assert [vars(s) for s in a.cfg.subtask_configs['tool0']]==[vars(s) for s in expected.subtask_configs['tool0']]
        np.testing.assert_array_equal(before[0],b.data.qpos);np.testing.assert_array_equal(before[1],b.data.qvel)
        a.close()


def test_datagen_resume_rejects_contract_change_before_copy(episode,tmp_path):
    from soarm101_lab.data.episodes import DatasetWriter
    from soarm101_lab.workflow.datagen import _resume
    values,attrs,videos,args,coordinates=episode
    args={**args,'generation_provenance':{'source_sha256':'original_pool'},'generation_seed':1047,
          'task_followup':{'python':{'sha256':'original_code'}}}
    old=tmp_path/'previous.hdf5'
    with DatasetWriter(old,args,coordinates) as writer:writer.commit(values,attrs,videos)
    with DatasetWriter(tmp_path/'new.hdf5',{**args,'task_followup':{'python':{'sha256':'changed_code'}}},coordinates) as writer:
        with pytest.raises(ValueError,match='Resume followup'):
            _resume(writer,{'resume_data':str(old),'generation_seed':1047})
        assert writer.count==0


@pytest.mark.parametrize('outcome',['reject','error','interrupt'])
def test_custom_adoption_does_not_relabel_native_physics(episode,outcome):
    from test_playback import adapter_fixture,drive
    from soarm101_lab.workflow.annotation import annotate_episode
    values,attrs,*_=episode;a,_=adapter_fixture(values)
    a.tracker=SimpleNamespace(completed=True);a.place_success=lambda:True
    a.check_success=lambda:True;a.failure_category=lambda:'place_failure'
    a.record_frame=lambda *args:None
    a.buffer=SimpleNamespace(finish=lambda *a,**kw:pytest.fail('Rejected contract must not finish/export'))
    def term(ctx):
        if outcome=='error':raise RuntimeError('user term failed')
        if outcome=='interrupt':raise KeyboardInterrupt()
        return False
    a.followup_acceptance=(object(),term)
    reader=SimpleNamespace(backend='mujoco')
    stream=annotate_episode(a,reader,'demo_0',values,attrs)
    if outcome=='interrupt':
        with pytest.raises(KeyboardInterrupt):drive(stream,a.step)
    else:
        row,prepared=drive(stream,a.step)
        assert row['physical_replay_passed'] and not row['annotation_accepted'] and prepared is None
        assert row['category']==('task_condition_not_met' if outcome=='reject' else 'task_contract_error')


def test_direct_isaac_entry_rejects_custom_python_before_any_resources(tmp_path):
    from soarm101_lab.workflow.unified import execute
    from soarm101_lab.workflow.datagen import run_isaac
    request={'stage':'annotate','backend':'isaac','options':{'followup_python':'not_loaded.py'}}
    with pytest.raises(ValueError,match='MuJoCo'):execute(request)
    with pytest.raises(ValueError,match='MuJoCo'):run_isaac(None,request,SimpleNamespace(args={}),tmp_path)
