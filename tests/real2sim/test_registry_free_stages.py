"""All remaining public stages accept Paths; model/physical policies stay native."""
import copy,json,subprocess,sys
from pathlib import Path
import pytest
from test_episode_paths import source,episode,published
from test_lerobot_merge import converted_record,write_video_dataset
from soarm101_lab.workflow.operations import Operations,execute_request
from soarm101_lab.workflow.worker_request import stage_input
from soarm101_lab.utils.artifact_identity import fingerprint


def policy_fixture(root,w,architecture='act'):
    from soarm101_lab.so101_dataset_contract import JOINT_SPACE,write_contract
    from soarm101_lab.real2sim.workspaces.demo import coordinate_contract
    from soarm101_lab.real2sim.workspaces.package import metadata
    root.mkdir(parents=True)
    for name,value in [('config.json',{'type':architecture}),('policy_preprocessor.json',{}),('policy_postprocessor.json',{}),
                       ('workflow_provenance.json',{'workspace':metadata(w),'task_definitions':[w['task_definition']],'backend':'mujoco'})]:
        (root/name).write_text(json.dumps(value))
    (root/'model.safetensors').write_bytes(b'fixture: never loaded by readiness')
    write_contract(root,{**coordinate_contract(w),'action_space':JOINT_SPACE})
    return root


def test_source_datagen_and_conversion_requests_are_paths(source,tmp_path):
    path,w,*_=source;s=Operations(root=tmp_path/'workflow')
    for stage,input_path,opts in [('source',w['root'],{'port':'FAKE'}),('datagen',path,{'max_attempts':1}),('convert',path,{'repo_id':'local/test'})]:
        options=dict(backend='mujoco',storage_name='val_green',**opts)
        run=tmp_path/stage;run.mkdir()
        plan=s.plan(stage,str(input_path),str(w['root']),options,run_dir=run)
        if (run/'request.json').is_file():
            request=json.loads((run/'request.json').read_text())
            assert isinstance(request['input'],str)
            assert 'id' not in request and 'parents' not in request
        assert plan['options']['storage_name']=='val_green'
    assert not (s.root/'artifacts').exists()


def test_policy_paths_require_native_contracts_and_do_not_repair(source,tmp_path):
    _,w,*_=source;path=policy_fixture(tmp_path/'model',w,'smolvla');s=Operations(root=tmp_path/'workflow')
    assert s.readiness('sim_eval',str(path),str(w['root']),{'backend':'mujoco','storage_name':'test'})==('READY','')
    (path/'workflow_provenance.json').unlink();before=fingerprint(path)
    state,reason=s.readiness('sim_eval',str(path),str(w['root']),{'backend':'mujoco','storage_name':'test'})
    assert state=='NOT_READY' and 'provenance missing' in reason
    assert fingerprint(path)==before
    assert not (s.root/'artifacts').exists()


@pytest.mark.parametrize('architecture',['act','smolvla'])
def test_policy_and_hub_plan_worker_requests_are_paths(source,tmp_path,architecture):
    _,w,*_=source;path=policy_fixture(tmp_path/'model',w,architecture);s=Operations(root=tmp_path/'workflow')
    for stage in ('sim_eval','real_eval','hf_policy'):
        run=tmp_path/stage;run.mkdir()
        options=dict(storage_name='test',backend='isaac',repo_id='local/test',private=True,
                     port='FAKE',follower_id='fixture',camera_devices={name:'FAKE' for name in w['profile']['cameras']},enable_motion=True)
        plan=s.plan(stage,str(path),str(w['root']),options,run_dir=run)
        request=json.loads((run/'request.json').read_text())
        assert request['input']==str(path) and 'parents' not in request
        assert plan['options']['tasks']
    options['enable_motion']=False
    assert s.readiness('real_eval',str(path),str(w['root']),options)[0]=='NOT_READY'
    assert not (s.root/'artifacts').exists()


def test_training_merge_and_upload_take_native_dataset_paths(tmp_path,monkeypatch):
    # Real tiny LeRobot files; no Trainer/Hub execution.
    record=converted_record(tmp_path,'a');write_video_dataset(record,.03)
    source_path=record['path'];before=fingerprint(source_path);s=Operations(root=tmp_path/'workflow')
    for architecture in ('act','smolvla'):
        run=tmp_path/architecture;run.mkdir()
        plan=s.plan('train',source_path,'',dict(storage_name=architecture,architecture=architecture),run_dir=run)
        request=json.loads((run/'request.json').read_text())
        assert request['input']==source_path and request['options']['architecture']==architecture
    with pytest.raises(ValueError,match='중복'):
        s.plan('merge',source_path,'',dict(storage_name='merge',merge_datasets=[source_path,source_path]))
    run=tmp_path/'upload';run.mkdir()
    s.plan('hf_dataset',source_path,'',dict(storage_name='upload',repo_id='local/test',private=True),run_dir=run)
    request=json.loads((run/'request.json').read_text());assert request['input']==source_path
    assert fingerprint(source_path)==before and not (s.root/'artifacts').exists()


def test_worker_rereads_native_metadata_before_trainer(tmp_path,monkeypatch):
    record=converted_record(tmp_path,'a');write_video_dataset(record,.03)
    s=Operations(root=tmp_path/'workflow');run=tmp_path/'run';run.mkdir()
    s.plan('train',record['path'],'',dict(storage_name='train',architecture='act'),run_dir=run)
    request=json.loads((run/'request.json').read_text())
    (Path(record['path'])/'so101_coordinates.json').unlink()
    monkeypatch.setattr('soarm101_lab.workflow.policy.train',lambda *a:pytest.fail('Must reject unknown coordinates before Trainer'))
    with pytest.raises(ValueError):execute_request(request)


def test_missing_checkpoint_contract_is_not_an_implicit_import(source,tmp_path):
    _,w,*_=source;path=policy_fixture(tmp_path/'model',w)
    (path/'so101_coordinates.json').unlink();before=fingerprint(path)
    with pytest.raises(ValueError,match='Coordinate'):
        stage_input('sim_eval',str(path))
    assert fingerprint(path)==before


def test_old_registry_commands_are_removed(tmp_path):
    cli=Path(__file__).resolve().parents[2]/'scripts/real2sim/workflow.py'
    result=subprocess.run([sys.executable,str(cli),'register','--path',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==2 and 'invalid choice' in result.stderr


def test_registry_implementation_has_no_production_importers():
    import ast
    root=Path(__file__).resolve().parents[2]
    assert not (root/'source/soarm101_lab/soarm101_lab/workflow/artifacts.py').exists()
    for path in [*root.joinpath('source').rglob('*.py'),*root.joinpath('scripts').rglob('*.py')]:
        tree=ast.parse(path.read_text())
        assert not any(isinstance(n,ast.ImportFrom) and n.module and (n.module=='artifacts' or n.module.endswith('workflow.artifacts')) for n in ast.walk(tree)),path


def test_policy_path_exposes_task_and_instruction_without_registration(source,tmp_path):
    from PyQt6.QtWidgets import QApplication
    from test_episode_path_ui import until
    from soarm101_lab.workflow.ui import WorkflowPanel
    _,w,*_=source;path=policy_fixture(tmp_path/'smolvla',w,'smolvla')
    app=QApplication.instance() or QApplication([])
    panel=WorkflowPanel(lambda *args:pytest.fail('No automatic execution'),root=tmp_path/'workflow')
    try:
        panel.workspace_path=str(w['root'])
        panel.stage.setCurrentIndex(panel.stage.findData('sim_eval'))
        panel.storage_name.setText('evaluation');panel.select_data(str(path))
        until(lambda:panel.run.isEnabled())
        assert panel.task_selector.selection()
        assert panel.task_selector.instruction.text()==w['task_definition']['language_instruction']
        panel.task_selector.instruction.setText('Move the selected object into its target.')
        panel.task_selector.edit_instruction(panel.task_selector.instruction.text())
        assert panel.stage_options()['tasks'][0]['language_instruction']=='Move the selected object into its target.'
        assert not (panel.root/'artifacts').exists()
    finally:panel.timer.stop();panel.close();app.processEvents()
