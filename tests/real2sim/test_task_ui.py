"""Task Python is previewed statically and generated through the existing UI."""
import ast
from pathlib import Path
import pytest
from test_mujoco_workflow_ui import panel
from test_artifact_storage import storage_workspace
from test_episode_path_ui import until


def test_static_preview_never_executes(tmp_path):
    from soarm101_lab.workflow.task_ui import task_file_preview
    marker=tmp_path/'executed'
    task=tmp_path/'task.py'
    task.write_text(f"open({str(marker)!r}, 'w').write('bad')\nTASK_ID='custom'\nINSTRUCTION='x'\ndef success(ctx): return False\n")
    assert task_file_preview(task)['TASK_ID']=='custom'
    assert not marker.exists()
    task.write_text('def success(ctx):\n    raise NotImplementedError("implement")\n')
    with pytest.raises(ValueError,match='not implemented'):task_file_preview(task)


def test_generate_select_and_scene_separation(panel,tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QInputDialog,QFileDialog
    from PyQt6.QtTest import QTest
    from PyQt6.QtCore import Qt
    panel.stage.setCurrentIndex(panel.stage.findData('sim_eval'))
    until(lambda:panel.task_selector.workspace is not None)
    original=panel.task_selector.preset.currentData()
    destination=tmp_path/'generic.py'
    monkeypatch.setattr(QInputDialog,'getItem',lambda *args:(args[3][0],True))
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *args:(str(destination),'Python'))
    QTest.mouseClick(panel.task_create,Qt.MouseButton.LeftButton)
    until(lambda:panel.task_python.text()==str(destination))
    ast.parse(destination.read_text())
    until(lambda:'not implemented' in panel.status.text())
    assert not panel.run.isEnabled()
    assert panel.task_selector.preset.currentData()==original
    assert panel.task_selector.pick.isHidden()
    assert not panel.task_selector.preset.isHidden()
    assert panel.stage_options()['task_python']==str(destination)
    before=destination.read_bytes()
    QTest.mouseClick(panel.task_create,Qt.MouseButton.LeftButton)
    until(lambda:'생성 실패' in panel.task_preview.text())
    assert destination.read_bytes()==before
    panel.task_clear.click()
    assert not panel.task_selector.pick.isHidden()
    assert 'task_python' not in panel.stage_options()
    panel.task_python.setText(str(destination))
    panel.stage.setCurrentIndex(panel.stage.findData('source'))
    assert not panel.task_tools.isHidden()  # Source now exposes the separate execution-script contract.
    assert 'task_python' not in panel.stage_options()


def test_task_request_restrictions_and_tab_browsing(panel,tmp_path,monkeypatch):
    import json
    marker=tmp_path/'executed'
    task=tmp_path/'task.py'
    task.write_text(f"open({str(marker)!r}, 'w').write('bad')\nTASK_ID='custom'\nINSTRUCTION='x'\ndef success(ctx): return False\n")
    panel.stage.setCurrentIndex(panel.stage.findData('sim_eval'))
    until(lambda:panel.task_selector.workspace is not None)
    panel.task_python.setText(str(task))
    from soarm101_lab.workflow.worker_request import validate_task_request
    options=panel.stage_options()
    options['num_envs']=2
    with pytest.raises(ValueError,match='num_envs'):validate_task_request('sim_eval','mujoco',options)
    options['num_envs']=1
    with pytest.raises(ValueError,match='MuJoCo'):validate_task_request('sim_eval','isaac',options)
    with pytest.raises(ValueError,match='MuJoCo'):validate_task_request('annotate','mujoco',options)
    panel.task_recheck.click()
    until(lambda:not panel.status.text().startswith('CHECKING'))
    for stage in ('train','annotate','sim_eval'):
        panel.stage.setCurrentIndex(panel.stage.findData(stage))
    assert not marker.exists()


def test_result_task_identity_and_source(panel,tmp_path,monkeypatch):
    task=tmp_path/'task.py';task.write_text('def success(ctx): return False\n')
    panel.stage.setCurrentIndex(panel.stage.findData('sim_eval'))
    until(lambda:panel.task_selector.workspace is not None)
    panel.task_python.setText(str(task))
    panel.run_worker=lambda fn,done:done(fn(lambda text:None))
    monkeypatch.setattr(panel.service,'run',lambda *args:{'path':str(tmp_path/'report.json'),'details':{'task_python':{'task_id':'custom'},'successes':0,'failures':1,'evaluation_success_contract':'task.success/1','episodes':[{'success_source':'user_task_success'}]}})
    panel.execute(artifact=str(tmp_path/'policy'))
    assert 'Task: custom' in panel.result.toPlainText()
    assert 'Success Source: user_task_success' in panel.result.toPlainText()
    assert 'Success/Failure: 0/1' in panel.result.toPlainText()


def test_source_script_owns_inputs_without_user_code_execution(panel,tmp_path):
    from soarm101_lab.real2sim.workspaces.tasks import generate_template,catalog
    workspace=panel._display_workspace
    script=generate_template(workspace,tmp_path/'source_script.py',catalog(workspace)[0],script_stage='source')
    marker=tmp_path/'executed'
    with script.open('a') as stream:stream.write(f"\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n")
    panel.task_kind.setCurrentIndex(panel.task_kind.findData('script'))
    panel.task_python.setText(str(script))
    panel.storage_name.clear();panel.port.setText('ignored_ui_port')
    until(lambda:panel.run.isEnabled())
    assert not marker.exists()
    assert panel.task_selector.isHidden() and not panel.primary.isEnabled()
    assert set(panel.stage_options())=={'execution_script','python'}
    from soarm101_lab.workflow.task_scripts import script_inputs
    artifact,path,opts=script_inputs(script,'source')
    assert opts['port']=='/dev/so101_leader' and opts['storage_name']=='source_script'
    assert path==panel.workspace_path
    panel.task_clear.click()
    assert not panel.task_selector.isHidden() and panel.primary.isEnabled()
