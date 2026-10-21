"""Identical workflow controls, artifact routing and operations for both engines."""
import json
import sys
from pathlib import Path
import pytest


@pytest.fixture
def panel(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui, operations, worker_request
    from soarm101_lab.workflow.artifacts import Registry
    from soarm101_lab.workflow import operations as service
    from soarm101_lab.simulation.mjcf import load_inputs
    from soarm101_lab.real2sim.storage import atomic
    app = QApplication.instance() or QApplication([])
    registry = Registry(tmp_path / 'registry')
    monkeypatch.setattr(ui, 'Registry', lambda: registry)
    p = ui.WorkflowPanel(lambda *args: None)
    profile = json.loads((operations.ROOT / 'tests/real2sim/fixtures/scene_profile.json').read_text())
    w, _, _ = load_inputs({'profile': profile}); w['root'] = tmp_path
    identity = {'fixture_workspace': 'same_identity'}
    for module in (ui, operations, service):
        monkeypatch.setattr(module, 'load', lambda *args: w)
        monkeypatch.setattr(module, 'metadata', lambda *args: identity)
    monkeypatch.setattr(worker_request, 'load', lambda *args: w)
    monkeypatch.setattr(operations, 'inspect_artifact', lambda *args: {})
    atomic(registry.root / 'artifacts/workspace.json', {
        'id': 'workspace', 'type': 'workspace', 'path': str(tmp_path), 'workspace': identity, 'created_at': 1})
    p.workspace_path = str(tmp_path); p.worker_python = sys.executable
    p.backend_pythons = {'isaac': sys.executable, 'mujoco': sys.executable}
    p.storage_names={p.stage.itemData(i):'fixture_'+p.stage.itemData(i) for i in range(p.stage.count())}
    p.storage_names['real_eval']='fixture_real_eval'
    p.storage_name.setText('fixture_source')
    p.refresh()
    yield p
    p.timer.stop(); p.close()


def test_backend_changes_only_execution_engine(panel):
    from soarm101_lab.workflow.operations import Operations
    before = [panel.stage.itemText(i) for i in range(panel.stage.count())]
    service = panel.service
    path, port = panel.workspace_path, panel.port.text()
    for backend in ('mujoco', 'isaac'):
        panel.backend.setCurrentIndex(panel.backend.findData(backend))
        assert panel.service is service and isinstance(service, Operations)
        assert [panel.stage.itemText(i) for i in range(panel.stage.count())] == before
        assert panel.workspace_path == path and panel.port.text() == port
        from PyQt6.QtTest import QTest
        for _ in range(200):
            if panel.run.isEnabled():break
            QTest.qWait(10)
        assert panel.run.isEnabled(), panel.status.text()
        assert panel.inputs.currentData() == 'workspace'
        assert panel.stage_options()['backend'] == backend


def test_scene_and_worker_python_are_automatic_without_editors(panel):
    calls = []
    panel.run_worker = lambda operation, done: calls.append((operation, done))
    for removed in ('workspace', 'python', 'connection_settings', 'artifact_search', 'artifact_type', 'artifact_task',
                    'artifact_details', 'suggest_storage', 'output_preview'):
        assert not hasattr(panel, removed)
    panel.workspace_path = ''
    panel.refresh()
    assert panel.workspace_path == panel.registry.get('workspace')['path']
    assert panel.stage_options()['python'] == sys.executable
    panel.execute()
    assert len(calls) == 1
    received = []
    panel.service.run = lambda *args: received.append(args) or {'no_artifact': True}
    result = calls[0][0](lambda *args: None)
    assert result['ok'] and received[0][2] == panel.workspace_path
    assert received[0][3]['python'] == sys.executable
    calls[0][1](result)


@pytest.mark.parametrize('backend',['mujoco','isaac'])
@pytest.mark.parametrize('storage',['hdf5','directory'])
def test_resume_picker_accepts_both_storage_formats_on_both_engines(panel,tmp_path,monkeypatch,backend,storage):
    import numpy as np
    from PyQt6.QtWidgets import QDialog
    from soarm101_lab.workflow.curation_ui import DatasetPicker
    from soarm101_lab.real2sim.storage import atomic
    from soarm101_lab.so101_joint_mapping import RobotCoordinates
    from test_source_partitions import source_fixture
    (panel.registry.root/'sources').mkdir()
    panel.backend.setCurrentIndex(panel.backend.findData(backend))
    if storage=='hdf5':
        selected=tmp_path/'previous.hdf5';source_fixture(selected,2)
    else:
        selected=tmp_path/'previous'
        atomic(selected/'project_provenance.json',{'backend':'mujoco','workspace':{},'coordinates':RobotCoordinates().metadata()})
        atomic(selected/'manifest.json',{'episodes':[{'id':'episode_0000','path':'episode_0000'}]})
        episode=selected/'episode_0000';episode.mkdir()
        for key in ('state','action'):np.save(episode/(key+'.npy'),np.zeros((4,6),np.float32))
        atomic(episode/'outcome.json',{'success':True})
    calls=[]
    def choose(picker):
        assert not picker.allow_evaluation
        calls.append(picker.directory().absolutePath())
        picker.selectFile(str(selected));picker.accept()
        return picker.result()
    monkeypatch.setattr(DatasetPicker,'exec',choose)
    assert 'HDF5 / 폴더' in panel.resume_data_browse.text()
    panel.browse_resume_data()
    assert calls==[str(panel.registry.root/'sources')]
    assert panel.resume_data.text()==str(selected) and panel.stage_options()['resume_data']==str(selected)
    assert panel.episodes.value()==(3 if storage=='hdf5' else 2)
    monkeypatch.setattr(DatasetPicker,'exec',lambda picker:QDialog.DialogCode.Rejected)
    panel.browse_resume_data()
    assert panel.resume_data.text()==str(selected)


@pytest.mark.parametrize('backend',['mujoco','isaac'])
def test_storage_name_keyboard_input_reaches_worker_and_stays_per_stage(panel,monkeypatch,backend):
    from PyQt6.QtTest import QTest
    panel.backend.setCurrentIndex(panel.backend.findData(backend))
    assert not panel.storage_name.isHidden()
    panel.storage_name.clear();panel.storage_name.setFocus();QTest.keyClicks(panel.storage_name,'red_cube_20')
    assert panel.stage_options()['storage_name']=='red_cube_20'
    panel.stage.setCurrentIndex(panel.stage.findData('convert'))
    assert not panel.storage_name.isHidden() and panel.stage_options()['storage_name']=='fixture_convert'
    panel.storage_name.setText('converted_custom')
    panel.stage.setCurrentIndex(panel.stage.findData('source'))
    assert panel.storage_name.text()=='red_cube_20'
    calls=[];received=[]
    panel.run_worker=lambda operation,done:calls.append((operation,done))
    panel.service.run=lambda *args:received.append(args) or {'no_artifact':True}
    panel.execute()
    assert not panel.storage_name.isEnabled() and not panel.resume_data_browse.isEnabled()
    result=calls[0][0](lambda *args:None)
    assert result['ok'] and received[0][3]['storage_name']=='red_cube_20'
    calls[0][1](result)
    assert panel.storage_name.isEnabled() and panel.resume_data_browse.isEnabled()
    panel.storage_name.setText('../bad_name')
    with pytest.raises(ValueError,match='저장 이름'):panel.stage_options()
    panel.storage_name.clear()
    with pytest.raises(ValueError,match='저장 이름을 입력'):panel.stage_options()


def test_source_requires_name_alert_and_ignores_retired_output_setting(panel,monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    alerts=[]
    monkeypatch.setattr(QMessageBox,'warning',lambda parent,title,message:alerts.append(message))
    panel.run_worker=lambda *args:pytest.fail('A missing name must not launch a worker')
    panel.storage_name.clear()
    panel.guard(panel.execute)
    assert alerts==['저장 이름을 입력하세요.'] and not panel.busy
    panel.options.setPlainText('{"output":"/tmp/old_source_output"}')
    assert 'output' not in panel.options.editors
    panel.storage_name.setText('테스트_1')
    options=panel.stage_options()
    assert options['storage_name']=='테스트_1' and 'output' not in options
    from soarm101_lab.workflow.artifacts import output_location
    for backend in ('mujoco','isaac'):
        output,boundary=output_location('source',{},dict(options,backend=backend),panel.registry.root)
        assert boundary==panel.registry.root/'sources/테스트_1'
        assert output==(boundary/'origin' if backend=='mujoco' else boundary/'origin/source.hdf5')


@pytest.mark.parametrize('stage',['source','replay','annotate','datagen','convert','train','sim_eval','hf_dataset','hf_policy','real_eval'])
def test_every_stage_has_required_name_and_passes_it_to_worker(panel,monkeypatch,stage):
    from PyQt6.QtWidgets import QMessageBox
    alerts=[];calls=[];received=[]
    monkeypatch.setattr(QMessageBox,'warning',lambda parent,title,message:alerts.append(message))
    if stage=='real_eval':panel.section.setCurrentIndex(1)
    else:panel.stage.setCurrentIndex(panel.stage.findData(stage))
    assert not panel.storage_name.isHidden()
    assert panel.main_form.labelForField(panel.storage_name).text()=='저장 이름 (필수)'
    panel.run_worker=lambda operation,done:calls.append((operation,done))
    panel.storage_name.clear();panel.guard(panel.execute)
    assert alerts==['저장 이름을 입력하세요.'] and not calls and not panel.busy
    # Exercise actual UI serialization/execute; only the external service is a fixture.
    panel.inputs.clear();panel.inputs.addItem('Input fixture','workspace')
    panel.task_setup_error=None
    panel.storage_name.setText('결과_'+stage)
    options=json.loads(panel.options.toPlainText());options['output']='/tmp/retired_nested_path'
    panel.options.setPlainText(json.dumps(options))
    panel.service.run=lambda *args:received.append(args) or {'no_artifact':True}
    panel.execute()
    assert not panel.storage_name.isEnabled()
    result=calls[0][0](lambda *args:None)
    assert result['ok'] and received[0][3]['storage_name']=='결과_'+stage
    assert 'output' not in received[0][3]
    calls[0][1](result)
    assert panel.storage_name.isEnabled()


def test_annotation_name_tracks_selected_source_and_keeps_custom_edits(panel):
    from soarm101_lab.real2sim.storage import atomic
    for name in ('source_A','source_B'):
        atomic(panel.registry.root/f'artifacts/{name}.json',dict(id=name,type='source',path=panel.workspace_path,
            display_name=name,created_at=1,workspace={'fixture_workspace':'same_identity'}))
    panel.storage_names.pop('annotate',None)
    panel.stage.setCurrentIndex(panel.stage.findData('annotate'))
    panel.inputs.setCurrentIndex(panel.inputs.findData('source_A'))
    assert panel.storage_name.text()=='annotation_source_A'
    panel.inputs.setCurrentIndex(panel.inputs.findData('source_B'))
    assert panel.stage_options()['storage_name']=='annotation_source_B'
    panel.storage_name.setText('내가_정한_주석')
    panel.inputs.setCurrentIndex(panel.inputs.findData('source_A'))
    assert panel.storage_name.text()=='내가_정한_주석'
    panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
    panel.storage_name.setText('별도_생성본')
    panel.stage.setCurrentIndex(panel.stage.findData('annotate'))
    assert panel.storage_name.text()=='내가_정한_주석'


def test_resume_picker_rejects_partitioned_directory_and_preserves_selection(panel,tmp_path,monkeypatch):
    import numpy as np
    from soarm101_lab.workflow.curation_ui import DatasetPicker
    from soarm101_lab.real2sim.storage import atomic
    from soarm101_lab.workflow.data_resume import validate
    split=tmp_path/'split_train'
    atomic(split/'project_provenance.json',{'workspace':{},'coordinates':{},'source_partition':{'split':'train'}})
    atomic(split/'manifest.json',{'episodes':[{'id':'episode_0000','path':'episode_0000'}]})
    e=split/'episode_0000';e.mkdir()
    for key in ('state','action'):np.save(e/(key+'.npy'),np.zeros((4,6),np.float32))
    atomic(e/'outcome.json',{})
    def choose(picker):
        picker.selectFile(str(split));picker.accept();return picker.result()
    monkeypatch.setattr(DatasetPicker,'exec',choose)
    panel.resume_data.setText('previous_selection.hdf5')
    with pytest.raises(ValueError,match='origin'):panel.browse_resume_data()
    assert panel.resume_data.text()=='previous_selection.hdf5'
    with pytest.raises(ValueError,match='Partition is locked'):validate('source',split,{},[])


def test_source_and_datagen_resume_fields_are_independent_through_other_stages(panel):
    panel.resume_data.setText('/tmp/source/dataset.videos')
    panel.stage.setCurrentIndex(panel.stage.findData('annotate'))
    panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
    assert panel.resume_data.text()=='' and 'resume_data' not in panel.stage_options()
    assert '새로 생성하면 비워' in panel.resume_data.placeholderText()
    assert panel.primary.layout().labelForField(panel.resume_data).text()=='기존 DataGen 결과 (선택)'
    panel.resume_data.setText('/tmp/previous_datagen/dataset.hdf5')
    panel.stage.setCurrentIndex(panel.stage.findData('source'))
    assert panel.resume_data.text()=='/tmp/source/dataset.videos'
    panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
    assert panel.stage_options()['resume_data']=='/tmp/previous_datagen/dataset.hdf5'
    panel.stage.setCurrentIndex(panel.stage.findData('convert'))
    assert 'resume_data' not in panel.stage_options()


def test_conversion_merge_dialog_checks_roles_and_uses_existing_worker(panel, tmp_path, monkeypatch):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QDialog
    from soarm101_lab.workflow import ui
    from soarm101_lab.real2sim.storage import atomic
    from test_lerobot_merge import converted_record
    records = []
    for name, role in [('first','train'), ('second','train'), ('validation','val')]:
        record = converted_record(tmp_path, name, role)
        root = Path(record['path'])
        atomic(root/'meta/info.json', record['format'])
        atomic(root/'workflow_provenance.json', record['provenance'])
        from soarm101_lab.so101_dataset_contract import write_contract
        write_contract(root, record['coordinates'])
        (root/'data').mkdir(); (root/'data/fixture.parquet').write_bytes(b'metadata-only fixture')
        records.append(panel.registry.register('dataset', root))
    panel.stage.setCurrentIndex(panel.stage.findData('source'))
    assert panel.merge_button.isHidden()
    panel.stage.setCurrentIndex(panel.stage.findData('convert'))
    assert not panel.merge_button.isHidden()
    dialog = ui.DatasetMergeDialog(panel.registry, panel)
    by_id = {dialog.datasets.item(i).data(Qt.ItemDataRole.UserRole)['id']: dialog.datasets.item(i) for i in range(dialog.datasets.count())}
    by_id[records[0]['id']].setCheckState(Qt.CheckState.Checked)
    by_id[records[2]['id']].setCheckState(Qt.CheckState.Checked)
    assert not dialog.start.isEnabled() and 'Train과 Val' in dialog.summary.text()
    by_id[records[2]['id']].setCheckState(Qt.CheckState.Unchecked)
    by_id[records[1]['id']].setCheckState(Qt.CheckState.Checked)
    dialog.name.setText('combined_train')
    assert dialog.start.isEnabled() and '총 2 episodes' in dialog.summary.text()
    assert 'datasets/combined_train' in dialog.location.text()
    monkeypatch.setattr(ui, 'DatasetMergeDialog', lambda *args: dialog)
    monkeypatch.setattr(dialog, 'exec', lambda: QDialog.DialogCode.Accepted)
    calls, received = [], []
    panel.run_worker = lambda operation,done:calls.append((operation,done))
    panel.service.run = lambda *args:received.append(args) or {'no_artifact':True}
    panel.merge_button.click()
    assert len(calls) == 1 and panel.busy and not panel.merge_button.isEnabled()
    result = calls[0][0](lambda *args:None)
    assert result['ok'] and received[0][0] == 'merge'
    assert received[0][3]['storage_name'] == 'combined_train'
    assert set(received[0][3]['merge_datasets']) == {r['id'] for r in records[:2]}
    calls[0][1](result)
    assert not panel.busy and panel.merge_button.isEnabled()
    dialog.close()


def test_merge_dialog_accepts_unpartitioned_multitask_inputs(panel, tmp_path):
    from PyQt6.QtCore import Qt
    from soarm101_lab.workflow.ui import DatasetMergeDialog
    from soarm101_lab.real2sim.storage import atomic
    from soarm101_lab.so101_dataset_contract import write_contract
    from test_lerobot_merge import unpartitioned_record
    for name in ('red', 'green', 'blue'):
        record = unpartitioned_record(tmp_path, name)
        root = Path(record['path'])
        atomic(root/'meta/info.json',record['format'])
        atomic(root/'workflow_provenance.json',record['provenance'])
        write_contract(root,record['coordinates'])
        (root/'data').mkdir();(root/'data/fixture.parquet').write_bytes(b'metadata-only fixture')
        panel.registry.register('dataset',root)
    dialog = DatasetMergeDialog(panel.registry,panel)
    try:
        for i in range(dialog.datasets.count()):dialog.datasets.item(i).setCheckState(Qt.CheckState.Checked)
        assert dialog.start.isEnabled() and '미분할' in dialog.summary.text()
        assert '총 3 episodes' in dialog.summary.text()
        dialog.datasets.item(0).setCheckState(Qt.CheckState.Unchecked)
        assert dialog.start.isEnabled()
        dialog.datasets.item(1).setCheckState(Qt.CheckState.Unchecked)
        assert not dialog.start.isEnabled() and '2개 이상' in dialog.summary.text()
    finally:dialog.close()


def test_real_evaluation_uses_editable_saved_defaults_and_explicit_start(panel,tmp_path,monkeypatch):
    from PyQt6.QtTest import QTest
    from soarm101_lab.workflow import ui,operations
    from soarm101_lab.real2sim import hardware,cameras
    from soarm101_lab.real2sim.storage import atomic
    from soarm101_lab.so101_dataset_contract import JOINT_SPACE
    # This routing fixture has no model contract. Hardware mappings are covered
    # by the separate real-worker calibration tests.
    monkeypatch.setattr(operations,'require_compatible',lambda *args:None)
    monkeypatch.setattr(operations,'coordinate_contract',lambda *args:{'action_space':JOINT_SPACE})
    saved={'follower_port':'/dev/saved_follower','follower_id':'saved_arm',
           'follower_calibration_dir':str(tmp_path/'calibration'),
           'side_camera':'/dev/saved_side','wrist_camera':'/dev/saved_wrist'}
    atomic(tmp_path/'outputs/real2sim/ui_settings.json',saved)
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    def forbidden(*args,**kwargs):
        raise AssertionError('UI/readiness must never open real hardware')
    monkeypatch.setattr(hardware,'Follower',forbidden)
    monkeypatch.setattr(cameras,'CameraReader',forbidden)
    identity=panel.registry.get('workspace')['workspace']
    atomic(panel.registry.root/'artifacts/policy.json',dict(id='policy',type='policy',
        path=str(tmp_path),workspace=identity,created_at=2,status='SUCCEEDED',
        format={'type':'act'},tasks=panel.task_selector.selection(),coordinates={'action_space':JOINT_SPACE}))
    panel.section.setCurrentIndex(1)
    panel.refresh();panel.inputs.setCurrentIndex(panel.inputs.findData('policy'))
    assert 'enable_motion' not in panel.options.editors
    for key,value in [('port',saved['follower_port']),('follower_id',saved['follower_id']),
                      ('calibration_dir',saved['follower_calibration_dir']),
                      ('side_device',saved['side_camera']),('wrist_device',saved['wrist_camera'])]:
        check,editor,_=panel.options.editors[key]
        assert check.isChecked() and editor.isEnabled() and editor.text()==value
    port=panel.options.editors['port'][1];port.selectAll();QTest.keyClicks(port,'/dev/other_follower')
    identifier=panel.options.editors['follower_id'][1];identifier.selectAll();QTest.keyClicks(identifier,'other_arm')
    assert panel.stage_options()['enable_motion'] is True
    assert panel.stage_options()['port']=='/dev/other_follower'
    panel.section.setCurrentIndex(0)
    assert 'enable_motion' not in panel.stage_options()
    panel.section.setCurrentIndex(1)
    assert panel.stage_options()['port']=='/dev/other_follower'
    assert panel.stage_options()['follower_id']=='other_arm'
    calls=[];received=[]
    panel.run_worker=lambda operation,done:calls.append((operation,done))
    panel.service.run=lambda *args:received.append(args) or {'no_artifact':True}
    assert not calls and not received
    panel.update_ready()
    for _ in range(200):
        if panel.run.isEnabled():break
        QTest.qWait(10)
    assert panel.run.isEnabled(),panel.status.text()
    panel.run.click()
    assert len(calls)==1 and not received
    result=calls[0][0](lambda *args:None)
    assert result['ok'] and received[0][0]=='real_eval'
    assert received[0][3]['enable_motion'] is True
    assert received[0][3]['port']=='/dev/other_follower'
    assert received[0][3]['follower_id']=='other_arm'
    calls[0][1](result)


def test_real_evaluation_defaults_use_current_connection_fields(panel,tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QWidget,QLineEdit
    from soarm101_lab.real2sim.storage import atomic
    parent=QWidget();parent.workspace=tmp_path
    atomic(tmp_path/'ui_settings.json',{'follower_id':'saved_arm','side_camera':'/dev/saved_side'})
    parent.fields={'follower_id':QLineEdit('new_arm'),'follower_port':QLineEdit('/dev/new_follower')}
    monkeypatch.setattr(panel,'window',lambda:parent)
    defaults=panel.real_evaluation_defaults()
    assert defaults['follower_id']=='new_arm' and defaults['port']=='/dev/new_follower'
    assert defaults['side_device']=='/dev/saved_side'
    parent.close()


def test_input_workspace_matches_recorded_identity_and_rejects_other_scene(panel, monkeypatch):
    from soarm101_lab.workflow import ui
    w = ui.load(panel.workspace_path)
    identity = ui.metadata(w)
    root = str(w['root'])
    calls = []
    def load(path):
        calls.append(path)
        if path != root:
            return {'root': path, 'identity': {'other': 'scene'}}
        return w
    monkeypatch.setattr(ui, 'load', load)
    monkeypatch.setattr(ui, 'metadata', lambda value: identity if value is w else value['identity'])
    panel.workspace_path = '/old/scene'
    assert panel.input_workspace({'type': 'dataset', 'workspace': identity}) is w
    assert calls == ['/old/scene']  # The already validated display Workspace is reused.
    with pytest.raises(ValueError, match='Scene'):
        panel.input_workspace({'type': 'policy', 'workspace': {'missing': 'scene'}})


def test_task_setup_failure_uses_existing_status_and_recovers(panel, monkeypatch):
    assert not hasattr(panel.task_selector, 'summary')
    def missing_scene(artifact):
        raise ValueError('Selected input Scene is unavailable')
    with monkeypatch.context() as patch:
        patch.setattr(panel, 'input_workspace', missing_scene)
        panel.sync_tasks()
        assert not panel.run.isEnabled()
        assert panel.status.text() == 'NOT_READY — Selected input Scene is unavailable'
        with pytest.raises(ValueError, match='Scene is unavailable'):
            panel.execute()
    panel.sync_tasks()
    assert panel.task_setup_error is None
    assert panel.stage_options()['tasks']


@pytest.mark.parametrize('stage,kind', [
    ('source','workspace'), ('replay','source'), ('annotate','source'),
    ('datagen','annotated'), ('convert','datagen'), ('train','dataset'),
    ('hf_dataset','dataset'), ('sim_eval','policy'), ('real_eval','policy'), ('hf_policy','policy')])
def test_data_selection_opens_stage_store_and_cancel_preserves_input(panel, tmp_path, monkeypatch, stage, kind):
    from soarm101_lab.workflow import ui
    from soarm101_lab.real2sim.storage import atomic
    identity = panel.registry.get('workspace')['workspace']
    path = tmp_path / 'upstream' / kind
    path.mkdir(parents=True)
    if kind in ('source','annotated','datagen'):
        path = path / 'dataset.hdf5';path.touch()
    if kind == 'workspace':
        path = tmp_path
        identifier = 'workspace'
    else:
        identifier = 'selected-input'
        atomic(panel.registry.root / 'artifacts/selected-input.json', dict(id=identifier, type=kind,
            path=str(path), workspace=identity, created_at=2, status='SUCCEEDED',
            format='so101.common-episode/2' if kind in ('source','annotated','datagen') else {},
            tasks=panel.task_selector.selection()))
    if stage == 'real_eval':panel.section.setCurrentIndex(1)
    else:panel.stage.setCurrentIndex(panel.stage.findData(stage))
    panel.refresh()
    panel.inputs.setCurrentIndex(panel.inputs.findData(identifier))
    calls = []
    monkeypatch.setattr(ui.QFileDialog, 'getExistingDirectory', lambda *args: calls.append(('directory',args)) or '')
    monkeypatch.setattr(ui.QFileDialog, 'getOpenFileName', lambda *args: calls.append(('file',args)) or ('',''))
    panel.data_select.click()
    assert len(calls) == 1
    assert calls[0][0] == 'directory'
    assert calls[0][1][2] == str(panel.registry.input_directory(stage))
    assert panel.inputs.currentData() == identifier
    assert not hasattr(panel,'kind') and not hasattr(panel,'browse_buttons')
    for removed in ('input_browse','collection_browse','browse','browse_collection','select_collection','browse_workspace'):
        assert not hasattr(panel,removed)
    panel.busy=True;panel.update_ready()
    assert not panel.data_select.isEnabled()
    panel.busy=False


@pytest.mark.parametrize('stage,kind',[('convert','datagen'),('real_eval','policy')])
def test_folder_selection_for_conversion_and_real_policy_ignores_sim_engine_filter(panel,tmp_path,stage,kind):
    from soarm101_lab.real2sim.storage import atomic
    folder=tmp_path/'selected';folder.mkdir()
    path=folder/'dataset.hdf5' if kind=='datagen' else folder/'pretrained_model'
    if kind=='datagen':path.touch()
    else:path.mkdir()
    atomic(panel.registry.root/'artifacts/selected.json',dict(id='selected',type=kind,path=str(path),
        workspace=panel.registry.get('workspace')['workspace'],created_at=2,status='SUCCEEDED',backend='mujoco',
        format='so101.common-episode/2' if kind=='datagen' else {'type':'act'}))
    panel.backend.setCurrentIndex(panel.backend.findData('isaac'))
    if stage=='real_eval':panel.section.setCurrentIndex(1)
    else:panel.stage.setCurrentIndex(panel.stage.findData(stage))
    panel.select_data(folder)
    assert panel.inputs.count()==1 and panel.inputs.currentData()=='selected'


def test_object_selection_persisted_per_workspace_and_not_overwritten_by_replay(panel,monkeypatch):
    import copy
    from PyQt6.QtCore import Qt
    from soarm101_lab.real2sim.workspaces.tasks import workspace_key
    from soarm101_lab.workflow import ui
    w=ui.load(panel.workspace_path)
    selector=panel.task_selector
    selector.configure(w,'source',active_object_ids=['cube_red'])
    first=json.dumps(workspace_key(w),sort_keys=True)
    assert panel.object_selections[first]==['cube_red']
    selector.configure(w,'replay',selector.selection(),w['task']['dynamic'])
    assert panel.object_selections[first]==['cube_red']
    other=copy.deepcopy(w);other['profile']['objects']['cube_red']['display_name']='New workspace red'
    selector.configure(other,'source')
    second=json.dumps(workspace_key(other),sort_keys=True)
    assert len(panel.object_selections[second])==3 and len(panel.object_selections[first])==1
    monkeypatch.setattr(ui,'load',lambda *args:w)
    panel.sync_tasks()
    assert selector.object_selection()==['cube_red']
    saved=json.loads((panel.registry.root/'ui.json').read_text())
    assert saved['object_selections'][first]==['cube_red']
    selector.objects.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert not panel.run.isEnabled() and 'least one' in panel.status.text()


@pytest.mark.parametrize('stage', ['annotate', 'datagen', 'convert', 'train', 'sim_eval', 'hf_dataset', 'hf_policy'])
def test_stage_controls_do_not_change_with_backend(panel, stage):
    panel.stage.setCurrentIndex(panel.stage.findData(stage))
    def state():
        return (panel.run.text(), panel.stage_summary.text(), panel.architecture.isHidden(),
                panel.annotation_mode.isHidden(), set(panel.options.editors),
                {k for k, v in panel.controls.items() if not v.isHidden()})
    before = state()
    panel.backend.setCurrentIndex(panel.backend.findData('mujoco'))
    assert state() == before


def test_source_completion_does_not_start_another_stage(panel, tmp_path):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.real2sim.storage import atomic
    calls = []
    panel.run_worker = lambda operation, done: calls.append((operation, done))
    panel.execute()
    source_path = tmp_path / 'source.hdf5'; source_path.touch()
    source = {'id': 'source', 'type': 'source', 'path': str(source_path), 'created_at': 2,
              'status': 'SUCCEEDED', 'tasks': panel.task_selector.selection()}
    atomic(panel.registry.root / 'artifacts/source.json', source)
    calls[0][1]({'ok': True, 'artifact': source})
    QApplication.processEvents()
    assert len(calls) == 1 and panel.active_stage == 'source'
    assert not panel.busy


def test_stage_options_survive_navigation(panel):
    panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
    values = json.loads(panel.options.toPlainText()); values['num_successful_demos'] = 7
    panel.options.setPlainText(json.dumps(values))
    panel.stage.setCurrentIndex(panel.stage.findData('source'))
    panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
    assert panel.stage_options()['num_successful_demos'] == 7


def test_model_specific_options_are_isolated_and_preserved(panel):
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    panel.options.editors['batch_size'][1].setValue(3)
    assert 'model_config' not in panel.options.editors
    assert 'source_groups' not in panel.options.editors
    panel.architecture.setCurrentIndex(panel.architecture.findData('minivla'))
    assert set(panel.options.editors) == {'model_config','batch_size','steps','val_every','save_freq','early_stopping','patience','legacy_group_split'}
    assert not hasattr(panel, "minivla_head")
    assert panel.stage_options()['batch_size']==8
    opts = panel.stage_options()
    from pathlib import Path
    assert json.loads(Path(opts['model_config']).read_text())['layers'] == 4
    panel.options.editors['steps'][1].setValue(12345)
    panel.architecture.setCurrentIndex(panel.architecture.findData('smolvla'))
    assert panel.stage_options()['batch_size'] == 3
    assert 'model_config' not in panel.stage_options()
    assert not hasattr(panel, "minivla_head")
    panel.architecture.setCurrentIndex(panel.architecture.findData('minivla'))
    assert panel.stage_options()['steps'] == 12345
    panel.stage.setCurrentIndex(panel.stage.findData('source'))
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    assert panel.stage_options()['steps'] == 12345


def test_model_change_refreshes_train_and_validation_input_roles(panel,tmp_path,monkeypatch):
    from soarm101_lab.real2sim.storage import atomic
    monkeypatch.setattr('soarm101_lab.workflow.episodes.validate_training_inputs',lambda *args:True)
    for role in ('train','val'):
        path=tmp_path/role;path.mkdir()
        atomic(panel.registry.root/('artifacts/'+role+'.json'),{
            'id':role,'type':'dataset','status':'SUCCEEDED','path':str(path),'created_at':2,
            'workspace':{'fixture_workspace':'same_identity'},'format':{'total_episodes':1},
            'provenance':{'source_partition':{'split':role,'source_ids':[role]}}})
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    assert panel.inputs.count()==2
    panel.architecture.setCurrentIndex(panel.architecture.findData('minivla'))
    assert panel.inputs.count()==1 and panel.inputs.currentData()=='train'
    assert panel.validation_input.count()==2 and panel.validation_input.itemData(1)=='val'
    panel.architecture.setCurrentIndex(panel.architecture.findData('act'))
    assert panel.inputs.count()==2 and panel.inputs.currentData()=='train'


def test_minivla_separate_validation_selection_does_not_affect_other_trainers(panel):
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    panel.architecture.setCurrentIndex(panel.architecture.findData('minivla'))
    panel.validation_input.addItem('Verified Val Source 6','fixture_validation_dataset')
    panel.validation_input.setCurrentIndex(1)
    assert panel.stage_options()['validation_dataset']=='fixture_validation_dataset'
    assert 'validation_dataset' not in panel.options.editors
    panel.architecture.setCurrentIndex(panel.architecture.findData('act'))
    assert 'validation_dataset' not in panel.stage_options()


def test_data_picker_can_add_minivla_val_from_another_folder_without_losing_train(panel,tmp_path,monkeypatch):
    from soarm101_lab.real2sim.storage import atomic
    from soarm101_lab.workflow import episodes
    identity=panel.registry.get('workspace')['workspace']
    for role in ('train','val'):
        path=tmp_path/('dataset_'+role);path.mkdir()
        atomic(panel.registry.root/f'artifacts/{role}.json',dict(id=role,type='dataset',path=str(path),
            workspace=identity,created_at=2,status='SUCCEEDED',format={'codebase_version':'v3.0'},
            provenance={'source_partition':{'split':role,'source_ids':[role]}},backend='isaac'))
    checks=[]
    monkeypatch.setattr(episodes,'validate_training_inputs',lambda artifact,opts:checks.append((artifact['id'],opts['validation_artifact']['id'])))
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    panel.architecture.setCurrentIndex(panel.architecture.findData('minivla'))
    panel.select_data(tmp_path/'dataset_train')
    assert panel.inputs.currentData()=='train'
    panel.select_data(tmp_path/'dataset_val')
    assert panel.inputs.currentData()=='train' and panel.validation_input.currentData()=='val'
    assert panel.input_folders['train']==str(tmp_path/'dataset_train')
    assert panel.stage_options()['validation_dataset']=='val' and ('train','val') in checks
    panel.refresh()
    assert panel.inputs.currentData()=='train' and panel.validation_input.currentData()=='val'


def test_minivla_custom_json_overrides_head_preset_and_drops_foreign_options(panel):
    from soarm101_lab.learning.minivla.config import MODEL_ROOT
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    panel.architecture.setCurrentIndex(panel.architecture.findData('minivla'))
    config = str(MODEL_ROOT / 'configs/default.json')
    panel.options.setPlainText(json.dumps({'model_config': config, 'batch_size': 999, 'base_model': 'unused'}))
    opts = panel.stage_options()
    assert opts['model_config'] == config and not hasattr(panel, "minivla_head")
    assert opts['batch_size']==999 and 'base_model' not in opts


@pytest.mark.parametrize('backend', ['isaac', 'mujoco'])
def test_source_uses_common_execution_and_keyboard(panel, tmp_path, backend):
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from soarm101_lab.real2sim.storage import atomic
    panel.backend.setCurrentIndex(panel.backend.findData(backend))
    calls = []
    panel.run_worker = lambda operation, done: calls.append((operation, done))
    panel.execute()
    assert panel.busy and not panel.backend.isEnabled() and not panel.navigation.isEnabled()
    root = tmp_path / 'presentation'
    atomic(root / 'status.json', {'state': 'RUNNING', 'frames': {}})
    panel.service.session = root
    panel.show(); panel.activateWindow(); panel.poll_preview()
    panel.controls['save'].setFocus(); QApplication.processEvents()
    QTest.keyClick(panel.controls['save'], Qt.Key.Key_Right)
    QTest.keyClick(panel.controls['save'], Qt.Key.Key_Left)
    QApplication.processEvents()
    assert [json.loads(p.read_text())['command'] for p in sorted((root / 'commands').glob('*.json'))] == ['save', 'discard']
    calls[0][1]({'ok': False, 'error': 'fixture completion'})
    assert not panel.busy and panel.backend.isEnabled()


@pytest.mark.parametrize('backend', ['isaac', 'mujoco'])
@pytest.mark.parametrize('architecture', ['act', 'smolvla', 'minivla'])
def test_lerobot_training_is_independent_of_selected_engine(panel, tmp_path, monkeypatch, backend, architecture):
    from soarm101_lab.workflow import operations
    from soarm101_lab.workflow import episodes
    from soarm101_lab.real2sim.workspaces.tasks import selected
    from soarm101_lab.so101_dataset_contract import JOINT_SPACE
    monkeypatch.setattr(operations, 'require_compatible', lambda *args: None)
    monkeypatch.setattr(operations, 'coordinate_contract', lambda *args: {'action_space': JOINT_SPACE})
    monkeypatch.setattr(panel.registry, 'verify', lambda *args: None)
    # This routing fixture has no dataset bytes; lineage is covered by partition tests.
    monkeypatch.setattr(episodes,'validate_training_inputs',lambda *args:None)
    w = operations.load('fixture')
    artifact = {'id': 'dataset', 'type': 'dataset', 'path': str(tmp_path / 'lerobot'),
        'format': {'codebase_version': 'v3.0'}, 'workspace': operations.metadata(w),
        'tasks': [selected(w)], 'coordinates': {'action_space': JOINT_SPACE}}
    plan = panel.service.plan('train', artifact, str(tmp_path), {
        'backend': backend, 'architecture': architecture, 'python': panel.worker_python}, run_dir=tmp_path)
    assert 'scripts/real2sim/workflow.py' in plan['argv']
    request = json.loads((tmp_path / 'request.json').read_text())
    assert request['options']['architecture'] == architecture
    assert request['options']['workspace'] == str(w['root'])


def test_builder_folder_import_uses_worker_python_and_persists(panel, tmp_path, monkeypatch):
    """GUI-only Python does not need YAML, USD or simulator dependencies."""
    import subprocess
    from types import SimpleNamespace
    from soarm101_lab.workflow import ui
    from soarm101_lab.real2sim.ui import Window
    export = tmp_path/'builder_export'; export.mkdir(); (export/'scene_manifest.yaml').write_text('schema: real2sim.package/1')
    monkeypatch.setattr(ui.QFileDialog, 'getExistingDirectory', lambda *a: str(export))
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda argv, **kw: (calls.append(argv) or SimpleNamespace(returncode=0, stdout=panel.workspace_path+'\n', stderr='')))
    registrations=[];monkeypatch.setattr(panel.registry,'register',lambda *a: registrations.append(a))
    Window.browse_scene(SimpleNamespace(workflow_panel=panel))
    assert calls[0][0] == panel.worker_python
    assert calls[0][-2:] == ['import-builder',str(export)]
    assert registrations[0][0] == 'workspace'
    saved=json.loads((panel.registry.root/'ui.json').read_text())
    assert saved['workspace'] == panel.workspace_path


def test_one_camera_scene_configures_pipeline_and_real_sim(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from test_builder_workspace import fruit_export
    from soarm101_lab.real2sim.workspaces.builder import import_scene
    from soarm101_lab.real2sim.ui import Window
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    app=QApplication.instance() or QApplication([])
    export=fruit_export.__wrapped__(tmp_path)
    scene=import_scene(export,registry=tmp_path/'environments')
    registry=Registry(tmp_path/'workflow');registry.register('workspace',scene)
    monkeypatch.setattr(ui,'Registry',lambda:registry)
    window=Window(tmp_path/'live')
    window.workflow_panel.workspace_path = str(scene);window.workflow_panel.refresh()
    panel=window.workflow_panel
    assert panel.scene_cameras==('front',)
    assert panel.feeds['sim_front'].isHidden() is False
    assert panel.feeds['sim_wrist'].isHidden()
    assert panel.task_selector.selection()[0]['task_id']=='fruit_sort'
    panel.section.setCurrentIndex(1)
    panel.stage.setCurrentIndex(panel.stage.findData('real_eval'))
    assert not panel.feeds['real_front'].isHidden() and panel.feeds['real_wrist'].isHidden()
    window.sync_scene_cameras()
    assert 'front_camera' in window.fields
    assert 'sim_front' in window.views and window.views['sim_wrist'].parentWidget().isHidden()
    window.close();app.processEvents()


def test_settings_validation_is_debounced_off_main_thread_and_latest_only(panel,monkeypatch):
    import threading,time
    from PyQt6.QtCore import QTimer
    from PyQt6.QtTest import QTest
    main_thread=threading.get_ident();started=threading.Event();release=threading.Event();calls=[];ticks=[]
    def readiness(stage,artifact,workspace,options):
        calls.append((options['port'],threading.get_ident()))
        if len(calls)==1:
            started.set();release.wait(3)
            return 'READY','old result'
        return 'NOT_READY','latest setting rejected'
    monkeypatch.setattr(panel.service,'readiness',readiness)
    timer=QTimer(panel);timer.timeout.connect(lambda:ticks.append(time.monotonic()));timer.start(10)
    try:
        before=time.monotonic()
        for text in ['/dev/a','/dev/ab','/dev/abc']:panel.port.setText(text)
        assert time.monotonic()-before<.15
        assert not calls and not panel.run.isEnabled()
        for _ in range(100):
            if started.is_set():break
            QTest.qWait(10)
        assert started.is_set() and len(calls)==1 and calls[0][0]=='/dev/abc'
        panel.port.setText('/dev/new')
        QTest.qWait(250)
        assert len(ticks)>10 and len(calls)==1 and not panel.run.isEnabled()
        release.set()
        for _ in range(200):
            if 'latest setting rejected' in panel.status.text():break
            QTest.qWait(10)
        assert panel.status.text()=='NOT_READY — latest setting rejected'
        assert [c[0] for c in calls]==['/dev/abc','/dev/new']
        assert all(c[1]!=main_thread for c in calls)
        assert not panel.run.isEnabled()
    finally:release.set();timer.stop()


def test_background_ready_result_cannot_enable_run_during_workflow(panel,monkeypatch):
    import threading
    from PyQt6.QtTest import QTest
    started=threading.Event();release=threading.Event()
    def readiness(*args):
        started.set();release.wait(3);return 'READY',''
    monkeypatch.setattr(panel.service,'readiness',readiness)
    panel.update_ready()
    try:
        for _ in range(100):
            if started.is_set():break
            QTest.qWait(10)
        assert started.is_set()
        panel.busy=True;panel.update_ready();release.set()
        QTest.qWait(150)
        assert panel.status.text().startswith('RUNNING') and not panel.run.isEnabled()
    finally:release.set();panel.busy=False


def test_display_metadata_never_replaces_full_readiness_or_execution_record(panel,monkeypatch):
    from soarm101_lab.real2sim.storage import atomic
    from PyQt6.QtTest import QTest
    record=panel.registry.get('workspace');record['report']={'proof':'full record'}
    atomic(panel.registry.root/'artifacts/workspace.json',record)
    checks=[]
    def readiness(stage,artifact,*args):
        assert artifact['report']==record['report'];checks.append('ready');return 'READY',''
    monkeypatch.setattr(panel.service,'readiness',readiness)
    panel.update_ready()
    assert 'report' not in panel._ready_request[2]
    for _ in range(100):
        if checks and panel.run.isEnabled():break
        QTest.qWait(10)
    assert checks and panel.run.isEnabled()
    def run(stage,artifact,*args):
        assert artifact['report']==record['report'];checks.append('run');return {'no_artifact':True}
    monkeypatch.setattr(panel.service,'run',run)
    panel.run_worker=lambda fn,done:done(fn(lambda *args:None))
    panel.execute()
    assert 'run' in checks


def test_preview_keeps_same_frame_without_redecoding_and_updates_new_frame(panel,tmp_path,monkeypatch):
    from PyQt6.QtGui import QImage
    from soarm101_lab.workflow import ui
    from soarm101_lab.real2sim.storage import atomic
    session=tmp_path/'ipc';session.mkdir();image=QImage(12,12,QImage.Format.Format_RGB32);image.fill(0xff0000)
    for name in ('sim_side','sim_wrist'):assert image.save(str(session/(name+'.jpg')))
    status={'state':'RUNNING','frames':{'sim_side':100.,'sim_wrist':100.}}
    atomic(session/'status.json',status);calls=[];pixmap=ui.QPixmap
    monkeypatch.setattr(ui,'QPixmap',lambda path:calls.append(path) or pixmap(path))
    monkeypatch.setattr(ui.time,'time',lambda:100.)
    panel.service.session=session;panel.active_stage='source'
    for _ in range(10):panel.poll_preview()
    assert len(calls)==2
    status['frames']={k:101. for k in status['frames']};atomic(session/'status.json',status)
    panel.poll_preview();assert len(calls)==4
    status['state']='PAUSED';atomic(session/'status.json',status)
    panel.poll_preview();assert len(calls)==4


def test_minivla_candidate_validation_runs_off_ui_thread(panel,tmp_path,monkeypatch):
    import threading
    from PyQt6.QtCore import QTimer
    from PyQt6.QtTest import QTest
    from soarm101_lab.real2sim.storage import atomic
    started=threading.Event();release=threading.Event();threads=[];ticks=[]
    main=threading.get_ident();identity=panel.registry.get('workspace')['workspace']
    for role in ('train','val'):
        path=tmp_path/role;path.mkdir()
        atomic(panel.registry.root/f'artifacts/{role}.json',dict(id=role,type='dataset',path=str(path),
            workspace=identity,created_at=2,status='SUCCEEDED',format={'total_episodes':1},
            provenance={'source_partition':{'split':role,'source_ids':[role]},'episode_source_lineage':{'0':'proof'}}))
    def check(artifact,options):
        assert artifact['provenance']['episode_source_lineage']
        assert options['validation_artifact']['provenance']['episode_source_lineage']
        threads.append(threading.get_ident());started.set();release.wait(3)
        raise ValueError('incompatible validation fixture')
    monkeypatch.setattr('soarm101_lab.workflow.episodes.validate_training_inputs',check)
    monkeypatch.setattr(panel.service,'readiness',lambda *a:('READY',''))
    timer=QTimer(panel);timer.timeout.connect(lambda:ticks.append(1));timer.start(10)
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('train'))
        panel.architecture.setCurrentIndex(panel.architecture.findData('minivla'))
        for _ in range(100):
            if started.is_set():break
            QTest.qWait(10)
        assert started.is_set()
        QTest.qWait(120);assert len(ticks)>=5 and all(t!=main for t in threads)
        release.set()
        for _ in range(100):
            if panel.validation_input.count()==1:break
            QTest.qWait(10)
        assert panel.validation_input.count()==1
    finally:release.set();timer.stop()


def test_rejected_validation_never_switches_silently_to_another_dataset(panel):
    from concurrent.futures import Future
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    panel.architecture.setCurrentIndex(panel.architecture.findData('minivla'))
    panel.validation_input.addItem('Rejected Val','invalid')
    panel.validation_input.addItem('Other Val','valid')
    panel.validation_input.setCurrentIndex(1)
    result=Future();result.set_result(('NOT_READY','bad dataset',['valid']))
    panel._ready_future=(panel._ready_generation,result);panel._poll_ready_check()
    assert panel.validation_input.findData('invalid')<0 and panel.validation_input.findData('valid')>0
    assert panel.validation_input.currentData() is None


@pytest.mark.parametrize('stage', ['sim_eval', 'real_eval'])
@pytest.mark.parametrize('architecture', ['act', 'smolvla', 'minivla'])
def test_evaluation_task_editor_visible_and_instruction_forwarded(panel, stage, architecture):
    from soarm101_lab.real2sim.storage import atomic
    from soarm101_lab.workflow import ui
    atomic(panel.registry.root / 'artifacts/evaluation_policy.json', {
        'id': 'evaluation_policy', 'type': 'policy', 'path': str(panel.registry.root / 'model'),
        'workspace': ui.metadata(ui.load(panel.workspace_path)), 'created_at': 2,
        'format': {'type': architecture}, 'tasks': panel.task_selector.selection()})
    panel.refresh()
    if stage == 'real_eval':
        panel.section.setCurrentIndex(panel.section.findData('real_eval'))
    else:
        panel.stage.setCurrentIndex(panel.stage.findData(stage))
    panel.inputs.setCurrentIndex(panel.inputs.findData('evaluation_policy'))
    panel.sync_tasks()
    selector = panel.task_selector
    assert panel.stage.currentData() == stage
    assert not selector.isHidden()
    assert selector.pick.isEnabled() and selector.target.isEnabled() and selector.instruction.isEnabled()
    text = 'Pick the selected block and place it in the selected cup.'
    selector.instruction.setText(text)
    selector.instruction.textEdited.emit(text)
    tasks = panel.stage_options(require_name=False)['tasks']
    assert tasks[0]['language_instruction'] == text
    assert tasks[0]['pick_object_id'] == selector.pick.currentText()
    assert tasks[0]['target_object_id'] == selector.target.currentText()
