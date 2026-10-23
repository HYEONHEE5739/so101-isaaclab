"""Identical workflow controls, artifact routing and operations for both engines."""
from soarm101_lab.workflow.paths import input_directory
import json
import sys
from pathlib import Path
import pytest


from test_artifact_storage import storage_workspace
from test_episode_path_ui import until

@pytest.fixture
def panel(tmp_path,storage_workspace):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    app=QApplication.instance() or QApplication([])
    p=ui.WorkflowPanel(lambda *args:None,root=tmp_path/'workflow')
    p.workspace_path=str(storage_workspace);p.worker_python=sys.executable
    p.backend_pythons={'isaac':sys.executable,'mujoco':sys.executable}
    p.storage_names={p.stage.itemData(i):'fixture_'+p.stage.itemData(i) for i in range(p.stage.count())}
    p.storage_names['real_eval']='fixture_real_eval';p.storage_name.setText('fixture_source')
    p.select_workspace(storage_workspace)
    until(lambda:p.run.isEnabled())
    yield p
    p.timer.stop();p.close();app.processEvents()


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
        assert panel.inputs.currentData()==panel.workspace_path
        assert panel.stage_options()['backend'] == backend






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




@pytest.mark.parametrize('stage',['source','annotate','datagen','convert','train','sim_eval','hf_dataset','hf_policy','real_eval'])
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


def test_annotation_name_is_user_owned_and_keeps_custom_edits(panel,tmp_path):
    from PyQt6.QtTest import QTest
    panel.storage_names.pop('annotate',None)
    panel.stage.setCurrentIndex(panel.stage.findData('annotate'))
    assert panel.storage_name.text()==''
    panel.storage_name.setText('내가_정한_주석')
    for name in ('source_A','val_임의'):
        path=tmp_path/(name+'.hdf5');path.write_bytes(b'candidate')
        panel.select_data(str(path))
        for _ in range(100):
            if panel.inputs.currentData()==str(path):break
            QTest.qWait(10)
        assert panel.inputs.currentData()==str(path)
        assert panel.storage_name.text()=='내가_정한_주석'
    panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
    panel.storage_name.setText('별도_생성본')
    panel.stage.setCurrentIndex(panel.stage.findData('annotate'))
    assert panel.storage_name.text()=='내가_정한_주석'


def test_resume_picker_defers_validation_and_partition_guard_remains(panel,tmp_path,monkeypatch):
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
    panel.browse_resume_data()
    assert panel.resume_data.text()==str(split)
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




def test_task_setup_failure_uses_existing_status_and_recovers(panel, monkeypatch):
    assert not hasattr(panel.task_selector, 'summary')
    from test_episode_path_ui import until
    def missing_scene(*args,**kwargs):
        raise ValueError('Selected input Scene is unavailable')
    with monkeypatch.context() as patch:
        patch.setattr(panel.service, 'plan', missing_scene)
        panel.sync_tasks()
        until(lambda:'Scene is unavailable' in panel.status.text())
        assert not panel.run.isEnabled()
    panel.sync_tasks()
    until(lambda:panel.status.text().startswith('READY'))
    assert panel.stage_options()['tasks']









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




def test_stage_options_survive_navigation(panel):
    panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
    values = json.loads(panel.options.toPlainText()); values['num_successful_demos'] = 7
    panel.options.setPlainText(json.dumps(values))
    panel.stage.setCurrentIndex(panel.stage.findData('source'))
    panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
    assert panel.stage_options()['num_successful_demos'] == 7


def test_model_specific_options_are_isolated_and_preserved(panel):
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    panel.architecture.setCurrentIndex(panel.architecture.findData('smolvla'))
    panel.options.editors['batch_size'][1].setValue(3)
    panel.architecture.setCurrentIndex(panel.architecture.findData('act'))
    panel.options.editors['steps'][1].setValue(12345)
    panel.architecture.setCurrentIndex(panel.architecture.findData('smolvla'))
    assert panel.stage_options()['batch_size']==3
    panel.architecture.setCurrentIndex(panel.architecture.findData('act'))
    assert panel.stage_options()['steps']==12345 and panel.architecture.findData('minivla')==-1


def test_optional_validation_shared_by_official_trainers(panel):
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    panel.validation_input.addItem('Val','fixture_validation_dataset')
    panel.validation_input.setCurrentIndex(1)
    for model in ('act','smolvla'):
        panel.architecture.setCurrentIndex(panel.architecture.findData(model))
        assert panel.stage_options()['validation_dataset']=='fixture_validation_dataset'


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








def test_settings_validation_is_debounced_off_main_thread_and_latest_only(panel,monkeypatch):
    import threading,time
    from PyQt6.QtCore import QTimer
    from PyQt6.QtTest import QTest
    main_thread=threading.get_ident();started=threading.Event();release=threading.Event();calls=[];ticks=[]
    def readiness(stage,artifact,workspace,options,**kwargs):
        calls.append((options['port'],threading.get_ident()))
        if len(calls)==1:
            started.set();release.wait(3)
            return {'ready':True,'reason':'old result'}
        return {'ready':False,'reason':'latest setting rejected'}
    monkeypatch.setattr(panel.service,'plan',readiness)
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
    def readiness(*args,**kwargs):
        started.set();release.wait(3);return {'ready':True,'reason':''}
    monkeypatch.setattr(panel.service,'plan',readiness)
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




def test_rejected_validation_never_switches_silently_to_another_dataset(panel):
    from concurrent.futures import Future
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    panel.validation_input.addItem('Rejected Val','invalid')
    panel.validation_input.addItem('Other Val','valid')
    panel.validation_input.setCurrentIndex(1)
    result=Future();result.set_result(('NOT_READY','bad dataset',['valid']))
    panel._ready_future=(panel._ready_generation,result);panel._poll_ready_check()
    assert panel.validation_input.currentData()=='invalid'
    assert panel.status.text().startswith('NOT_READY')
