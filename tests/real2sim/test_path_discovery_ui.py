"""Actual Qt path consumers use the existing executor and discard stale work."""
import json
import threading
import time

import pytest

from test_path_discovery import native,canonical,put


@pytest.fixture
def panel(tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    app=QApplication.instance() or QApplication([])
    root=tmp_path/'workflow'
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    p=ui.WorkflowPanel(lambda *a:pytest.fail('No execution in discovery tests'))
    p.stage.setCurrentIndex(p.stage.findData('train'))
    yield p
    p.timer.stop();p.close();app.processEvents()


def until(predicate,timeout=3):
    from PyQt6.QtTest import QTest
    deadline=time.monotonic()+timeout
    while not predicate():
        assert time.monotonic()<deadline
        QTest.qWait(5)


def test_native_model_selection_without_artifact_json(panel):
    p=native(panel.root/'models/한글 이름/checkpoints/010000/pretrained_model','act')
    panel._request_model_discovery()
    until(lambda:panel.resume_models.findData(str(p))>=0)
    assert not (panel.root/'artifacts').exists()
    panel.resume_models.setCurrentIndex(panel.resume_models.findData(str(p)))
    panel.select_resume_model()
    assert not panel.run.isEnabled()
    until(lambda:panel._checkpoint_future is None)
    assert panel.architecture.currentData()=='act'
    assert panel.resume_training.isChecked() and panel.resume_path.text()==str(p)
    assert json.loads(panel.options.toPlainText())['steps']==200
    assert not (panel.root/'artifacts').exists()


def test_discovery_worker_heartbeat_latest_request_and_close(panel,tmp_path,monkeypatch):
    from PyQt6.QtCore import QTimer
    from soarm101_lab.workflow import ui
    # Drain the initial empty discovery before installing the controlled slow IO.
    until(lambda:panel._discovery_future is None)
    entered=threading.Event();release=threading.Event();calls=[];main=threading.get_ident()
    def scan(folder,**kw):
        assert threading.get_ident()!=main
        calls.append(str(folder));entered.set();release.wait(2)
        return [dict(name=str(folder.name),path=str(folder/'checkpoint'),format='lerobot-policy',category='policy')]
    monkeypatch.setattr(ui,'discover_inputs',scan)
    beats=[];timer=QTimer();timer.timeout.connect(lambda:beats.append(time.perf_counter()));timer.start(5)
    try:
        panel._request_model_discovery();until(entered.is_set)
        from PyQt6.QtTest import QTest
        QTest.qWait(100);assert len(beats)>=5
        panel.root=tmp_path/'latest'
        panel._request_model_discovery();release.set()
        until(lambda:panel._discovery_future is None and panel._discovery_request is None)
        assert panel.resume_models.findData(str(tmp_path/'latest/models/checkpoint'))>=0
        assert panel.resume_models.count()==2 and len(calls)==2
    finally:release.set();timer.stop()


def test_stale_checkpoint_selection_and_user_edit_are_ignored(panel,tmp_path,monkeypatch):
    from soarm101_lab.workflow import policy
    until(lambda:panel._discovery_future is None)
    first=threading.Event();release=threading.Event();main=threading.get_ident()
    def read(path):
        assert threading.get_ident()!=main
        if path=='first':first.set();release.wait(2)
        return dict(path=path,architecture='smolvla',steps=111 if path=='first' else 222)
    monkeypatch.setattr(policy,'checkpoint_metadata',read)
    try:
        panel.browse_checkpoint('first');until(first.is_set)
        panel.browse_checkpoint('second');release.set()
        until(lambda:panel._checkpoint_future is None)
        assert panel.resume_path.text()=='second' and json.loads(panel.options.toPlainText())['steps']==222
        panel.browse_checkpoint('first');panel.resume_path.setText('manual')
        until(lambda:panel._checkpoint_future is None)
        assert panel.resume_path.text()=='manual' and json.loads(panel.options.toPlainText())['steps']==222
    finally:release.set()


def test_close_invalidates_pending_discovery(panel,monkeypatch):
    from soarm101_lab.workflow import ui
    until(lambda:panel._discovery_future is None)
    release=threading.Event();entered=threading.Event()
    def scan(*a,**kw):entered.set();release.wait(2);return []
    monkeypatch.setattr(ui,'discover_inputs',scan)
    panel._request_model_discovery();until(entered.is_set)
    try:
        panel.close()
        assert panel._discovery_future is None and panel._discovery_request is None
    finally:release.set()


def test_dataset_picker_acceptance_is_read_only_marker_check(tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QApplication,QDialog
    from soarm101_lab.workflow.curation_ui import DatasetPicker
    from soarm101_lab.workflow import curation
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(curation,'evaluation_inputs',lambda *a:pytest.fail('Reader must not run in the picker'))
    for path in (canonical(tmp_path/'legacy'),put(tmp_path/'source.hdf5'),tmp_path/'evaluation'):
        if path.name=='evaluation':put(path/'evaluation_datasets.json',{'schema':'so101.sim-evaluation/1'})
        picker=DatasetPicker(None,str(tmp_path));picker.selectFile(str(path));picker.accept()
        assert picker.result()==QDialog.DialogCode.Accepted
        picker.close()


def test_refresh_finds_added_and_renamed_paths_without_registration(panel):
    p=native(panel.root/'models/one/pretrained_model')
    panel._request_model_discovery();until(lambda:panel.resume_models.findData(str(p))>=0)
    panel.resume_models.setCurrentIndex(panel.resume_models.findData(str(p)))
    moved=p.with_name('renamed');p.rename(moved)
    second=native(panel.root/'models/two/pretrained_model')
    panel._request_model_discovery()
    until(lambda:panel._discovery_future is None)
    assert panel.resume_models.currentData() is None
    assert {panel.resume_models.itemData(i) for i in range(1,panel.resume_models.count())}=={str(moved),str(second)}
    assert not (panel.root/'artifacts').exists()


def test_bad_selected_checkpoint_does_not_enable_execution(panel):
    p=native(panel.root/'models/bad/pretrained_model')
    (p/'config.json').write_text('invalid json')
    panel.browse_checkpoint(str(p))
    until(lambda:panel._checkpoint_future is None)
    assert panel.status.text().startswith('NOT_READY') and not panel.run.isEnabled()
    assert panel.resume_training.isChecked()
