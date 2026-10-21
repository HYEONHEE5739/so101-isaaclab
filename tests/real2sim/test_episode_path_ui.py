"""Real Qt signals/executor for Path input, READY, Run and stale work."""
import threading,time
from pathlib import Path
import h5py
import pytest
from test_episode_paths import source,episode,published


def until(predicate,timeout=5):
    from PyQt6.QtTest import QTest
    deadline=time.monotonic()+timeout
    while not predicate():
        assert time.monotonic()<deadline
        QTest.qWait(10)


@pytest.fixture
def panel(source,tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    app=QApplication.instance() or QApplication([])
    p=ui.WorkflowPanel(lambda *a:pytest.fail('No implicit execution'),root=tmp_path/'empty registry')
    p.workspace_path=str(source[1]['root'])
    p.stage.setCurrentIndex(p.stage.findData('replay'))
    yield p
    p.timer.stop();p.close();app.processEvents()


@pytest.mark.parametrize('stage',['replay','annotate'])
def test_path_header_ready_and_run_have_no_id_or_main_thread_hdf5(panel,source,monkeypatch,stage):
    path,w,*_=source;main=threading.get_ident();opened=[];real=h5py.File
    def hdf(*a,**kw):
        assert threading.get_ident()!=main
        opened.append(str(a[0]));return real(*a,**kw)
    monkeypatch.setattr(h5py,'File',hdf)
    panel.stage.setCurrentIndex(panel.stage.findData(stage));panel.storage_name.setText('自由_結果')
    panel.select_data(str(path))
    until(lambda:panel.run.isEnabled())
    assert panel.inputs.currentData()==str(path) and opened
    assert panel.task_selector.selection()==[w['task_definition']]
    assert panel.input_selections[stage]=={'path':str(path)}
    calls=[]
    def run(stage,input_path,workspace,options,notify):
        calls.append((stage,input_path,workspace,options));return {'path':str(path.parent/'result/report.json'),'stage':stage,'status':'SUCCEEDED'}
    monkeypatch.setattr(panel.service,'run',run)
    panel.run_worker=lambda fn,done:done(fn(lambda *a:None))
    panel.execute()
    assert calls and calls[0][1]==str(path) and calls[0][2]==str(w['root'])
    assert not (panel.root/'artifacts').exists()


def test_input_scan_heartbeat_latest_request_and_stage_change(panel,tmp_path,monkeypatch):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtTest import QTest
    from soarm101_lab.workflow import ui
    until(lambda:panel._input_future is None)
    entered=threading.Event();release=threading.Event();main=threading.get_ident();calls=[]
    def scan(folder,**kw):
        assert threading.get_ident()!=main
        calls.append(str(folder));entered.set();release.wait(3)
        return [{'name':'candidate','path':str(Path(folder)/'source.hdf5'),'format':'hdf5'}]
    monkeypatch.setattr(ui,'discover_inputs',scan)
    beats=[];timer=QTimer();timer.timeout.connect(lambda:beats.append(time.perf_counter()));timer.start(5)
    try:
        panel.select_data(str(tmp_path/'first'));until(entered.is_set)
        QTest.qWait(100);assert len(beats)>=5
        panel.select_data(str(tmp_path/'second'));release.set()
        until(lambda:panel._input_future is None and panel._input_request is None)
        assert panel.inputs.findData(str(tmp_path/'second/source.hdf5'))>=0 and len(calls)==2
        panel.select_data(str(tmp_path/'third'))
        panel.stage.setCurrentIndex(panel.stage.findData('train'))
        until(lambda:panel._input_future is None)
        assert panel.inputs.findData(str(tmp_path/'third/source.hdf5'))<0
    finally:release.set();timer.stop()


def test_late_header_cannot_enable_run_for_other_stage_or_after_close(panel,source,monkeypatch):
    path=source[0];entered=threading.Event();release=threading.Event();plan=panel.service.plan
    def slow(*a,**kw):entered.set();release.wait(3);return plan(*a,**kw)
    monkeypatch.setattr(panel.service,'plan',slow)
    panel.storage_name.setText('new');panel.select_data(str(path));until(entered.is_set)
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('train'));release.set()
        until(lambda:panel._ready_future is None)
        assert panel.stage.currentData()=='train' and not panel.run.isEnabled()
        panel.stage.setCurrentIndex(panel.stage.findData('replay'));panel.select_data(str(path))
        panel.close()
        assert panel._input_future is None and panel._ready_future is None
    finally:release.set()


def test_missing_name_still_shows_recorded_task_and_corruption_is_not_ready(panel,source):
    path,w,*_=source;panel.storage_name.clear();panel.select_data(str(path))
    until(lambda:panel._path_details is not None)
    assert panel.task_selector.selection()==[w['task_definition']] and not panel.run.isEnabled()
    panel.storage_name.setText('new')
    until(lambda:panel.run.isEnabled())
    path.write_bytes(b'damaged header')
    panel.update_ready()
    until(lambda:panel._ready_future is None and not panel._ready_timer.isActive())
    assert panel.status.text().startswith('NOT_READY') and not panel.run.isEnabled()
