from pathlib import Path
import json
from PyQt6.QtWidgets import QApplication
from soarm101_lab.workflow.ui import WorkflowPanel


def test_independent_learning_paths_options_and_output(tmp_path):
    app=QApplication.instance() or QApplication([])
    panel=WorkflowPanel(lambda *a:None,root=tmp_path/'workflow')
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('train'))
        assert panel.architecture.findData('minivla')==-1
        panel.storage_name.setText('model')
        panel.learning_output.setText(str(tmp_path/'output'))
        for architecture in ('act','smolvla'):
            panel.architecture.setCurrentIndex(panel.architecture.findData(architecture))
            assert not panel.validation_input.isHidden()
            assert 'validation_dataset' not in panel.stage_options()
        path=str(tmp_path/'external_val')
        panel.validation_input.addItem('external',path);panel.validation_input.setCurrentIndex(panel.validation_input.findData(path))
        assert panel.stage_options()['validation_dataset']==path
        assert panel.stage_options()['output']==str(tmp_path/'output/model')
        panel.stage.setCurrentIndex(panel.stage.findData('validate'));panel.storage_name.setText('report')
        panel.validation_input.addItem('independent',path);panel.validation_input.setCurrentIndex(panel.validation_input.findData(path))
        opts=panel.stage_options()
        assert opts['validation_dataset']==path and 'architecture' not in opts and not panel.validation_input.isHidden()
        assert panel.stage.currentData()=='validate' and not panel.run.isEnabled()
    finally:panel.close()


def test_hf_import_runs_in_worker_and_selects_path(tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    from types import SimpleNamespace
    app=QApplication.instance() or QApplication([])
    jobs=[];panel=WorkflowPanel(lambda operation,done:jobs.append((operation,done)),root=tmp_path/'workflow')
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('train'))
        assert not panel.hf_import_button.isHidden()
        output=tmp_path/'external';answers=iter([('owner/dataset',True),('revision123',True),(str(output),True)])
        monkeypatch.setattr(QInputDialog,'getText',lambda *a,**kw:next(answers))
        selected=[];monkeypatch.setattr(panel,'select_data',selected.append)
        calls=[]
        def run(argv,**kw):
            calls.append(argv);return SimpleNamespace(returncode=0,stderr='')
        monkeypatch.setattr('subprocess.run',run)
        panel.import_hf_dataset()
        assert panel.busy and len(jobs)==1 and not panel.hf_import_button.isEnabled()
        operation,done=jobs[0];done(operation(lambda message:None))
        assert not panel.busy and selected==[str(output)]
        assert calls[0][-2:]==['--revision','revision123']
    finally:panel.close()


def test_native_hub_model_error_releases_ui(tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    from types import SimpleNamespace
    app=QApplication.instance() or QApplication([]);jobs=[]
    panel=WorkflowPanel(lambda fn,done:jobs.append((fn,done)),root=tmp_path/'workflow')
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('train'))
        answers=iter([('owner/model',True),('revision',True)])
        monkeypatch.setattr(QInputDialog,'getText',lambda *a,**kw:next(answers))
        monkeypatch.setattr('subprocess.run',lambda *a,**k:SimpleNamespace(returncode=1,stderr='native model config missing'))
        panel.import_hf_model();assert panel.busy and len(jobs)==1
        fn,done=jobs[0];done(fn(lambda msg:None))
        assert not panel.busy and 'native model config missing' in panel.result.toPlainText()
    finally:panel.close()
