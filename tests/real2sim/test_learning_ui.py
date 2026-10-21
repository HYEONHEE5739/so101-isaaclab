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
