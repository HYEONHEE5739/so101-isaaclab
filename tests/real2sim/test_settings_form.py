import json
from PyQt6.QtWidgets import QApplication, QToolButton, QLabel, QTableWidgetItem
from soarm101_lab.workflow.settings_form import SettingsForm
from soarm101_lab.workflow.option_help import template


def test_defaults_override_optional_and_help():
    app=QApplication.instance() or QApplication([])
    form=SettingsForm();form.set_stage('train');form.setPlainText(json.dumps(template('train')))
    enabled, editor, _=form.editors['steps']
    editor.setValue(50000)
    assert json.loads(form.toPlainText())['steps']==50000
    enabled.setChecked(False)
    assert 'steps' not in json.loads(form.toPlainText())
    enabled, editor, _=form.editors['base_model']
    assert not enabled.isChecked() and not editor.isEnabled()
    enabled.setChecked(True);editor.setText('lerobot/smolvla_base')
    assert json.loads(form.toPlainText())['base_model']=='lerobot/smolvla_base'
    button=form.findChildren(QToolButton)[0]
    button.setChecked(True)
    assert any(not label.isHidden() and '최종' not in label.text() for label in form.findChildren(QLabel))
    form.close()


def test_map_rows_preserve_string_camera_keys():
    app=QApplication.instance() or QApplication([])
    form=SettingsForm();form.set_stage('train');form.setPlainText('{}')
    check,table,_=form.editors['rename_map'];check.setChecked(True)
    table.insertRow(0)
    table.setItem(0,0,QTableWidgetItem('observation.images.side_cam'))
    table.setItem(0,1,QTableWidgetItem('observation.images.camera1'))
    assert json.loads(form.toPlainText())['rename_map']=={'observation.images.side_cam':'observation.images.camera1'}
    form.close()
