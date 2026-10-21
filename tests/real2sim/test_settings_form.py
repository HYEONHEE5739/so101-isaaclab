import json
from PyQt6.QtWidgets import QApplication, QToolButton, QLabel, QTableWidgetItem
from soarm101_lab.workflow.settings_form import SettingsForm
from soarm101_lab.workflow.option_help import template


def test_all_output_editors_and_persisted_path_values_are_removed():
    from soarm101_lab.workflow.option_help import help_html,KEYS
    app=QApplication.instance() or QApplication([])
    form=SettingsForm()
    for stage in KEYS:
        form.set_stage(stage);form.setPlainText('{"output":"/tmp/old_nested_output"}')
        assert 'output' not in form.editors and 'output' not in json.loads(form.toPlainText())
        assert 'output' not in template(stage) and '<b>output</b>' not in help_html(stage)
    form.close()


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

def test_datagen_environment_choices_roundtrip():
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow.settings_form import SettingsForm
    app=QApplication.instance() or QApplication([])
    form=SettingsForm();form.set_stage('datagen');form.setPlainText('{"num_envs":4}')
    enabled,choice,kind=form.editors['num_envs']
    assert [choice.itemData(i) for i in range(choice.count())]==[1,2,4,8]
    assert json.loads(form.toPlainText())['num_envs']==4
    form.close()


def test_annotation_environment_choices_roundtrip():
    app=QApplication.instance() or QApplication([])
    form=SettingsForm();form.set_stage('annotate');form.setPlainText('{"num_envs":2}')
    enabled,choice,_=form.editors['num_envs']
    assert enabled.text()=='동시 주석 환경 수'
    assert [choice.itemData(i) for i in range(choice.count())]==[1,2,4,8]
    choice.setCurrentIndex(choice.findData(8))
    assert json.loads(form.toPlainText())['num_envs']==8
    form.close()


def test_ik_combobox_is_single_authority_and_datagen_only():
    import pytest
    from PyQt6.QtWidgets import QComboBox
    app=QApplication.instance() or QApplication([])
    form=SettingsForm();form.set_stage('datagen');form.setPlainText('{}')
    enabled,choice,kind=form.editors['ik_solver']
    assert isinstance(choice,QComboBox) and enabled.isChecked()
    assert json.loads(form.toPlainText())['ik_solver']=='scipy_ls'
    choice.setCurrentIndex(choice.findData('pinocchio_dls'))
    assert json.loads(form.toPlainText())['ik_solver']=='pinocchio_dls'
    form.setPlainText('{"ik_solver":"scipy_ls"}')
    assert form.editors['ik_solver'][1].currentData()=='scipy_ls'
    with pytest.raises(ValueError,match='Unknown ik_solver'):form.setPlainText('{"ik_solver":"invalid"}')
    form.set_stage('replay');form.setPlainText('{}')
    assert 'ik_solver' not in form.editors
    form.close()


def test_failure_policy_choices_roundtrip():
    import pytest
    app=QApplication.instance() or QApplication([])
    form=SettingsForm();form.set_stage('datagen');form.setPlainText('{}')
    check,choice,_=form.editors['ik_failure_policy']
    assert [choice.itemData(i) for i in range(choice.count())]==['strict','hold','best_effort']
    assert json.loads(form.toPlainText())['ik_failure_policy']=='strict'
    choice.setCurrentIndex(choice.findData('best_effort'))
    assert json.loads(form.toPlainText())['ik_failure_policy']=='best_effort'
    with pytest.raises(ValueError,match='Unknown ik_failure_policy'):form.setPlainText('{"ik_failure_policy":"fallback"}')
    form.close()


def test_minivla_checkpoint_frequency_is_editable():
    from soarm101_lab.workflow.option_help import TRAIN_KEYS
    app=QApplication.instance() or QApplication([])
    form=SettingsForm();form.allowed_keys=TRAIN_KEYS['act'];form.set_stage('train');form.setPlainText('{}')
    check,editor,_=form.editors['save_freq'];check.setChecked(True);editor.setValue(250)
    assert json.loads(form.toPlainText())['save_freq']==250
    form.close()


def test_training_has_no_minivla_preparation_options():
    from soarm101_lab.workflow.option_help import TRAIN_KEYS
    app=QApplication.instance() or QApplication([])
    f=SettingsForm();f.allowed_keys=TRAIN_KEYS['act'];f.set_stage('train');f.setPlainText('{}')
    assert not {'source_groups','val_every','model_config','legacy_group_split'} & set(f.editors)
    assert 'minivla' not in TRAIN_KEYS
    f.close()


def test_minivla_batch_size_numeric_control_keeps_other_models_default():
    from soarm101_lab.workflow.option_help import TRAIN_KEYS
    app=QApplication.instance() or QApplication([])
    f=SettingsForm();f.set_stage('train');f.minivla=True;f.allowed_keys=TRAIN_KEYS['act']
    f.setPlainText('{}');enabled,editor,_=f.editors['batch_size']
    assert editor.value()==8 and 'accumulation' not in f.editors
    enabled.setChecked(True);editor.setValue(4)
    assert json.loads(f.toPlainText())['batch_size']==4
    f.minivla=False;f.allowed_keys=TRAIN_KEYS['act'];f.setPlainText('{}')
    assert f.editors['batch_size'][1].value()==8
    f.close()


def test_real_evaluation_has_no_motion_checkbox_and_clear_camera_mapping():
    app=QApplication.instance() or QApplication([])
    form=SettingsForm();form.set_stage('real_eval')
    form.setPlainText(json.dumps({**template('real_eval'),'enable_motion':False}))
    assert 'enable_motion' not in form.editors
    assert 'enable_motion' not in json.loads(form.toPlainText())
    check,table,_=form.editors['camera_devices']
    assert table.horizontalHeaderItem(0).text()=='카메라 이름 (예: front)'
    assert table.horizontalHeaderItem(1).text()=='장치 경로 / 인덱스'
    table.insertRow(0)
    table.setItem(0,0,QTableWidgetItem('front'))
    table.setItem(0,1,QTableWidgetItem('/dev/cam_front'))
    assert json.loads(form.toPlainText())['camera_devices']=={'front':'/dev/cam_front'}
    form.close()
