from pathlib import Path
from soarm101_lab.workflow.view import LogTail


def test_incremental_utf8_and_progress_filter(tmp_path):
    path = tmp_path / 'process.log'; tail = LogTail()
    message = '[WORKFLOW] Episode 1 저장 완료\n'.encode()
    path.write_bytes(message[:-2])
    assert tail.read(path) == []
    with path.open('ab') as f: f.write(message[-2:])
    lines = tail.read(path)
    assert lines == ['[WORKFLOW] Episode 1 저장 완료'] and tail.relevant(lines[0])
    assert tail.read(path) == []
    assert not tail.relevant('| Index | Observation |')
    assert tail.relevant('The final task was not completed.')
    assert tail.relevant('Traceback (most recent call last):')
    assert tail.relevant('  File "example.py", line 1')
    assert tail.relevant('torch.OutOfMemoryError: CUDA out of memory')


def test_pipeline_starts_at_source_and_separates_management(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, 'Registry', lambda: Registry(tmp_path))
    panel = ui.WorkflowPanel(lambda *args: None)
    assert panel.stage.currentData() == 'source'
    assert panel.stage.findData('accept') == -1
    assert panel.feeds['real_side'].isHidden() and panel.feeds['real_wrist'].isHidden()
    assert not panel.feeds['sim_side'].isHidden()
    assert panel.result.maximumHeight() > 1000
    assert len(panel.shortcuts) == 2
    assert panel.controls['discard'].text().startswith('리셋')
    panel.section.setCurrentIndex(1)
    assert panel.stage.currentData() == 'accept' and panel.stage.findData('publish') >= 0
    panel.section.setCurrentIndex(2)
    assert panel.stage.currentData() == 'real_eval'
    assert panel.feeds['sim_side'].isHidden() and not panel.feeds['real_side'].isHidden()
    panel.close()


def test_source_arrow_shortcuts_use_existing_commands(tmp_path, monkeypatch):
    import json
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    from soarm101_lab.real2sim.storage import atomic
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, 'Registry', lambda: Registry(tmp_path / 'registry'))
    panel = ui.WorkflowPanel(lambda *args: None); panel.show(); panel.activateWindow()
    root = tmp_path / 'presentation'; atomic(root / 'status.json', {'state':'RUNNING','frames':{}})
    panel.service.session = root; panel.busy = True; panel.active_stage = 'source'
    panel.poll_preview(); panel.controls['save'].setFocus(); app.processEvents()
    QTest.keyClick(panel.controls['save'], Qt.Key.Key_Right)
    QTest.keyClick(panel.controls['save'], Qt.Key.Key_Left)
    app.processEvents()
    commands = [json.loads(p.read_text())['command'] for p in sorted((root / 'commands').glob('*.json'))]
    assert commands == ['save', 'discard']
    panel.close()


def test_training_log_keeps_loss_and_checkpoint():
    from soarm101_lab.workflow.view import LogTail
    tail = LogTail()
    assert tail.relevant('loss=0.024 lr=0.0001')
    assert tail.relevant('Writing checkpoint 1000')


def test_stage_option_help_and_restore(tmp_path, monkeypatch):
    import json
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    from soarm101_lab.workflow.option_help import KEYS, template, help_html
    from soarm101_lab.workflow.operations import STAGES
    assert set(KEYS) == set(STAGES)
    for stage in STAGES:
        assert isinstance(json.loads(json.dumps(template(stage))), dict)
        assert all(key in help_html(stage) for key in template(stage))
    assert template('real_eval')['enable_motion'] is False
    assert 'episode_time_s' not in template('source')
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, 'Registry', lambda: Registry(tmp_path))
    panel = ui.WorkflowPanel(lambda *args: None)
    assert 'Episode Time' in panel.options_help.toPlainText()
    panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
    assert json.loads(panel.options.toPlainText())['trials'] == 10
    assert '생성 trial' in panel.options_help.toPlainText()
    panel.options.setPlainText('{"trials": 3}')
    assert panel.stage_options()['trials'] == 3
    panel.reset_options()
    assert json.loads(panel.options.toPlainText())['trials'] == 10
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    assert 'batch_size' in panel.options_help.toPlainText()
    assert 'trials' not in json.loads(panel.options.toPlainText())
    panel.options.setPlainText('[]')
    import pytest
    with pytest.raises(ValueError, match='JSON 객체'):
        panel.stage_options()
    panel.close()
