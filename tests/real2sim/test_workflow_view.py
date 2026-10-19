from pathlib import Path
import pytest
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


def test_pipeline_starts_at_source_without_retired_publication(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, 'Registry', lambda: Registry(tmp_path))
    panel = ui.WorkflowPanel(lambda *args: None)
    assert panel.stage.currentData() == 'source'
    assert not panel.task_selector.isHidden()
    assert panel.stage.findData('accept') == -1
    assert panel.feeds['real_side'].isHidden() and panel.feeds['real_wrist'].isHidden()
    assert not panel.feeds['sim_side'].isHidden()
    assert panel.result.maximumHeight() > 1000
    assert len(panel.shortcuts) == 2
    assert panel.controls['discard'].text().startswith('리셋')
    for index in range(1, panel.stage.count()):
        panel.stage.setCurrentIndex(index)
        assert panel.task_selector.isHidden()
    panel.stage.setCurrentIndex(panel.stage.findData('source'))
    assert not panel.task_selector.isHidden()
    panel.section.setCurrentIndex(1)
    assert panel.section.count() == 2
    assert panel.stage.findData('publish') == -1
    assert panel.stage.currentData() == 'real_eval'
    assert panel.task_selector.isHidden()
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
    assert set(KEYS) == set(STAGES)-{'merge'}  # Merge owns its separate dialog.
    for stage in KEYS:
        assert isinstance(json.loads(json.dumps(template(stage))), dict)
        assert all(key in help_html(stage) for key in template(stage))
    assert 'enable_motion' not in template('real_eval')  # Explicit start owns this internal flag.
    assert 'episode_time_s' not in template('source')
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, 'Registry', lambda: Registry(tmp_path))
    panel = ui.WorkflowPanel(lambda *args: None)
    assert 'Episode Time' in panel.options_help.toPlainText()
    panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
    assert json.loads(panel.options.toPlainText())['num_successful_demos'] == 10
    assert '만들 성공 데이터 수' in panel.options_help.toPlainText()
    panel.options.setPlainText('{"num_successful_demos": 3}')
    assert panel.stage_options(require_name=False)['num_successful_demos'] == 3
    panel.reset_options()
    assert json.loads(panel.options.toPlainText())['num_successful_demos'] == 10
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    assert 'batch_size' in panel.options_help.toPlainText()
    assert 'num_successful_demos' not in json.loads(panel.options.toPlainText())
    panel.options.setPlainText('[]')
    import pytest
    with pytest.raises(ValueError, match='JSON 객체'):
        panel.stage_options(require_name=False)
    panel.close()


def test_camera_decodes_only_new_frames_and_recovers_after_clear(tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QImage
    from soarm101_lab.workflow import view
    app=QApplication.instance() or QApplication([])
    path=tmp_path/'frame.png';image=QImage(12,12,QImage.Format.Format_RGB32);image.fill(0xff0000)
    assert image.save(str(path));calls=[];original=view.QPixmap
    def pixmap(path):calls.append(path);return original(path)
    monkeypatch.setattr(view,'QPixmap',pixmap)
    camera=view.CameraView()
    for _ in range(10):camera.show_image(path)
    assert len(calls)==1 and not camera.pixmap().isNull()
    # Some filesystems coalesce rapid writes into one timestamp tick.
    import os
    old=path.stat();image.fill(0x00ff00);assert image.save(str(path))
    os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns+1_000_000_000));camera.show_image(path)
    assert len(calls)==2
    assert camera.pixmap().toImage().pixelColor(0,0).green()==255
    camera.clear();camera.show_image(path)
    assert len(calls)==3 and not camera.pixmap().isNull()
    camera.close()


def test_conversion_progress_hides_encoder_info_but_keeps_errors():
    tail = LogTail()
    assert not tail.relevant('Svt[info]: SVT [config]: preset / tune / pred struct : 8 / PSNR / random access')
    assert tail.relevant('Svt[error]: Failed to encode frame')
    assert tail.relevant('[WORKFLOW] LeRobot 변환 [####----------------] 10/50 episodes · 20.0%')
    assert tail.relevant('[WORKFLOW] Episode 변환 완료 · dataset 최종 정리 중')


def test_training_model_selector_owns_architecture(tmp_path, monkeypatch):
    import json
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, 'Registry', lambda: Registry(tmp_path))
    panel = ui.WorkflowPanel(lambda *args: None)
    assert panel.architecture.isHidden()
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    assert not panel.primary.isHidden() and not panel.architecture.isHidden()
    assert 'architecture' not in json.loads(panel.options.toPlainText())
    assert panel.stage_options(require_name=False)['architecture'] == 'smolvla'
    panel.architecture.setCurrentIndex(panel.architecture.findData('act'))
    panel.options.setPlainText('{"architecture": "smolvla", "steps": 42}')
    assert panel.stage_options(require_name=False)['architecture'] == 'act'
    assert panel.stage_options(require_name=False)['steps'] == 42
    panel.reset_options()
    assert panel.stage_options(require_name=False)['architecture'] == 'act'
    panel.stage.setCurrentIndex(panel.stage.findData('convert'))
    assert panel.architecture.isHidden()
    panel.close()


def test_dashboard_navigation_selects_existing_stage(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, 'Registry', lambda: Registry(tmp_path))
    panel = ui.WorkflowPanel(lambda *args: None)
    assert panel.stage.isHidden()
    panel.navigation.setCurrentRow(panel.stage.findData('train'))
    assert panel.stage.currentData() == 'train'
    assert panel.dashboard_title.text() == ui.LABELS['train']
    panel.section.setCurrentIndex(1)
    assert panel.navigation.count() == panel.stage.count()
    assert panel.stage.currentData() == 'real_eval'
    panel.close()


def test_common_mimic_log_summary_keeps_diagnostic_values():
    import json
    line='[MIMIC] '+json.dumps({'env_id':2,'attempt':7,'category':'ik_failure','steps':3,
        'reason':'long raw diagnostics','ik':{'position_error_m':.001,'position_tolerance_m':.003,
            'rotation_error_rad':.14,'rotation_tolerance_rad':.12}})
    summary=LogTail.summary(line)
    assert 'Env 2' in summary and 'IK 계산 탈락' in summary and '0.140/0.12 rad' in summary
    assert 'long raw diagnostics' not in summary and LogTail().relevant(line)


@pytest.mark.parametrize('batch_size,accumulation,omitted',[(8,1,False),(1,8,True),(8,8,False)])
def test_minivla_periodic_resume_picker_reads_model_root(tmp_path,monkeypatch,batch_size,accumulation,omitted):
    import json
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    from soarm101_lab.learning.minivla.config import MiniVLAConfig
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui,'Registry',lambda:Registry(tmp_path/'registry'))
    model=tmp_path/'model';(model/'checkpoints').mkdir(parents=True)
    selected=model/'checkpoints/step_000002.pt';selected.write_bytes(b'not loaded by the GUI')
    MiniVLAConfig(steps=17,batch_size=batch_size,accumulation=accumulation).save(model/'config.json')
    if omitted:
        saved=json.loads((model/'config.json').read_text());saved.pop('batch_size');(model/'config.json').write_text(json.dumps(saved))
    panel=ui.WorkflowPanel(lambda *args:None)
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    panel.architecture.setCurrentIndex(panel.architecture.findData('minivla'))
    monkeypatch.setattr(ui.QFileDialog,'getOpenFileName',lambda *args:(str(selected),''))
    panel.browse_checkpoint()
    opts=panel.stage_options(require_name=False)
    assert opts['resume_checkpoint']==str(selected) and opts['steps']==17 and opts['batch_size']==batch_size
    assert opts['model_config']==str(model/'config.json')
    panel.close()


def test_training_log_json_uses_readable_lerobot_fields():
    import json
    row={'step':25,'loss':.125,'grad_norm':.5,'learning_rate':[.0001],'seconds':.75,'peak_allocated_gib':1.6}
    line=LogTail.summary(json.dumps(row))
    assert '[TRAIN] step:25' in line and 'loss:0.125' in line and 'lr:1.0e-04' in line
    assert 'grdn:0.500' in line and 'val:' not in line
    assert LogTail.summary('Traceback: example')=='Traceback: example'


def test_training_graph_waits_and_updates_without_a_file_dialog(tmp_path,monkeypatch):
    import json
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtTest import QTest
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui,'Registry',lambda:Registry(tmp_path/'registry'))
    monkeypatch.setattr(ui.QFileDialog,'getOpenFileName',lambda *args:(_ for _ in ()).throw(AssertionError('automatic graph must not ask for a file')))
    panel=ui.WorkflowPanel(lambda *args:None);panel.busy=True;panel.active_stage='train'
    panel.show_training_loss();chart=panel._loss_dialog.findChild(ui.TrainingLossPlot)
    assert chart.path is None and chart.rows==[] and not panel._loss_dialog.isModal()
    run=tmp_path/'run';(run/'presentation').mkdir(parents=True)
    custom=tmp_path/'custom_output';metrics=custom/'policy/metrics.jsonl'
    (run/'state.json').write_text(json.dumps({'stage':'train','plan':{'output':str(custom),'options':{'architecture':'minivla'}}}))
    panel.service.session=run/'presentation';chart.reload()
    assert chart.path==metrics and chart.rows==[]
    metrics.parent.mkdir(parents=True);(metrics.parent/'config.json').write_text('{"steps":100}')
    metrics.write_text('{"step":1,"loss":2.0}\n')
    QTest.qWait(1700)
    assert len(chart.rows)==1 and chart.target==100
    with metrics.open('a') as f:f.write('{"step":2,"training_loss":1.5,"validation_loss":1.8}\n')
    QTest.qWait(1700)
    assert len(chart.rows)==2 and chart.rows[-1]['validation_action_loss']==1.8
    chart.reload();assert len(chart.rows)==2
    panel._loss_dialog.close();panel.busy=False;panel.close()


def test_training_graph_recovers_latest_matching_run_after_reopen(tmp_path,monkeypatch):
    import json
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    app=QApplication.instance() or QApplication([])
    registry=Registry(tmp_path/'registry');monkeypatch.setattr(ui,'Registry',lambda:registry)
    panel=ui.WorkflowPanel(lambda *args:None)
    panel.inputs.blockSignals(True);panel.inputs.clear();panel.inputs.addItem('dataset','train_id');panel.inputs.blockSignals(False)
    monkeypatch.setattr(registry,'get',lambda key,**kwargs:{'type':'dataset','path':str(tmp_path/'dataset')})
    chosen=None
    for stamp,source in [('20261017_100000','train_id'),('20261017_110000','train_id'),('20261017_120000','different_id')]:
        run=registry.root/'runs'/(stamp+'_train_cube');run.mkdir(parents=True)
        output=tmp_path/stamp;metrics=output/'policy/metrics.jsonl';metrics.parent.mkdir(parents=True);metrics.write_text('')
        (run/'state.json').write_text(json.dumps({'stage':'train','source':source,'plan':{'output':str(output),'options':{'architecture':'minivla'}}}))
        if stamp=='20261017_110000':chosen=metrics
    assert panel.service.session is None and panel.training_metrics_path()==chosen
    panel.close()


def test_loss_graph_incremental_partial_rows_and_aliases(tmp_path):
    import json
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow.ui import TrainingLossPlot
    app=QApplication.instance() or QApplication([])
    path=tmp_path/'metrics.jsonl';(tmp_path/'config.json').write_text('{"steps":3000}')
    rows=[{'step':i,'loss':float(i)} for i in range(1,2501)]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"step":2501,')
    chart=TrainingLossPlot(path);chart.reload();assert len(chart.rows)==2500
    with path.open('a') as f:f.write('"training_loss":1.25,"validation_loss":1.5}\n')
    chart.reload();assert len(chart.rows)==2501 and chart.rows[-1]['validation_action_loss']==1.5
    chart.reload();assert len(chart.rows)==2501
    path.write_text('{"step":1,"loss":0.5}\n');chart.reload();assert len(chart.rows)==1
    chart.close()


def test_minivla_batch_size_defaults_and_stage_options(tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui,'Registry',lambda:Registry(tmp_path/'registry'))
    panel=ui.WorkflowPanel(lambda *args:None)
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    panel.architecture.setCurrentIndex(panel.architecture.findData('minivla'))
    assert panel.options.editors['batch_size'][1].value()==8
    panel.options.editors['batch_size'][1].setValue(4)
    assert panel.stage_options(require_name=False)['batch_size']==4 and 'accumulation' not in panel.options.editors
    panel.close()
