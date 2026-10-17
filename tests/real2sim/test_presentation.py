import json
from types import SimpleNamespace
import numpy as np
import pytest
from soarm101_lab.workflow import presentation as module
from soarm101_lab.workflow.presentation import Presentation, RecorderControls, attach, send
from soarm101_lab.workflow.operations import Operations
from soarm101_lab.workflow.artifacts import Registry
from soarm101_lab.real2sim.storage import atomic
from test_workspaces import published


def session(tmp_path, monkeypatch, stage='replay'):
    root = tmp_path / 'presentation'
    atomic(root / 'config.json', {'stage': stage, 'mode': 'operator'})
    monkeypatch.setenv(module.SESSION_ENV, str(root))
    monkeypatch.setattr(module, '_current', None)
    return root


def test_modes_same_source_environment(published, tmp_path):
    r = Registry(tmp_path / 'registry'); a = r.register('workspace', published[3]); op = Operations(r)
    operator = op.plan('source', a, '', {'port': 'test', 'mode': 'operator', 'storage_name':'mode_test'}, run_dir=tmp_path)
    developer = op.plan('source', a, '', {'port': 'test', 'mode': 'developer', 'storage_name':'mode_test'}, run_dir=tmp_path)
    assert '--headless' in operator['argv']
    assert [v for v in operator['argv'] if v != '--headless'] == developer['argv']
    assert operator['workspace'] == developer['workspace']
    assert op.readiness('source', a, '', {'port': 'test', 'mode': 'invalid'})[0] == 'NOT_READY'


def test_adapter_preserves_observations_actions_and_close(tmp_path, monkeypatch):
    root = session(tmp_path, monkeypatch)
    obs = {'policy': {'side_cam': np.full((1, 8, 10, 3), 100, np.uint8),
                      'wrist_cam': np.full((1, 8, 10, 3), 30, np.uint8)}}
    result = (obs, {'untouched': True})
    actions = []; closed = []
    env = SimpleNamespace(step=lambda action: (actions.append(action) or result), reset=lambda: result,
                          close=lambda: closed.append(True))
    bridge = attach(env)
    assert attach(env) is bridge
    assert env.reset() is result
    action = object()
    assert env.step(action) is result and actions == [action]
    assert (root / 'sim_side.jpg').exists() and (root / 'sim_wrist.jpg').exists()
    assert not (root / 'real_side.jpg').exists()
    send(root, 'pause'); send(root, 'step')
    assert env.step(action) is result and bridge.paused
    send(root, 'resume'); env.step(action)
    assert not bridge.paused and bridge.steps == 3
    env.close()
    assert closed == [True] and json.loads((root / 'status.json').read_text())['state'] == 'FINISHED'


def test_source_controls_without_window_and_run_isolation(tmp_path, monkeypatch):
    root = session(tmp_path, monkeypatch, 'source'); bridge = module.current()
    controls = RecorderControls(bridge)
    identifier = send(root, 'save')
    assert controls.consume_save() and not controls.consume_save()
    assert json.loads((root / 'replies' / (identifier + '.json')).read_text())['accepted']
    send(root, 'discard'); assert controls.consume_discard()
    send(root, 'quit'); assert controls.should_quit()
    other = tmp_path / 'other'; atomic(other / 'config.json', {'stage': 'source', 'mode': 'operator'})
    assert not RecorderControls(Presentation(other)).should_quit()
    controls.destroy()


def test_manual_annotation_callbacks(tmp_path, monkeypatch):
    root = session(tmp_path, monkeypatch, 'annotate'); bridge = module.current(); events = []
    bridge.callbacks = {name: lambda n=name: events.append(n) for name in ('pause', 'resume', 'mark', 'skip')}
    for name in bridge.callbacks: send(root, name)
    bridge.poll()
    assert events == ['pause', 'resume', 'mark', 'skip']
    assert not bridge.paused  # Annotation's own replay pause loop owns this state.


def test_ui_operator_default_frames_and_controls(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, 'Registry', lambda: Registry(tmp_path / 'registry'))
    panel = ui.WorkflowPanel(lambda *args: None)
    assert panel.mode.currentData() == 'operator'
    root = session(tmp_path, monkeypatch); bridge = module.current()
    bridge.publish({'sim_side': np.zeros((10, 10, 3), np.uint8)}, force=True)
    panel.select_stage('replay')
    panel.service.session = root; panel.controller.busy = True; panel.controller.active = {'stage':'replay','owner':panel.current_key}; panel.current_screen.monitor.begin()
    panel.poll_preview()
    assert panel.current_screen.preview.feeds['sim_side'].pixmap() is not None
    assert 'real_side' not in panel.current_screen.preview.feeds
    assert panel.current_screen.monitor.controls['pause'].isEnabled() and 'save' not in panel.current_screen.monitor.controls
    panel.controller.busy = False; panel.current_screen.monitor.finish({'ok':True,'artifact':{'path':'fixture'}}); panel.poll_preview()
    assert panel.current_screen.preview.feeds['sim_side'].pixmap() is not None
    assert panel.current_screen.monitor.status.text()=='SUCCEEDED'
    panel.close()


def test_pause_waits_for_ui_resume(tmp_path, monkeypatch):
    root = session(tmp_path, monkeypatch); bridge = module.current()
    send(root, 'pause')
    waits = []
    def release(seconds):
        waits.append(seconds)
        assert json.loads((root / 'status.json').read_text())['state'] == 'PAUSED'
        send(root, 'resume')
    monkeypatch.setattr(module.time, 'sleep', release)
    bridge.before_step()
    assert len(waits) == 1 and not bridge.paused


def test_stage_specific_ui(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, 'Registry', lambda: Registry(tmp_path / 'registry'))
    panel = ui.WorkflowPanel(lambda *args: None)
    assert not panel.current_screen.episode_time.isHidden()
    assert not panel.current_screen.monitor.controls['save'].isHidden()
    assert 'mark' not in panel.current_screen.monitor.controls
    panel.select_stage('annotate')
    assert not panel.current_screen.preview.feeds['sim_overview'].isHidden()
    assert not hasattr(panel.current_screen, 'episode_time')
    assert 'mark' not in panel.current_screen.monitor.controls
    panel.current_screen.annotation_mode.setCurrentIndex(1)
    assert not panel.current_screen.monitor.controls['mark'].isHidden()
    assert not panel.current_screen.monitor.controls['skip'].isHidden()
    for stage in ('convert', 'train'):
        panel.select_stage(stage)
        assert panel.current_screen.preview is None
        assert all(button.isHidden() for button in panel.current_screen.monitor.controls.values())
    panel.close()


def test_generation_progress_uses_completed_attempts():
    from types import SimpleNamespace
    from soarm101_lab.workflow.presentation import GenerationProgress
    counters = SimpleNamespace(num_attempts=0, num_success=0, num_failures=0)
    progress = GenerationProgress(counters)
    assert progress()['episode_outcome'] == '진행 중'
    counters.num_attempts = 1; counters.num_success = 1
    assert progress()['episode_outcome'].startswith('성공')
    counters.num_attempts = 2; counters.num_failures = 1
    assert progress()['episode_outcome'].startswith('실패')


def test_observation_diagnostics_preserve_policy(tmp_path, monkeypatch):
    session(tmp_path, monkeypatch, 'sim_eval')
    bridge = module.current()
    policy = {'side_cam': np.zeros((1, 8, 8, 3), np.uint8)}
    obs = {'policy': policy, 'subtask_terms': {'grasp': np.array([True])}, 'task': {'success': np.array([False])}}
    bridge.observe(obs, force=True)
    assert bridge.details['task_signals'] == {'grasp': True, 'place': False}
    assert list(policy) == ['side_cam']


def test_generation_success_target_progress():
    from types import SimpleNamespace
    from soarm101_lab.workflow.presentation import GenerationProgress
    g = SimpleNamespace(num_attempts=12, num_success=3, num_failures=9)
    status = GenerationProgress(g, target=10)()
    assert status['outcome_reason'] == '성공 데이터 3/10개'
    assert status['attempts'] == 12 and status['failures'] == 9


def test_generation_success_rate():
    from types import SimpleNamespace
    from soarm101_lab.workflow.presentation import GenerationProgress
    g = SimpleNamespace(num_attempts=0, num_success=0, num_failures=0)
    progress = GenerationProgress(g, 10)
    assert progress()['success_rate'] is None
    g.num_attempts = 20; g.num_success = 8; g.num_failures = 12
    status = progress()
    assert status['success_rate'] == .4 and status['target_successes'] == 10


def test_datagen_progress_distinguishes_source_validation_and_live_successes(tmp_path,monkeypatch):
    from soarm101_lab.workflow.mimic import GenerationScheduler
    root=session(tmp_path,monkeypatch,'datagen');bridge=module.current()
    bridge.start_datagen(2,4);bridge.source_progress(1,14,'demo_1',88,440);bridge.status()
    status=json.loads((root/'status.json').read_text());text=module.datagen_status_text(status)
    assert status['successes']==status['attempts']==0 and len(status['env_status'])==4
    assert 'Source 검증 완료 1/14' in text and 'frame 88/440' in text and '전체 성공 0/2' in text
    scheduler=GenerationScheduler(2,20,4);bridge.bind_generation(scheduler)
    attempt=scheduler.claim(2);scheduler.finish(2,attempt,'success',{'steps':400},lambda:'demo_0')
    scheduler.set_state(0,'running',steps=12);bridge.status()
    status=json.loads((root/'status.json').read_text());text=module.datagen_status_text(status)
    assert '전체 성공 1/2' in text and '최근 결과: Env 2' in text
    assert 'Env 2: 성공 · 시도 1 · 성공 1' in text and 'step 12' in text
    pending=scheduler.claim(3)
    attempt=scheduler.claim(1);scheduler.finish(1,attempt,'success',{'steps':401},lambda:'demo_1')
    scheduler.finish(3,pending,'interrupted',{'steps':10})
    bridge.close();final=json.loads((root/'status.json').read_text())
    assert final['state']=='FINISHED' and final['successes']==2 and final['env_status']['1']['successes']==1
    text=module.datagen_status_text(final)
    assert text.startswith('목표 달성') and f'최근 결과: Env 3 · 시도 {pending} · 중단' in text
    assert f'최근 결과: Env 3 · 시도 {pending} · 목표 달성' not in text


@pytest.mark.parametrize('backend',['isaac','mujoco'])
def test_datagen_ui_uses_identical_progress_for_both_backends(tmp_path,monkeypatch,backend):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui,'Registry',lambda:Registry(tmp_path/'registry'))
    panel=ui.WorkflowPanel(lambda *args:None)
    panel.select_stage('datagen')
    panel.backend.setCurrentIndex(panel.backend.findData(backend))
    panel.controller.active={'stage':'datagen','owner':panel.current_key};panel.current_screen.monitor.begin();panel.controller.busy=True
    root=session(tmp_path,monkeypatch,'datagen');bridge=module.current();panel.service.session=root
    bridge.details['ik_solver']='pinocchio_dls'
    bridge.start_datagen(100,4);bridge.source_progress(1,14,'demo_1',88,440);bridge.status()
    panel.show();app.processEvents();panel.poll_preview()
    assert 'Source 검증 완료 1/14' in panel.current_screen.monitor.task_state.text()
    assert '전체 성공 0/100' in panel.current_screen.monitor.task_state.text() and 'Env 3: 대기' in panel.current_screen.monitor.task_state.text()
    assert 'Source 사전 검증 중' in panel.current_screen.monitor.status.text()
    panel.resize(1440,1050);app.processEvents();panel.poll_preview();app.processEvents()
    from PyQt6.QtCore import QRect,Qt
    bounds=panel.current_screen.monitor.task_state.fontMetrics().boundingRect(QRect(0,0,panel.current_screen.monitor.task_state.width()-20,10000),
        Qt.TextFlag.TextWordWrap,panel.current_screen.monitor.task_state.text())
    assert 'IK: pinocchio_dls' in panel.current_screen.monitor.task_state.text()
    assert panel.current_screen.monitor.task_state.height()>=bounds.height()+16
    panel.controller.busy=False;panel.close()


def test_datagen_failure_reasons_and_frozen_completion_duration():
    status={'target_successes':100,'successes':100,'attempts':108,'failures':5,
        'target_completed_seconds':125.9,'wall_seconds':999.,
        'failure_breakdown':{'grasp_failure':1,'drop_failure':2,'place_failure':2},
        'env_status':{'0':{'state':'interrupted','attempts':27,'successes':24,'failures':{'drop_failure':2}}}}
    text=module.datagen_status_text(status)
    assert '목표 100개 저장 완료: 2분 05초' in text
    assert '집기 실패: 1' in text and '이동 중 놓침: 2' in text and '컵 배치 실패: 2' in text
    assert '실패 원인: 이동 중 놓침 2' in text
    status.pop('target_completed_seconds');status['successes']=20;status['wall_seconds']=61
    assert '이번 생성 경과: 1분 01초' in module.datagen_status_text(status)


def test_training_status_text_reports_only_observed_metrics():
    from soarm101_lab.workflow.presentation import training_status_text
    row=dict(step=1200,target_steps=20000,samples=9600,loss=.4321,grad_norm=.765,
        learning_rate=[1e-4,1e-5],dataloading_s=.1,preprocessing_s=.02,update_s=.3,seconds=.42,
        peak_allocated_gib=1.64)
    text=training_status_text(row)
    assert 'step:1,200/20,000' in text and 'smpl:9,600' in text
    assert 'lr:1.0e-04,1.0e-05' in text and 'updt_s:0.300' in text and 'data_s:0.100' in text
    assert 'val:' not in text
    row.update(validation_loss=.51,best_validation_loss=.48)
    assert 'val:0.510' in training_status_text(row) and 'best_val:0.480' in training_status_text(row)
