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
    operator = op.plan('source', a, '', {'port': 'test', 'mode': 'operator'}, run_dir=tmp_path)
    developer = op.plan('source', a, '', {'port': 'test', 'mode': 'developer'}, run_dir=tmp_path)
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
    panel.stage.setCurrentIndex(panel.stage.findData('replay'))
    panel.service.session = root; panel.busy = True; panel.active_stage = 'replay'
    panel.poll_preview()
    assert panel.feeds['sim_side'].pixmap() is not None
    assert panel.feeds['real_side'].isHidden()
    assert panel.controls['pause'].isEnabled() and not panel.controls['save'].isEnabled()
    panel.busy = False; panel.poll_preview()
    assert panel.feeds['sim_side'].pixmap() is not None
    assert '실행 종료' in panel.preview_status.text()
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
    assert not panel.episode_time.isHidden()
    assert not panel.controls['save'].isHidden()
    assert panel.controls['mark'].isHidden()
    panel.stage.setCurrentIndex(panel.stage.findData('annotate'))
    assert not panel.feeds['sim_overview'].isHidden()
    assert panel.episode_time.isHidden()
    assert panel.controls['mark'].isHidden()
    panel.annotation_mode.setCurrentIndex(1)
    assert not panel.controls['mark'].isHidden()
    assert not panel.controls['skip'].isHidden()
    for stage in ('convert', 'train'):
        panel.stage.setCurrentIndex(panel.stage.findData(stage))
        assert panel.images.isHidden()
        assert all(button.isHidden() for button in panel.controls.values())
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
    panel=ui.WorkflowPanel(lambda *args:None);panel.save_backend_settings=lambda:None
    panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
    panel.backend.setCurrentIndex(panel.backend.findData(backend))
    panel.active_stage='datagen';panel.busy=True
    root=session(tmp_path,monkeypatch,'datagen');bridge=module.current();panel.service.session=root
    bridge.start_datagen(100,4);bridge.source_progress(1,14,'demo_1',88,440);bridge.status()
    panel.show();app.processEvents();panel.poll_preview()
    assert 'Source 검증 완료 1/14' in panel.task_state.text()
    assert '전체 성공 0/100' in panel.task_state.text() and 'Env 3: 대기' in panel.task_state.text()
    assert 'Source 사전 검증 중' in panel.status.text()
    panel.busy=False;panel.close()
