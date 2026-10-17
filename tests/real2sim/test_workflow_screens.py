"""Task ownership and run isolation; no simulator, network, GPU training or motors."""
import copy
import json
import threading
from pathlib import Path
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QMessageBox
from soarm101_lab.workflow.artifacts import Registry
from soarm101_lab.real2sim.storage import atomic
from test_mujoco_workflow_ui import panel


def wait_for(predicate):
    for _ in range(300):
        if predicate():
            return
        QTest.qWait(10)
    assert predicate()


def add_record(panel, identifier, kind, **extra):
    path = panel.registry.root/identifier
    path.mkdir(exist_ok=True)
    record = dict(id=identifier, type=kind, path=str(path), created_at=len(panel.registry.all())+1,
                  status='SUCCEEDED', parents=[], display_name='사용자_'+identifier, **extra)
    atomic(panel.registry.root/'artifacts'/f'{identifier}.json', record)
    return record


def test_five_navigation_groups_and_all_operations_accessible(panel):
    from soarm101_lab.workflow.operations import STAGES
    from soarm101_lab.workflow.ui_components.navigation import GROUPS
    assert panel.navigation.topLevelItemCount() == 5
    visited = set()
    for _, entries in GROUPS:
        for key, _ in entries:
            panel.select_stage(key)
            if panel.current_stage():
                visited.add(panel.current_stage())
    assert visited == set(STAGES)
    assert not any(hasattr(panel, name) for name in ('real_enabled', 'real_side', 'real_wrist', 'options_help', 'section', 'stage'))


@pytest.mark.parametrize('stage,commands', [
    ('source', {'save','discard','quit'}), ('replay', {'resume','pause','step'}),
    ('annotate', {'resume','pause'}), ('datagen', {'resume','pause','step'}),
    ('train:act', set()), ('train:smolvla', set()), ('train:minivla', set()),
    ('convert', set()), ('hf_dataset', set()), ('hf_policy', set()), ('real_eval', {'resume','pause'})])
def test_each_screen_only_constructs_its_controls(panel, stage, commands):
    panel.select_stage(stage)
    screen = panel.current_screen
    assert set(screen.monitor.controls) == commands
    if stage.startswith('train') or stage in ('convert','hf_dataset','hf_policy'):
        assert screen.preview is None
        from soarm101_lab.workflow.view import CameraView
        assert screen.findChildren(CameraView) == []
    if not stage.startswith('train'):
        assert not hasattr(screen, 'loss_button') and not hasattr(screen, 'resume_training')
    if stage == 'annotate':
        screen.annotation_mode.setCurrentIndex(1)
        assert set(screen.monitor.controls) == {'resume','pause','mark','skip'}


def test_screen_settings_have_one_owner_and_collapsed_advanced(panel):
    from soarm101_lab.workflow.option_help import KEYS, TRAIN_KEYS
    for stage in ('source','replay','annotate','datagen','convert','train:act','train:smolvla','train:minivla','sim_eval','real_eval','hf_dataset','hf_policy'):
        panel.select_stage(stage)
        screen = panel.current_screen
        expected = TRAIN_KEYS[screen.model] if stage.startswith('train:') else set(KEYS[stage])
        assert set(screen.options.editors) == expected
        assert screen.options.advanced_panel.isHidden()
        assert screen.options.advanced_toggle.isChecked() is False
        assert set(screen.options.editors) >= screen.basic_keys


def test_input_selection_is_not_replaced_by_filter_backend_or_missing_file(panel):
    a = add_record(panel, 'old-source', 'source', backend='isaac')
    b = add_record(panel, 'new-source', 'source', backend='isaac')
    panel.select_stage('replay', artifact_id=a['id'])
    picker = panel.current_screen.picker
    picker.search.setText('new-source')
    assert picker.combo.currentData() is None and picker.preferred_id == a['id']
    assert b['id'] in picker.combo.itemData(0)
    picker.search.clear()
    assert picker.combo.currentData() == a['id']
    panel.backend.setCurrentIndex(panel.backend.findData('mujoco'))
    assert picker.combo.currentData() is None and picker.preferred_id == a['id']
    panel.backend.setCurrentIndex(panel.backend.findData('isaac'))
    Path(a['path']).rmdir()
    picker.refresh()
    assert picker.combo.currentData() is None
    with pytest.raises(ValueError, match='호환'):
        picker.select(a['id'])


def test_execution_freezes_context_snapshot_rejects_double_run_and_never_autoruns(panel, monkeypatch):
    callbacks, calls = [], []
    panel.run_worker = lambda op, done: callbacks.append((op, done))
    monkeypatch.setattr(panel.service, 'run', lambda stage, artifact, workspace, opts, notify: calls.append(copy.deepcopy(opts)) or artifact)
    screen = panel.current_screen
    original = panel.context.snapshot()
    original_port = screen.port.text()
    screen.execute()
    assert panel.busy and not screen.settings_panel.isEnabled() and not panel.navigation.isEnabled()
    with pytest.raises(ValueError, match='이미'):
        screen.execute()
    with pytest.raises(ValueError, match='작업 화면'):
        panel.select_stage('datagen')
    with pytest.raises(ValueError, match='Scene'):
        panel.select_workspace('invalid-scene')
    panel.backend.setCurrentIndex(panel.backend.findData('mujoco'))
    panel.workspace.setText('changed-by-another-caller')
    panel.python.setText('changed-python')
    assert panel.context.snapshot() == original
    screen.port.setText('programmatic-edit-after-run')
    result = callbacks[0][0](lambda message: None)
    callbacks[0][1](result)
    assert calls[0]['port'] == original_port
    assert len(callbacks) == 1 and not panel.busy and screen.monitor.status.text() == 'SUCCEEDED'
    assert panel.current_stage() == 'source'


def test_stop_before_worker_start_does_not_launch_operation(panel, monkeypatch):
    callbacks, runs = [], []
    panel.run_worker = lambda operation, done: callbacks.append((operation, done))
    monkeypatch.setattr(panel.service, 'run', lambda *args: runs.append(args))
    panel.current_screen.execute()
    panel.controller.cancel()
    result = callbacks[0][0](lambda message: None)
    callbacks[0][1](result)
    assert not runs and not panel.busy
    assert panel.current_screen.monitor.status.text() == 'STOPPED'


@pytest.mark.parametrize('backend', ['isaac','mujoco'])
def test_datagen_ik_options_reach_existing_service_snapshot(panel, monkeypatch, backend):
    record = add_record(panel, 'annotated', 'annotated', format='so101.common-episode/2')
    panel.backend.setCurrentIndex(panel.backend.findData(backend))
    panel.select_stage('datagen', artifact_id=record['id'])
    screen = panel.current_screen
    screen.options.setPlainText(json.dumps(dict(num_envs=4,num_successful_demos=100,ik_solver='pinocchio_dls',ik_failure_policy='best_effort')))
    captured = []
    monkeypatch.setattr(panel.service, 'run', lambda stage, artifact, workspace, opts, notify: captured.append(opts) or record)
    panel.run_worker = lambda operation, done: done(operation(lambda message: None))
    screen.execute()
    assert captured[0]['backend'] == backend and captured[0]['num_envs'] == 4
    assert captured[0]['ik_solver'] == 'pinocchio_dls' and captured[0]['ik_failure_policy'] == 'best_effort'
    assert captured[0]['num_successful_demos'] == 100


def test_old_readiness_and_logs_cannot_cross_screen_or_run(panel, tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    def check(*args):
        started.set()
        release.wait(3)
        return 'READY', 'old source validation'
    monkeypatch.setattr(panel.service, 'readiness', check)
    source = panel.current_screen
    source.update_ready()
    try:
        wait_for(started.is_set)
        panel.select_stage('convert')
        release.set()
        QTest.qWait(100)
        assert 'old source' not in panel.current_screen.monitor.status.text()
        assert not panel.current_screen.monitor.run.isEnabled()
    finally:
        release.set()
    root = tmp_path/'old-run/presentation'
    atomic(root/'status.json', {'state':'FINISHED','frames':{}})
    (root.parent/'process.log').write_text('[WORKFLOW] old run only\n')
    source.monitor.session = root
    source.monitor.poll()
    panel.service.session = root
    panel.controller.active = dict(owner='source',stage='source')
    panel.poll_preview()
    assert 'old run only' in source.monitor.result.toPlainText()
    assert 'old run only' not in panel.current_screen.monitor.result.toPlainText()
    assert panel.current_screen.monitor.session is None


def test_validation_compatibility_is_background_and_never_changes_train_selection(panel, monkeypatch):
    from soarm101_lab.workflow import episodes
    train = add_record(panel, 'train', 'dataset', format={}, provenance={'source_partition':{'split':'train','source_ids':['a']}})
    valid = add_record(panel, 'val-good', 'dataset', format={}, provenance={'source_partition':{'split':'val','source_ids':['b']}})
    invalid = add_record(panel, 'val-bad', 'dataset', format={}, provenance={'source_partition':{'split':'val','source_ids':['a']}})
    main_thread = threading.get_ident()
    checks = []
    def validate(artifact, options):
        checks.append(threading.get_ident())
        if options['validation_artifact']['id'] == invalid['id']:
            raise ValueError('Source leakage')
    monkeypatch.setattr(episodes,'validate_training_inputs',validate)
    monkeypatch.setattr(panel.service,'readiness',lambda stage, artifact, workspace, opts: ('NOT_READY','별도 Validation 선택 필요') if not opts.get('validation_dataset') else ('READY',''))
    panel.select_stage('train:minivla', artifact_id=train['id'])
    screen = panel.current_screen
    wait_for(lambda: screen.validation_input.findData(valid['id']) >= 0)
    assert checks and all(identifier != main_thread for identifier in checks)
    assert screen.validation_input.findData(invalid['id']) < 0
    assert screen.inputs.currentData() == train['id'] and not screen.monitor.run.isEnabled()
    screen.validation_input.setCurrentIndex(screen.validation_input.findData(valid['id']))
    wait_for(screen.monitor.run.isEnabled)
    assert screen.stage_options()['validation_dataset'] == valid['id']
    assert screen.inputs.currentData() == train['id']


def test_real_motion_and_publication_require_explicit_confirmation(panel, monkeypatch):
    panel.select_stage('real_eval')
    screen = panel.current_screen
    assert not screen.stage_options()['enable_motion']
    snapshot = dict(stage='real_eval',artifact={'id':'policy'},workspace='fixture',options={'enable_motion':False})
    monkeypatch.setattr(screen, 'snapshot', lambda: copy.deepcopy(snapshot))
    with pytest.raises(ValueError, match='enable_motion'):
        screen.execute()
    snapshot['options'].update(enable_motion=True,port='fake',follower_id='identity')
    asked, calls = [], []
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: asked.append(args) or QMessageBox.StandardButton.No)
    monkeypatch.setattr(panel.controller, 'execute', lambda *args: calls.append(args))
    screen.execute()
    assert asked and not calls
    panel.select_stage('hf_dataset')
    screen = panel.current_screen
    monkeypatch.setattr(screen, 'snapshot', lambda: dict(stage='hf_dataset',artifact={'path':'fixture'},options={'repo_id':'example/test','private':True}))
    screen.execute()
    assert len(asked) == 2 and not calls


def test_registry_explorer_retains_identity_lineage_and_exact_checkpoint_navigation(panel):
    source = add_record(panel,'source-parent','source',source_partition={'split':'train','source_ids':['a']})
    policy = add_record(panel,'policy-child','policy',format='so101.minivla-policy/1')
    policy['parents'] = [source['id']]
    atomic(panel.registry.root/'artifacts/policy-child.json',policy)
    root = Path(policy['path'])
    (root/'best.pt').write_bytes(b'fixture only; never loaded')
    (root/'last.pt').write_bytes(b'fixture only; never loaded')
    (root/'checkpoints').mkdir()
    chosen = root/'checkpoints/step_000008.pt'
    chosen.write_bytes(b'fixture only; never loaded')
    before = {path: path.read_bytes() for path in (panel.registry.root/'artifacts').glob('*.json')}
    panel.select_stage('artifacts:checkpoint')
    explorer = panel.current_screen
    item = explorer.tree.topLevelItem(0)
    assert item.childCount() == 3
    explorer.tree.setCurrentItem(item.child(2))
    assert 'step_000008.pt' in explorer.details.toPlainText()
    explorer.action('lineage')
    assert source['id'] in explorer.details.toPlainText()
    explorer.action('sim_eval')
    assert panel.current_screen.inputs.currentData() == policy['id']
    assert panel.current_screen.stage_options()['checkpoint_file'] == str(chosen)
    assert {path:path.read_bytes() for path in before} == before
    assert len(panel.registry.all()) == len(before)
    panel.select_stage('artifacts:source')
    explorer = panel.current_screen
    explorer.filters['partition'].setText('val')
    assert explorer.tree.topLevelItemCount() == 0
    explorer.filters['partition'].setText('train')
    assert explorer.tree.topLevelItemCount() == 1
