"""Identical workflow controls, artifact routing and operations for both engines."""
import json
import sys
import pytest


@pytest.fixture
def panel(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui, operations
    from soarm101_lab.workflow.artifacts import Registry
    from soarm101_lab.workflow import operations as service
    from soarm101_lab.simulation.mjcf import load_inputs
    from soarm101_lab.real2sim.storage import atomic
    app = QApplication.instance() or QApplication([])
    registry = Registry(tmp_path / 'registry')
    monkeypatch.setattr(ui, 'Registry', lambda: registry)
    p = ui.WorkflowPanel(lambda *args: None)
    profile = json.loads((operations.ROOT / 'tests/real2sim/fixtures/scene_profile.json').read_text())
    w, _, _ = load_inputs({'profile': profile}); w['root'] = tmp_path
    identity = {'fixture_workspace': 'same_identity'}
    for module in (ui, operations, service):
        monkeypatch.setattr(module, 'load', lambda *args: w)
        monkeypatch.setattr(module, 'metadata', lambda *args: identity)
    monkeypatch.setattr(operations, 'inspect_artifact', lambda *args: {})
    atomic(registry.root / 'artifacts/workspace.json', {
        'id': 'workspace', 'type': 'workspace', 'path': str(tmp_path), 'workspace': identity, 'created_at': 1})
    p.workspace.setText(str(tmp_path)); p.python.setText(sys.executable)
    p.backend_pythons = {'isaac': sys.executable, 'mujoco': sys.executable}
    p.refresh()
    yield p
    p.timer.stop(); p.close()


def test_backend_changes_only_execution_engine(panel):
    from soarm101_lab.workflow.operations import Operations
    before = [panel.stage.itemText(i) for i in range(panel.stage.count())]
    service = panel.service
    path, port = panel.workspace.text(), panel.port.text()
    for backend in ('mujoco', 'isaac'):
        panel.backend.setCurrentIndex(panel.backend.findData(backend))
        assert panel.service is service and isinstance(service, Operations)
        assert [panel.stage.itemText(i) for i in range(panel.stage.count())] == before
        assert panel.workspace.text() == path and panel.port.text() == port
        from PyQt6.QtTest import QTest
        for _ in range(200):
            if panel.run.isEnabled():break
            QTest.qWait(10)
        assert panel.run.isEnabled(), panel.status.text()
        assert panel.inputs.currentData() == 'workspace'
        assert panel.stage_options()['backend'] == backend


def test_object_selection_persisted_per_workspace_and_not_overwritten_by_replay(panel,monkeypatch):
    import copy
    from PyQt6.QtCore import Qt
    from soarm101_lab.workflow.tasks import workspace_key
    from soarm101_lab.workflow import ui
    w=ui.load(panel.workspace.text())
    selector=panel.task_selector
    selector.configure(w,'source',active_object_ids=['cube_red'])
    first=json.dumps(workspace_key(w),sort_keys=True)
    assert panel.object_selections[first]==['cube_red']
    selector.configure(w,'replay',selector.selection(),w['task']['dynamic'])
    assert panel.object_selections[first]==['cube_red']
    other=copy.deepcopy(w);other['profile']['objects']['cube_red']['display_name']='New workspace red'
    selector.configure(other,'source')
    second=json.dumps(workspace_key(other),sort_keys=True)
    assert len(panel.object_selections[second])==3 and len(panel.object_selections[first])==1
    monkeypatch.setattr(ui,'load',lambda *args:w)
    panel.sync_tasks()
    assert selector.object_selection()==['cube_red']
    saved=json.loads((panel.registry.root/'ui.json').read_text())
    assert saved['object_selections'][first]==['cube_red']
    selector.objects.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert not panel.run.isEnabled() and 'least one' in panel.status.text()


@pytest.mark.parametrize('stage', ['annotate', 'datagen', 'convert', 'train', 'sim_eval', 'hf_dataset', 'hf_policy'])
def test_stage_controls_do_not_change_with_backend(panel, stage):
    panel.stage.setCurrentIndex(panel.stage.findData(stage))
    def state():
        return (panel.run.text(), panel.stage_summary.text(), panel.architecture.isHidden(),
                panel.annotation_mode.isHidden(), set(panel.options.editors),
                {k for k, v in panel.controls.items() if not v.isHidden()})
    before = state()
    panel.backend.setCurrentIndex(panel.backend.findData('mujoco'))
    assert state() == before


def test_source_completion_does_not_start_another_stage(panel, tmp_path):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.real2sim.storage import atomic
    calls = []
    panel.run_worker = lambda operation, done: calls.append((operation, done))
    panel.execute()
    source_path = tmp_path / 'source.hdf5'; source_path.touch()
    source = {'id': 'source', 'type': 'source', 'path': str(source_path), 'created_at': 2,
              'status': 'SUCCEEDED', 'tasks': panel.task_selector.selection()}
    atomic(panel.registry.root / 'artifacts/source.json', source)
    calls[0][1]({'ok': True, 'artifact': source})
    QApplication.processEvents()
    assert len(calls) == 1 and panel.active_stage == 'source'
    assert not panel.busy


def test_stage_options_survive_navigation(panel):
    panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
    values = json.loads(panel.options.toPlainText()); values['num_successful_demos'] = 7
    panel.options.setPlainText(json.dumps(values))
    panel.stage.setCurrentIndex(panel.stage.findData('source'))
    panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
    assert panel.stage_options()['num_successful_demos'] == 7


def test_model_specific_options_are_isolated_and_preserved(panel):
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    panel.options.editors['batch_size'][1].setValue(3)
    assert 'model_config' not in panel.options.editors
    assert 'source_groups' not in panel.options.editors
    panel.architecture.setCurrentIndex(panel.architecture.findData('minivla'))
    assert set(panel.options.editors) == {'model_config','batch_size','steps','val_every','save_freq','early_stopping','patience','legacy_group_split','output'}
    assert not hasattr(panel, "minivla_head")
    assert panel.stage_options()['batch_size']==8
    opts = panel.stage_options()
    from pathlib import Path
    assert json.loads(Path(opts['model_config']).read_text())['layers'] == 4
    panel.options.editors['steps'][1].setValue(12345)
    panel.architecture.setCurrentIndex(panel.architecture.findData('smolvla'))
    assert panel.stage_options()['batch_size'] == 3
    assert 'model_config' not in panel.stage_options()
    assert not hasattr(panel, "minivla_head")
    panel.architecture.setCurrentIndex(panel.architecture.findData('minivla'))
    assert panel.stage_options()['steps'] == 12345
    panel.stage.setCurrentIndex(panel.stage.findData('source'))
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    assert panel.stage_options()['steps'] == 12345


def test_minivla_separate_validation_selection_does_not_affect_other_trainers(panel):
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    panel.architecture.setCurrentIndex(panel.architecture.findData('minivla'))
    panel.validation_input.addItem('Verified Val Source 6','fixture_validation_dataset')
    panel.validation_input.setCurrentIndex(1)
    assert panel.stage_options()['validation_dataset']=='fixture_validation_dataset'
    assert 'validation_dataset' not in panel.options.editors
    panel.architecture.setCurrentIndex(panel.architecture.findData('act'))
    assert 'validation_dataset' not in panel.stage_options()


def test_minivla_custom_json_overrides_head_preset_and_drops_foreign_options(panel):
    from soarm101_lab.learning.minivla.config import MODEL_ROOT
    panel.stage.setCurrentIndex(panel.stage.findData('train'))
    panel.architecture.setCurrentIndex(panel.architecture.findData('minivla'))
    config = str(MODEL_ROOT / 'configs/default.json')
    panel.options.setPlainText(json.dumps({'model_config': config, 'batch_size': 999, 'base_model': 'unused'}))
    opts = panel.stage_options()
    assert opts['model_config'] == config and not hasattr(panel, "minivla_head")
    assert opts['batch_size']==999 and 'base_model' not in opts


@pytest.mark.parametrize('backend', ['isaac', 'mujoco'])
def test_source_uses_common_execution_and_keyboard(panel, tmp_path, backend):
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from soarm101_lab.real2sim.storage import atomic
    panel.backend.setCurrentIndex(panel.backend.findData(backend))
    calls = []
    panel.run_worker = lambda operation, done: calls.append((operation, done))
    panel.execute()
    assert panel.busy and not panel.backend.isEnabled() and not panel.navigation.isEnabled()
    root = tmp_path / 'presentation'
    atomic(root / 'status.json', {'state': 'RUNNING', 'frames': {}})
    panel.service.session = root
    panel.show(); panel.activateWindow(); panel.poll_preview()
    panel.controls['save'].setFocus(); QApplication.processEvents()
    QTest.keyClick(panel.controls['save'], Qt.Key.Key_Right)
    QTest.keyClick(panel.controls['save'], Qt.Key.Key_Left)
    QApplication.processEvents()
    assert [json.loads(p.read_text())['command'] for p in sorted((root / 'commands').glob('*.json'))] == ['save', 'discard']
    calls[0][1]({'ok': False, 'error': 'fixture completion'})
    assert not panel.busy and panel.backend.isEnabled()


@pytest.mark.parametrize('backend', ['isaac', 'mujoco'])
@pytest.mark.parametrize('architecture', ['act', 'smolvla', 'minivla'])
def test_lerobot_training_is_independent_of_selected_engine(panel, tmp_path, monkeypatch, backend, architecture):
    from soarm101_lab.workflow import operations
    from soarm101_lab.workflow import episodes
    from soarm101_lab.workflow.tasks import selected
    from soarm101_lab.so101_dataset_contract import JOINT_SPACE
    monkeypatch.setattr(operations, 'require_compatible', lambda *args: None)
    monkeypatch.setattr(operations, 'coordinate_contract', lambda *args: {'action_space': JOINT_SPACE})
    monkeypatch.setattr(panel.registry, 'verify', lambda *args: None)
    # This routing fixture has no dataset bytes; lineage is covered by partition tests.
    monkeypatch.setattr(episodes,'validate_training_inputs',lambda *args:None)
    w = operations.load('fixture')
    artifact = {'id': 'dataset', 'type': 'dataset', 'path': str(tmp_path / 'lerobot'),
        'format': {'codebase_version': 'v3.0'}, 'workspace': operations.metadata(w),
        'tasks': [selected(w)], 'coordinates': {'action_space': JOINT_SPACE}}
    plan = panel.service.plan('train', artifact, str(tmp_path), {
        'backend': backend, 'architecture': architecture, 'python': panel.python.text()}, run_dir=tmp_path)
    assert 'scripts/real2sim/workflow.py' in plan['argv']
    request = json.loads((tmp_path / 'request.json').read_text())
    assert request['options']['architecture'] == architecture
    assert request['options']['workspace'] == str(w['root'])


def test_builder_folder_import_uses_worker_python_and_persists(panel, tmp_path, monkeypatch):
    """GUI-only Python does not need YAML, USD or simulator dependencies."""
    import subprocess
    from types import SimpleNamespace
    from soarm101_lab.workflow import ui
    export = tmp_path/'builder_export'; export.mkdir(); (export/'scene_manifest.yaml').write_text('schema: real2sim.package/1')
    monkeypatch.setattr(ui.QFileDialog, 'getExistingDirectory', lambda *a: str(export))
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda argv, **kw: (calls.append(argv) or SimpleNamespace(returncode=0, stdout=panel.workspace.text()+'\n', stderr='')))
    registrations=[];monkeypatch.setattr(panel.registry,'register',lambda *a: registrations.append(a))
    panel.browse_workspace()
    assert calls[0][0] == panel.python.text()
    assert calls[0][-2:] == ['import-builder',str(export)]
    assert registrations[0][0] == 'workspace'
    saved=json.loads((panel.registry.root/'ui.json').read_text())
    assert saved['workspace'] == panel.workspace.text()


def test_one_camera_scene_configures_pipeline_and_real_sim(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from test_builder_workspace import fruit_export
    from soarm101_lab.real2sim.workspaces.builder import import_scene
    from soarm101_lab.real2sim.ui import Window
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    app=QApplication.instance() or QApplication([])
    export=fruit_export.__wrapped__(tmp_path)
    scene=import_scene(export,registry=tmp_path/'environments')
    registry=Registry(tmp_path/'workflow');registry.register('workspace',scene)
    monkeypatch.setattr(ui,'Registry',lambda:registry)
    window=Window(tmp_path/'live')
    window.workflow_panel.workspace.setText(str(scene));window.workflow_panel.refresh()
    panel=window.workflow_panel
    assert panel.scene_cameras==('front',)
    assert panel.feeds['sim_front'].isHidden() is False
    assert panel.feeds['sim_wrist'].isHidden()
    assert panel.task_selector.selection()[0]['task_id']=='fruit_sort'
    panel.section.setCurrentIndex(1)
    panel.stage.setCurrentIndex(panel.stage.findData('real_eval'))
    assert not panel.feeds['real_front'].isHidden() and panel.feeds['real_wrist'].isHidden()
    window.sync_scene_cameras()
    assert 'front_camera' in window.fields
    assert 'sim_front' in window.views and window.views['sim_wrist'].parentWidget().isHidden()
    window.close();app.processEvents()


def test_settings_validation_is_debounced_off_main_thread_and_latest_only(panel,monkeypatch):
    import threading,time
    from PyQt6.QtCore import QTimer
    from PyQt6.QtTest import QTest
    main_thread=threading.get_ident();started=threading.Event();release=threading.Event();calls=[];ticks=[]
    def readiness(stage,artifact,workspace,options):
        calls.append((options['port'],threading.get_ident()))
        if len(calls)==1:
            started.set();release.wait(3)
            return 'READY','old result'
        return 'NOT_READY','latest setting rejected'
    monkeypatch.setattr(panel.service,'readiness',readiness)
    timer=QTimer(panel);timer.timeout.connect(lambda:ticks.append(time.monotonic()));timer.start(10)
    try:
        before=time.monotonic()
        for text in ['/dev/a','/dev/ab','/dev/abc']:panel.port.setText(text)
        assert time.monotonic()-before<.15
        assert not calls and not panel.run.isEnabled()
        for _ in range(100):
            if started.is_set():break
            QTest.qWait(10)
        assert started.is_set() and len(calls)==1 and calls[0][0]=='/dev/abc'
        panel.port.setText('/dev/new')
        QTest.qWait(250)
        assert len(ticks)>10 and len(calls)==1 and not panel.run.isEnabled()
        release.set()
        for _ in range(200):
            if 'latest setting rejected' in panel.status.text():break
            QTest.qWait(10)
        assert panel.status.text()=='NOT_READY — latest setting rejected'
        assert [c[0] for c in calls]==['/dev/abc','/dev/new']
        assert all(c[1]!=main_thread for c in calls)
        assert not panel.run.isEnabled()
    finally:release.set();timer.stop()


def test_background_ready_result_cannot_enable_run_during_workflow(panel,monkeypatch):
    import threading
    from PyQt6.QtTest import QTest
    started=threading.Event();release=threading.Event()
    def readiness(*args):
        started.set();release.wait(3);return 'READY',''
    monkeypatch.setattr(panel.service,'readiness',readiness)
    panel.update_ready()
    try:
        for _ in range(100):
            if started.is_set():break
            QTest.qWait(10)
        assert started.is_set()
        panel.busy=True;panel.update_ready();release.set()
        QTest.qWait(150)
        assert panel.status.text().startswith('RUNNING') and not panel.run.isEnabled()
    finally:release.set();panel.busy=False
