"""Real/Sim settings share the existing Workflow and Capture state, without motion."""
import json
import time
from unittest.mock import Mock

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture
def scenes(tmp_path, monkeypatch):
    from test_builder_workspace import make_export, fruit_export
    from soarm101_lab.real2sim.workspaces.builder import import_scene
    from soarm101_lab.workflow import ui
    from soarm101_lab.real2sim.ui import Window
    registry=tmp_path/'workflow'
    blocks=tmp_path/'blocks';blocks.mkdir()
    fruit=tmp_path/'fruit';fruit.mkdir()
    first=import_scene(make_export(blocks),'blocks',registry=tmp_path/'outputs/real2sim/environments')
    second=import_scene(fruit_export.__wrapped__(fruit),'fruit',registry=tmp_path/'outputs/real2sim/environments')
    artifacts=[{'path':str(path)} for path in (first,second)]
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    import soarm101_lab.real2sim.ui as desktop
    monkeypatch.setattr(desktop,'ROOT',tmp_path)
    app=QApplication.instance() or QApplication([])
    window=Window(tmp_path/'live')
    window.runtime_launcher.connect=Mock(return_value=None)
    window.runtime_launcher.disconnect=Mock(return_value=None)
    window.bus.send=Mock()
    window.show();window.tabs.setCurrentIndex(1);app.processEvents()
    yield app,window,first,second,artifacts
    window.runtime_enabled=False;window.timer.stop();window.workflow_panel.timer.stop()
    window.close();app.processEvents()


def test_scene_selection_shares_workflow_camera_ids_and_task(scenes):
    app,w,first,second,artifacts=scenes
    w.live_scene.setCurrentIndex(w.live_scene.findData(str(first)))
    from test_episode_path_ui import until
    until(lambda:w.workflow_panel.task_selector.workspace is not None and w.workflow_panel.workspace_path==str(first))
    assert w.workflow_panel.workspace_path==str(first)
    assert w.workflow_panel.inputs.currentData()==str(first)
    assert len(w.workflow_panel.task_selector.definitions)==6
    assert not w.live_tiles['sim_wrist'].isHidden()
    # Source dropdown can select another Workspace despite the old Scene filter.
    panel=w.workflow_panel
    panel.input_folders.pop('source',None);panel.refresh()
    until(lambda:panel.inputs.findData(str(second))>=0)
    index=panel.inputs.findData(str(second));assert index>=0
    panel.inputs.setCurrentIndex(index)
    until(lambda:panel.workspace_path==str(second) and panel.scene_cameras==('front',))
    assert panel.workspace_path==str(second)
    assert w.live_scene.currentData()==str(second)
    assert panel.scene_cameras==('front',)
    assert w.live_tiles['sim_wrist'].isHidden()
    assert not w.live_tiles['sim_front'].isHidden()
    assert panel.task_selector.selection()[0]['task_id']=='fruit_sort'
    assert 'front_camera' in w.fields
    saved=json.loads((panel.root/'ui.json').read_text())
    assert saved['workspace']==str(second)
    w.runtime_launcher.connect.assert_not_called()
    w.runtime_launcher.disconnect.assert_not_called();w.bus.send.assert_not_called()


def test_settings_visible_without_capture_tab_and_connect_payload(scenes):
    app,w,first,second,_=scenes
    w.live_scene.setCurrentIndex(w.live_scene.findData(str(second)))
    from test_episode_path_ui import until
    until(lambda:w.workflow_panel.workspace_path==str(second) and w.workflow_panel.scene_cameras==('front',))
    assert w.scene_controls.isVisible() and w.scene_options.isVisible()
    assert w.use_precision.isVisible() and w.identity_sim.isVisible()
    assert w.capture_options.isVisible()
    assert w.use_precision.parentWidget()==w.scene_options
    w.use_precision.setChecked(True);w.identity_sim.setChecked(False)
    for tab in (0,1,3,1):w.tabs.setCurrentIndex(tab)
    assert w.use_precision.isChecked() and not w.identity_sim.isChecked()
    w.runtime_launcher.connect.assert_not_called();w.bus.send.assert_not_called()
    for backend in ('isaac','mujoco'):
        w.connect_hw(preview=True,backend=backend)
        payload=w.runtime_launcher.connect.call_args.args[0]
        assert payload['workspace']==str(second)
        assert payload['precision_options']=={'use_precision':True,'identity_sim_for_validation':False}
        assert payload['task_definition']['task_id']=='fruit_sort'
        assert payload['preview_only'] is True
        assert w.runtime_launcher.connect.call_args.kwargs['backend']==backend
    w.bus.send.assert_not_called()


def test_active_scene_not_silently_replaced_by_selection(scenes,monkeypatch):
    app,w,first,second,_=scenes
    w.live_scene.setCurrentIndex(w.live_scene.findData(str(first)))
    w.runtime_enabled=True;w.runtime_session='connected'
    state={'session':'connected','timestamp':time.time(),'backend':'mujoco','state':'CONNECTED',
           'workspace':str(first),'sim_connected':True,
           'precision_run_configuration':{'use_precision':True,'identity_sim_for_validation':False}}
    monkeypatch.setattr(w.bus,'status',lambda:state)
    w.live_scene.setCurrentIndex(w.live_scene.findData(str(second)))
    w.identity_sim.setChecked(True);w.refresh()
    assert w.workflow_panel.workspace_path==str(second)
    assert str(first) in w.live_details.text()
    assert 'Scene 관절 보정 제외: OFF' in w.live_details.text()
    assert 'Disconnect → Connect' in w.scene_selection_hint.text()
    assert not w.live_tiles['sim_wrist'].isHidden()  # Still displaying the connected cameras.
    w.runtime_launcher.connect.assert_not_called();w.runtime_launcher.disconnect.assert_not_called()
    w.bus.send.assert_not_called()


def test_shared_choices_survive_restart(scenes):
    app,w,_,second,_=scenes
    from soarm101_lab.real2sim.ui import Window
    w.live_scene.setCurrentIndex(w.live_scene.findData(str(second)))
    w.use_precision.setChecked(True);w.identity_sim.setChecked(False);w.live_workspace.setChecked(True)
    restored=Window(w.workspace)
    try:
        assert restored.live_scene.currentData()==str(second)
        assert restored.use_precision.isChecked() and not restored.identity_sim.isChecked()
        assert restored.live_workspace.isChecked()
        assert not restored.runtime_enabled and restored.runtime_launcher.process is None
    finally:restored.timer.stop();restored.workflow_panel.timer.stop();restored.close()


def test_scene_controls_locked_during_workflow(scenes):
    app,w,_,second,_=scenes
    w.workflow_panel.busy=True;w.refresh()
    assert not w.live_scene.isEnabled() and not w.scene_browse.isEnabled()
    assert not w.use_precision.isEnabled()
    with pytest.raises(ValueError,match='Workflow'):w.workflow_panel.select_workspace(second)
    w.runtime_launcher.connect.assert_not_called();w.bus.send.assert_not_called()
    w.workflow_panel.busy=False
