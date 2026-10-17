"""Reusable stage assembly; task-specific options belong to concrete screens."""
import json
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFormLayout, QLabel, QMessageBox, QPushButton, QComboBox, QSizePolicy
from ...settings_form import SettingsForm
from ...option_help import template
from ...stages import feeds
from ...task_ui import TaskSelector
from ...tasks import from_artifact, workspace_key, recorded_scene
from ....real2sim.workspaces.package import load
from ..artifact_picker import ArtifactPicker
from ..settings import OutputSettings
from ..monitor import RunMonitor
from ..preview import CameraPreview
from ..context import InheritedTaskInfo


class StageScreen(QWidget):
    navigate = pyqtSignal(str, object)
    basic_keys = set()
    action = '실행'

    def __init__(self, stage, key, context, controller, parent=None):
        super().__init__(parent)
        self.stage, self.key = stage, key
        self.context, self.controller, self.registry = context, controller, context.registry
        self.initializing = True
        self.updating = False
        self.loaded = False
        self.active = False
        self.validation_candidates = ()
        outer = QHBoxLayout(self)
        self.monitor = RunMonitor(stage)
        self.monitor.run.setText(self.action)
        outer.addWidget(self.monitor, 3)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumWidth(390)
        self.settings_panel = QWidget()
        self.scroll.setWidget(self.settings_panel)
        self.settings_layout = QVBoxLayout(self.settings_panel)
        outer.addWidget(self.scroll, 1)
        self.picker = ArtifactPicker(stage, context)
        self.inputs = self.picker.combo
        self.settings_layout.addWidget(self.picker)
        self.task_selector = None
        if stage in ('source', 'replay', 'annotate', 'datagen', 'sim_eval', 'real_eval'):
            self.task_selector = InheritedTaskInfo() if stage in ('replay', 'annotate', 'datagen') else TaskSelector(compact=True)
            if isinstance(self.task_selector, TaskSelector):
                self.task_selector.layout().setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
            self.settings_layout.addWidget(self.task_selector)
            self.task_selector.changed.connect(self.objects_changed)
        self.form_widget = QWidget()
        self.form = QFormLayout(self.form_widget)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.settings_layout.addWidget(self.form_widget)
        self.build_inputs()
        self.output = OutputSettings(self)
        self.settings_layout.addWidget(self.output)
        self.options = SettingsForm()
        self.options.basic_keys = self.basic_keys
        self.options.basic_order = getattr(self, 'basic_order', ())
        self.options.set_stage(stage)
        self.configure_options()
        self.settings_layout.addWidget(self.options)
        if self.stage in ('datagen', 'train'):
            self.settings_layout.removeWidget(self.output)
            self.settings_layout.addWidget(self.output)
        reset = QPushButton('설정을 기본값으로 복원')
        reset.clicked.connect(self.reset_options)
        self.settings_layout.addWidget(reset)
        self.settings_layout.addStretch()
        self.preview = CameraPreview(stage) if feeds(stage) else None
        if self.preview:
            self.monitor.media_layout.addWidget(self.preview)
        self.picker.changed.connect(self.input_changed)
        self.inputs.activated.connect(self.activate_input)
        self.picker.error.connect(self.show_error)
        self.options.textChanged.connect(self.update_ready)
        self.monitor.run.clicked.connect(lambda: self.guard(self.execute))
        self.monitor.stop.clicked.connect(controller.cancel)
        controller.readinessChanged.connect(self.readiness_changed)
        controller.started.connect(self.started)
        controller.finished.connect(self.finished)
        for combo in self.settings_panel.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        for label in self.settings_panel.findChildren(QLabel):
            if label.wordWrap():
                label.setMinimumWidth(0)
                label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.initializing = False

    def build_inputs(self):
        pass

    def configure_options(self):
        self.options.setPlainText(json.dumps(template(self.stage)))

    def selected_record(self, required=True):
        if required:
            return self.picker.record()
        return self.registry.get(self.inputs.currentData()) if self.inputs.currentData() else None

    def activate(self):
        self.loaded = True
        self.active = True
        self.refresh()

    def deactivate(self):
        self.active = False

    def refresh(self):
        self.updating = True
        self.picker.refresh()
        self.updating = False
        self.input_changed()

    def input_changed(self):
        if self.initializing or self.updating:
            return
        self.sync_tasks()
        self.update_ready()

    def activate_input(self):
        record = self.selected_record(False)
        if record and record['type'] == 'workspace' and not self.controller.busy:
            self.context.select_workspace(record['path'])

    def sync_tasks(self):
        if self.task_selector is None:
            return
        self.updating = True
        try:
            workspace = load(self.context.workspace.text())
            self.scene_cameras = tuple(workspace['profile']['cameras'])
            if self.preview:
                self.preview.configure(self.scene_cameras)
            record = self.selected_record(False) or {}
            selected = self.context.object_selections.get(json.dumps(workspace_key(workspace), sort_keys=True))
            if self.stage in ('replay', 'annotate', 'datagen'):
                scene = recorded_scene(record, workspace, required=False)
                if scene:
                    selected = scene['active_object_ids']
            self.task_selector.configure(workspace, self.stage, from_artifact(record, workspace), selected)
            builder = workspace.get('manifest', {}).get('builder')
            if builder:
                self.task_selector.summary.setText(self.task_selector.summary.text() + f"\nBuilder scene: {builder['status']} · 물성: 시뮬레이션 임시값 (실측 아님)")
        except (ValueError, OSError, TypeError, KeyError):
            self.task_selector.workspace = None
            self.task_selector.summary.setText('유효한 Workspace와 입력 Artifact를 선택하세요.')
        finally:
            self.updating = False

    def objects_changed(self):
        if self.initializing or self.updating or self.task_selector is None:
            return
        selector = self.task_selector
        if not selector.loading and selector.workspace is not None and selector.stage in ('source', 'sim_eval'):
            self.context.object_selections[json.dumps(selector.workspace_identity, sort_keys=True)] = [
                selector.objects.item(i).data(Qt.ItemDataRole.UserRole) for i in range(selector.objects.count())
                if selector.objects.item(i).checkState() == Qt.CheckState.Checked]
            self.context.save()
        self.update_ready()

    def collect_options(self):
        opts = json.loads(self.options.toPlainText())
        if not isinstance(opts, dict):
            raise ValueError('실행 설정은 JSON 객체여야 합니다.')
        shared = self.context.snapshot()
        opts.update({key: shared[key] for key in ('backend', 'python', 'mode')})
        if self.task_selector is not None and self.task_selector.workspace is not None:
            if self.stage != 'real_eval':
                opts['active_object_ids'] = self.task_selector.object_selection()
            if self.stage in ('source', 'sim_eval', 'real_eval'):
                opts['tasks'] = self.task_selector.selection(evaluation=self.stage != 'source')
                if self.stage != 'source':
                    opts['task_scope'] = self.task_selector.scope.currentData()
        return opts

    def stage_options(self):
        return self.output.options(self.collect_options())

    def snapshot(self):
        return dict(stage=self.stage, artifact=self.selected_record(), workspace=self.context.workspace.text(),
                    options=self.stage_options())

    def update_ready(self):
        if self.initializing or self.updating or not self.active:
            return
        if self.controller.busy:
            self.monitor.run.setEnabled(False)
            return
        try:
            snapshot = self.snapshot()
            self.output.update_preview(snapshot['options'])
            live = getattr(self.window(), 'bus', None)
            if self.stage in ('source', 'real_eval') and live and live.owner_alive() and live.status().get('backend') == 'calibration':
                raise ValueError('Precision Calibration owns the motor ports; End Calibration Session first')
        except Exception as exc:
            try:
                self.output.update_preview(self.stage_options())
            except (ValueError, OSError, KeyError):
                pass
            self.controller.request_ready(self.key, error=exc)
            return
        self.controller.request_ready(self.key, snapshot, validation_candidates=self.validation_candidates)

    def readiness_changed(self, owner, state, reason):
        if owner == self.key:
            self.monitor.set_readiness(state, reason)

    def execute(self):
        live = getattr(self.window(), 'bus', None)
        if self.stage in ('source', 'replay', 'annotate', 'datagen', 'sim_eval', 'real_eval') and live and live.owner_alive():
            raise ValueError('Real/Sim을 Disconnect한 뒤 Workflow를 실행하세요. 장비·카메라 중복 연결을 방지합니다.')
        self.controller.execute(self.key, self.snapshot())

    def started(self, active):
        self.settings_panel.setEnabled(False)
        if active['owner'] == self.key:
            self.monitor.begin()

    def finished(self, active, result):
        self.settings_panel.setEnabled(True)
        if active['owner'] != self.key:
            return
        self.monitor.poll(self.controller.service.session)
        artifact = result.get('artifact', {})
        self.monitor.finish(result, self.controller.service.cancelled or artifact.get('status') == 'STOPPED')
        self.refresh()

    def poll(self):
        active = self.controller.active
        session = self.controller.service.session if active and active['owner'] == self.key else None
        status = self.monitor.poll(session)
        if self.preview:
            self.preview.update_frames(self.monitor.session, status, self.monitor.running)

    def reset_options(self):
        self.configure_options()
        self.update_ready()

    def guard(self, fn):
        try:
            return fn()
        except Exception as exc:
            self.show_error(str(exc))

    def show_error(self, message):
        QMessageBox.warning(self, 'Workflow', message)
