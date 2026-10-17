"""Desktop integration and task-screen lifecycle; Operations remains the execution API."""
import subprocess
from pathlib import Path
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QStackedWidget, QFileDialog, QMessageBox, QToolButton
from .artifacts import Registry, browse_directory
from .operations import Operations
from ..real2sim.workspaces.package import ROOT
from .ui_components.context import WorkflowContext
from .ui_components.execution import WorkflowExecutionController
from .ui_components.navigation import WorkflowNavigation
from .ui_components.training_metrics import TrainingLossPlot  # Public plot import remains compatible.

LABELS = dict(source='Source Demo', replay='Replay', annotate='Annotation', datagen='DataGen',
              convert='LeRobot 변환', train='모델 학습', sim_eval='Sim Rollout', real_eval='Real Rollout',
              hf_dataset='Dataset 업로드', hf_policy='Policy 업로드', curation='Data Curation', artifacts='결과물 관리')


class WorkflowPanel(QWidget):
    def __init__(self, run_worker, parent=None):
        super().__init__(parent)
        from .theme import STYLE
        self.setStyleSheet(STYLE)
        self.run_worker = run_worker
        self.registry = Registry()
        self.service = Operations(self.registry)
        self.context = WorkflowContext(self.registry, self)
        self.controller = WorkflowExecutionController(self.service, self.context,
            lambda operation, done: self.run_worker(operation, done), self)
        self.screens = {}
        self.current_screen = None
        self.current_key = None
        self.last_training = 'smolvla'
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.dashboard_title = QLabel('Source Demo')
        self.dashboard_title.setObjectName('dashboardTitle')
        header.addWidget(self.dashboard_title, 1)
        environment = QToolButton()
        environment.setText('고급 실행 환경 · Python / Mode')
        environment.setCheckable(True)
        header.addWidget(environment)
        layout.addLayout(header)
        layout.addWidget(self.context.widget)
        environment.toggled.connect(self.context.advanced.setVisible)
        body = QHBoxLayout()
        layout.addLayout(body, 1)
        self.navigation = WorkflowNavigation()
        body.addWidget(self.navigation)
        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)
        self.navigation.selected.connect(lambda key: self.guard(lambda: self.select_stage(key)))
        self.context.browseRequested.connect(lambda: self.guard(self.browse_workspace))
        self.context.changed.connect(self.refresh)
        self.controller.busyChanged.connect(self._busy_changed)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_preview)
        self.timer.start(33)
        self.select_stage('source')

    @property
    def workspace(self): return self.context.workspace
    @property
    def python(self): return self.context.python
    @property
    def mode(self): return self.context.mode
    @property
    def backend(self): return self.context.backend
    @property
    def busy(self): return self.controller.busy
    @property
    def task_selector(self):
        selector = getattr(self.current_screen, 'task_selector', None)
        return selector if selector is not None else self.screens['source'].task_selector

    def current_stage(self):
        return self.current_screen.stage

    def scene_tasks(self):
        """Real/Sim uses the current scene task, including non-task Workflow screens."""
        if getattr(self.current_screen, 'task_selector', None) is None:
            self.screens['source'].sync_tasks()
        return self.task_selector.selection()

    def _make_screen(self, key):
        from .ui_components.screens.source import SourceScreen
        from .ui_components.screens.replay import ReplayScreen
        from .ui_components.screens.annotation import AnnotationScreen
        from .ui_components.screens.datagen import DataGenScreen
        from .ui_components.screens.conversion import ConversionScreen
        from .ui_components.screens.training import TrainingScreen
        from .ui_components.screens.sim_eval import SimEvaluationScreen
        from .ui_components.screens.real_eval import RealEvaluationScreen
        from .ui_components.screens.publication import PublicationScreen
        from .ui_components.screens.curation import CurationScreen
        from .ui_components.screens.artifacts import ArtifactExplorer
        if key.startswith('train:'):
            screen = TrainingScreen(key.split(':')[1], self.context, self.controller)
        elif key == 'artifacts':
            screen = ArtifactExplorer(self.context, self.controller)
        elif key == 'curation':
            screen = CurationScreen(self.context, self.controller)
        else:
            cls = dict(source=SourceScreen, replay=ReplayScreen, annotate=AnnotationScreen, datagen=DataGenScreen,
                convert=ConversionScreen, sim_eval=SimEvaluationScreen, real_eval=RealEvaluationScreen,
                hf_dataset=PublicationScreen, hf_policy=PublicationScreen)[key]
            screen = cls(key, key, self.context, self.controller)
        screen.navigate.connect(lambda target, payload: self.guard(lambda: self.select_stage(target, artifact_id=payload)))
        self.stack.addWidget(screen)
        self.screens[key] = screen
        return screen

    def select_stage(self, key, architecture=None, artifact_id=None):
        if self.busy:
            self.navigation.select(self.current_key)
            raise ValueError('실행이 끝나거나 중지한 뒤 작업 화면을 변경하세요.')
        if self.current_screen is not None and hasattr(self.current_screen, 'can_leave') and not self.current_screen.can_leave():
            self.navigation.select(self.current_key)
            raise ValueError('Curation 분석/저장이 끝난 뒤 작업 화면을 변경하세요.')
        if key == 'train':
            key = 'train:'+(architecture or self.last_training)
        if key not in self.navigation.items:
            raise ValueError('Unknown Workflow screen: '+key)
        if key.startswith('train:'):
            self.last_training = key.split(':')[1]
        collection = key.split(':')[1] if key.startswith('artifacts:') else None
        screen_key = 'artifacts' if collection else key
        screen = self.screens.get(screen_key) or self._make_screen(screen_key)
        # Validate explicit navigation before changing either selection or the active screen.
        payload = dict(artifact_id=artifact_id) if isinstance(artifact_id, str) else artifact_id
        if payload and 'picker' in screen.__dict__:
            from .ui_components.artifact_picker import compatible_input
            record = self.registry.get(payload['artifact_id'])
            if not compatible_input(record, screen.picker.stage, self.context, screen.picker.role):
                raise ValueError('이 Artifact ID는 현재 화면·Workspace·Backend와 호환되지 않습니다.')
            if payload.get('checkpoint_file'):
                from .ui_components.screens.sim_eval import checkpoint_paths
                if payload['checkpoint_file'] not in [str(p) for p in checkpoint_paths(record['path'])]:
                    raise ValueError('선택한 Checkpoint가 이 Policy에 없습니다.')
        self.controller.invalidate()
        if self.current_screen is not None:
            self.current_screen.deactivate()
        self.current_screen, self.current_key = screen, key
        self.stack.setCurrentWidget(screen)
        self.navigation.select(key)
        self.dashboard_title.setText(LABELS.get(screen.stage or screen.key, key))
        if collection:
            screen.set_collection(collection)
        screen.activate()
        if payload:
            identifier = payload['artifact_id']
            if hasattr(screen, 'select_artifact'):
                screen.select_artifact(identifier)
            elif hasattr(screen, 'picker'):
                screen.picker.select(identifier)
            if payload.get('checkpoint_file') and hasattr(screen, 'checkpoint_choice'):
                index = screen.checkpoint_choice.findData(payload['checkpoint_file'])
                if index < 0:
                    raise ValueError('선택한 Checkpoint가 이 Policy에 없습니다.')
                screen.checkpoint_choice.setCurrentIndex(index)

    def _busy_changed(self, busy):
        self.navigation.setEnabled(not busy)
        self.backend.setEnabled(not busy)
        if busy:
            for screen in self.screens.values():
                if screen is not self.current_screen and hasattr(screen, 'monitor'):
                    screen.monitor.run.setEnabled(False)

    def refresh(self):
        if self.current_screen is not None and not self.busy:
            self.current_screen.refresh()

    def select_workspace(self, path):
        self.context.select_workspace(path)
        screen = self.current_screen
        if screen.stage == 'source':
            record = next((r for r in self.registry.all() if r['type']=='workspace' and r['path']==self.workspace.text()), None)
            if record:
                screen.picker.select(record['id'])

    def browse_workspace(self):
        if self.busy:
            raise ValueError('Workflow 실행이 끝난 뒤 Scene을 선택하세요.')
        path = QFileDialog.getExistingDirectory(self, 'Workspace 또는 Builder export 폴더 선택', browse_directory('workspace', self.workspace.text()))
        if not path:
            return
        if (Path(path)/'scene_manifest.yaml').is_file():
            result = subprocess.run([self.python.text(), str(ROOT/'scripts/real2sim/workflow.py'), 'import-builder', path],
                                    capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise RuntimeError('Builder 가져오기 실패: '+result.stderr[-3000:])
            path = result.stdout.strip().splitlines()[-1]
        self.registry.register('workspace', path)
        self.select_workspace(path)

    def poll_preview(self):
        self.controller.poll()
        if self.current_screen is not None:
            self.current_screen.poll()

    def guard(self, fn):
        try:
            return fn()
        except Exception as exc:
            QMessageBox.warning(self, 'Workflow', str(exc))

    def closeEvent(self, event):
        for screen in self.screens.values():
            if hasattr(screen, 'can_leave') and not screen.can_leave():
                event.ignore()
                return
        self.timer.stop()
        self.controller.close()
        for screen in self.screens.values():
            screen.deactivate()
        super().closeEvent(event)
