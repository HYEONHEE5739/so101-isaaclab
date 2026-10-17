"""Shared desktop context and persisted backend/workspace choices."""
import json
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from ..operations import default_python
from ...real2sim.storage import atomic
from ...real2sim.workspaces.package import load, metadata


class WorkflowContext(QObject):
    changed = pyqtSignal()
    browseRequested = pyqtSignal()

    def __init__(self, registry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.locked = False
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        toolbar = QHBoxLayout()
        layout.addLayout(toolbar)
        self.advanced = QWidget()
        advanced_form = QFormLayout(self.advanced)
        self.advanced.hide()
        self.workspace = QLineEdit()
        self.python = QLineEdit(default_python())
        self.backend = QComboBox()
        self.backend.addItem('Isaac Sim / Isaac Lab', 'isaac')
        self.backend.addItem('MuJoCo', 'mujoco')
        self.mode = QComboBox()
        self.mode.addItem('Operator — 통합 UI', 'operator')
        self.mode.addItem('Developer — 시뮬레이터 창', 'developer')
        self.browse = QPushButton('Workspace 선택 / Import')
        toolbar.addWidget(QLabel('Workspace'))
        toolbar.addWidget(self.workspace, 1)
        toolbar.addWidget(self.browse)
        toolbar.addWidget(self.backend)
        advanced_form.addRow('실행 Python', self.python)
        advanced_form.addRow('실행 모드', self.mode)
        layout.addWidget(self.advanced)
        try:
            saved = json.loads((registry.root / 'ui.json').read_text())
        except (OSError, ValueError):
            saved = {}
        self.workspace.setText(saved.get('workspace', ''))
        self.mode.setCurrentIndex(max(0, self.mode.findData(saved.get('mode', 'operator'))))
        self.backend_pythons = dict(isaac=saved.get('python', default_python()), mujoco=default_python())
        self.backend_pythons.update(saved.get('backend_pythons', {}))
        self.object_selections = saved.get('object_selections', {})
        self.backend.setCurrentIndex(max(0, self.backend.findData(saved.get('backend', 'isaac'))))
        self.previous_backend = self.backend.currentData()
        self.python.setText(self.backend_pythons[self.previous_backend])
        self.backend.currentIndexChanged.connect(self._backend_changed)
        self.workspace.textChanged.connect(self._edited)
        self.python.textChanged.connect(self._edited)
        self.mode.currentIndexChanged.connect(self._edited)
        self.browse.clicked.connect(self.browseRequested)

    def snapshot(self):
        return dict(workspace=self.workspace.text().strip(), python=self.python.text().strip(),
                    backend=self.backend.currentData(), mode=self.mode.currentData())

    def workspace_identity(self):
        path = self.workspace.text()
        if getattr(self, '_identity_path', None) == path:
            return self._identity
        try:
            result = metadata(load(path)) if path else None
        except (OSError, ValueError, KeyError):
            result = None
        self._identity_path, self._identity = path, result
        return result

    def _edited(self):
        if self.locked:
            self._restore_locked()
            return
        self.changed.emit()

    def _backend_changed(self):
        if self.locked:
            self._restore_locked()
            return
        self.backend_pythons[self.previous_backend] = self.python.text()
        self.previous_backend = self.backend.currentData()
        self.python.setText(self.backend_pythons[self.previous_backend])
        self.save()
        self.changed.emit()

    def _restore_locked(self):
        for key in ('workspace', 'python', 'backend', 'mode'):
            widget = getattr(self, key)
            widget.blockSignals(True)
            if key in ('workspace', 'python'):
                widget.setText(self._locked_snapshot[key])
            else:
                widget.setCurrentIndex(widget.findData(self._locked_snapshot[key]))
            widget.blockSignals(False)

    def set_locked(self, locked):
        if locked:
            self._locked_snapshot = self.snapshot()
        self.locked = locked
        self.widget.setEnabled(not locked)

    def select_workspace(self, path):
        if self.locked:
            raise ValueError('Workflow 실행이 끝난 뒤 Scene을 선택하세요.')
        w = load(path)
        self.workspace.setText(str(w['root']))
        self.save()

    def save(self):
        values = self.snapshot()
        self.backend_pythons[values['backend']] = values['python']
        atomic(self.registry.root / 'ui.json', dict(values, backend_pythons=self.backend_pythons,
                                                  object_selections=self.object_selections))


class InheritedTaskInfo(QWidget):
    """Derived stages display recorded tasks; there are no disabled task editing controls."""
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.workspace = None
        self.loading = False
        self.summary = QLabel('Task / Scene은 입력 Artifact에서 상속합니다.')
        self.summary.setWordWrap(True)
        QVBoxLayout(self).addWidget(self.summary)

    def configure(self, workspace, stage, inherited=None, active_object_ids=None):
        from ..tasks import active_objects, workspace_key
        self.workspace, self.stage = workspace, stage
        self.workspace_identity = workspace_key(workspace)
        self.definitions = inherited or []
        self.active_object_ids = active_objects(workspace, active_object_ids)
        self.summary.setText('입력 Task: ' + ', '.join(t['task_id'] for t in self.definitions) +
                             '\nActive Objects: ' + ', '.join(self.active_object_ids) + '\nTask / Scene은 입력에서 상속합니다.')

    def object_selection(self):
        return list(self.active_object_ids)

    def selection(self, evaluation=False):
        import copy
        return copy.deepcopy(self.definitions)
