"""Dedicated Curation navigation using the existing dialog and analysis/save workers."""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QMessageBox
from ...curation_ui import CurationDialog
from ..artifact_picker import ArtifactPicker


class CurationScreen(QWidget):
    navigate = pyqtSignal(str, object)

    def __init__(self, context, controller, parent=None):
        super().__init__(parent)
        self.context, self.controller = context, controller
        self.key, self.stage = 'curation', None
        self.picker = ArtifactPicker('replay', context)
        self.dialog = CurationDialog(context.registry, self, workspace=context.workspace_identity())
        self.dialog.setWindowFlags(Qt.WindowType.Widget)
        layout = QVBoxLayout(self)
        layout.addWidget(self.picker)
        self.open_button = QPushButton('선택 데이터 열기')
        layout.addWidget(self.open_button)
        layout.addWidget(self.dialog, 1)
        self.open_button.clicked.connect(self.open_selected)
        self.picker.error.connect(lambda message: QMessageBox.warning(self, 'Curation', message))

    def activate(self):
        self.refresh()

    def deactivate(self):
        self.dialog.pause()

    def refresh(self):
        self.picker.refresh()
        self.dialog.workspace = self.context.workspace_identity()

    def select_artifact(self, identifier):
        self.picker.select(identifier)
        self.open_selected()

    def open_selected(self):
        try:
            self.dialog.load_data(self.picker.record()['path'])
        except Exception as exc:
            QMessageBox.warning(self, 'Curation', str(exc))

    def poll(self):
        pass

    def can_leave(self):
        return not any(job is not None and job.isRunning() for job in (self.dialog.job, self.dialog.frame_job))
