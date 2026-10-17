"""Automatic/manual annotation with only the supported commands."""
from PyQt6.QtWidgets import QComboBox
from .base import StageScreen


class AnnotationScreen(StageScreen):
    action = 'Annotation 시작'

    def build_inputs(self):
        self.annotation_mode = QComboBox()
        self.annotation_mode.addItems(['자동 Annotation', '수동 Annotation'])
        self.form.addRow('Annotation 방식', self.annotation_mode)
        self.annotation_mode.currentIndexChanged.connect(self.change_mode)

    def change_mode(self):
        self.monitor.configure_commands(self.annotation_mode.currentIndex() == 1)
        self.update_ready()

    def collect_options(self):
        return dict(super().collect_options(), auto=self.annotation_mode.currentIndex() == 0)
