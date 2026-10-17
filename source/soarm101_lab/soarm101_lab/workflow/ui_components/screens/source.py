"""Leader recording controls and timing; no training or evaluation widgets."""
from PyQt6.QtWidgets import QLineEdit, QSpinBox, QDoubleSpinBox
from pathlib import Path
from .base import StageScreen
from ..settings import fold, ResumeDataSettings


class SourceScreen(StageScreen):
    action = '데모 기록 시작'

    def activate(self):
        if not self.loaded and self.context.workspace.text():
            record = next((r for r in self.registry.all() if r['type'] == 'workspace' and r['path'] == self.context.workspace.text()), None)
            self.picker.preferred_id = record['id'] if record else None
            self.picker.initialized = True
        super().activate()

    def snapshot(self):
        snapshot = super().snapshot()
        if snapshot['workspace'] and Path(snapshot['workspace']).expanduser().resolve() != Path(snapshot['artifact']['path']).resolve():
            raise ValueError('Source 입력 Workspace와 공통 Workspace가 다릅니다. 사용할 Workspace를 명시적으로 선택하세요.')
        return snapshot

    def build_inputs(self):
        self.port = QLineEdit('/dev/so101_leader')
        self.episodes = QSpinBox()
        self.episodes.setRange(1, 100000)
        self.episode_time = QDoubleSpinBox()
        self.episode_time.setRange(.1, 3600)
        self.episode_time.setValue(20)
        self.episode_time.setSuffix(' s')
        self.reset_time = QDoubleSpinBox()
        self.reset_time.setRange(0, 600)
        self.reset_time.setValue(5)
        self.reset_time.setSuffix(' s')
        for name, widget in [('Leader port', self.port), ('Episodes', self.episodes),
                             ('Episode Time', self.episode_time), ('Reset Time', self.reset_time)]:
            self.form.addRow(name, widget)
            (widget.textChanged if isinstance(widget, QLineEdit) else widget.valueChanged).connect(self.update_ready)
        advanced = fold(self.settings_layout, '고급 기록 설정 · Resume Data')
        self.resume = ResumeDataSettings(self.update_ready)
        advanced.addRow(self.resume)
        self.resume_data = self.resume.path

    def collect_options(self):
        opts = super().collect_options()
        opts.pop('resume_data', None)
        opts.update(port=self.port.text(), episodes=self.episodes.value(), episode_time_s=self.episode_time.value(), reset_time_s=self.reset_time.value())
        if self.resume_data.text().strip():
            opts['resume_data'] = self.resume_data.text().strip()
        return opts
