"""Simulation rollout and exact MiniVLA checkpoint selection."""
from pathlib import Path
from PyQt6.QtWidgets import QComboBox, QSpinBox
from .base import StageScreen


def checkpoint_paths(root):
    root = Path(root)
    return [p for p in [root/'best.pt', root/'last.pt', *sorted((root/'checkpoints').glob('step_*.pt')),
                       *sorted(root.glob('step_*.pt'))] if p.is_file()]


class SimEvaluationScreen(StageScreen):
    action = 'Sim 평가 시작'
    basic_keys = {'episodes', 'max_steps'}

    def build_inputs(self):
        self.checkpoint_choice = QComboBox()
        self.form.addRow('MiniVLA Checkpoint · Best / Last / Step', self.checkpoint_choice)
        self.checkpoint_choice.currentIndexChanged.connect(self.update_ready)
        self.eval_seed_mode = QComboBox()
        self.eval_seed_mode.addItem('고정 seed — 모델 비교용', 'fixed')
        self.eval_seed_mode.addItem('실행마다 랜덤 seed', 'random')
        self.eval_seed = QSpinBox()
        self.eval_seed.setRange(0, 2147483647)
        self.form.addRow('평가 Seed Mode', self.eval_seed_mode)
        self.form.addRow('Seed', self.eval_seed)
        self.eval_seed_mode.currentIndexChanged.connect(lambda: self.eval_seed.setEnabled(self.eval_seed_mode.currentData() == 'fixed'))
        self.eval_seed_mode.currentIndexChanged.connect(self.update_ready)
        self.eval_seed.valueChanged.connect(self.update_ready)

    def input_changed(self):
        if self.initializing or self.updating:
            return
        record = self.selected_record(False) or {}
        old = self.checkpoint_choice.currentData()
        self.checkpoint_choice.blockSignals(True)
        self.checkpoint_choice.clear()
        mini = record.get('format') == 'so101.minivla-policy/1'
        self.form.setRowVisible(self.checkpoint_choice, mini)
        if mini:
            root = Path(record['path'])
            for path in checkpoint_paths(root):
                self.checkpoint_choice.addItem(str(path.relative_to(root)), str(path))
            self.checkpoint_choice.setCurrentIndex(max(0, self.checkpoint_choice.findData(old)))
        self.checkpoint_choice.blockSignals(False)
        super().input_changed()

    def collect_options(self):
        opts = super().collect_options()
        opts.pop('checkpoint_file', None)
        if self.checkpoint_choice.currentData():
            opts['checkpoint_file'] = self.checkpoint_choice.currentData()
        opts.update(eval_seed_mode=self.eval_seed_mode.currentData(), eval_seed=self.eval_seed.value())
        return opts
