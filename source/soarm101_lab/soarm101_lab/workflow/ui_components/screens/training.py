"""Independent model settings with shared trainer/validation/checkpoint contracts."""
import json
import time
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QCheckBox, QLineEdit, QPushButton, QFileDialog, QLabel, QWidget, QFormLayout
from .base import StageScreen
from ...option_help import template, TRAIN_KEYS
from ...artifacts import artifact_description
from ...policy import minivla_root
from ..artifact_picker import partition
from ..training_metrics import TrainingMetrics, TrainingLossPlot


class TrainingScreen(StageScreen):
    action = '학습 시작'
    basic_order = ('steps', 'batch_size', 'save_freq', 'val_every', 'early_stopping')

    def __init__(self, model, context, controller, parent=None):
        self.model = model
        self.basic_keys = {'steps', 'batch_size', 'save_freq'}
        if model == 'minivla':
            self.basic_keys |= {'val_every', 'early_stopping'}
        super().__init__('train', 'train:'+model, context, controller, parent)
        self.metrics = TrainingMetrics(self) if model == 'minivla' else None
        self.last_metrics_poll = 0
        if self.metrics:
            self.chart = TrainingLossPlot(resolver=self.metrics.training_metrics_path)
            self.monitor.media_layout.addWidget(self.chart)
            self.training_status = QLabel('Loss / Validation: 미측정 · 첫 학습 기록 대기')
            self.training_status.setWordWrap(True)
            self.monitor.media_layout.addWidget(self.training_status)
            self.metrics_source = QLabel('학습 기록: 새 실행의 첫 기록 대기')
            self.metrics_source.setWordWrap(True)
            self.metrics_source.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.monitor.media_layout.addWidget(self.metrics_source)
            self.loss_button = QPushButton('학습 Loss 그래프')
            self.loss_button.clicked.connect(self.metrics.show_training_loss)
            self.monitor.tools.addWidget(self.loss_button)
        controller.validationChoices.connect(self.apply_validation_choices)

    def build_inputs(self):
        self.architecture = QComboBox()
        for label, model in [('ACT', 'act'), ('SmolVLA', 'smolvla'), ('MiniVLA · Qwen3-VL + Flow-DiT', 'minivla')]:
            self.architecture.addItem(label, model)
        self.architecture.setCurrentIndex(self.architecture.findData(self.model))
        self.form.addRow('Model Architecture', self.architecture)
        self.architecture.currentIndexChanged.connect(self.change_model)
        self.validation_input = None
        if self.model == 'minivla':
            self.picker.role = 'train'
            self.validation_input = QComboBox()
            self.validation_input.addItem('별도 Validation 선택 필요 / Legacy는 고급 설정', None)
            self.validation_input.setToolTip('호환되는 독립 Val Source Partition을 변환한 LeRobot Dataset. 자동 생성/변환/Train 재분할을 하지 않습니다.')
            self.form.addRow('Validation Dataset', self.validation_input)
            self.validation_input.currentIndexChanged.connect(self.update_ready)
            self.picker.validation_imported = self.validation_imported
        self.resume_training = QCheckBox('이어서 학습')
        self.resume_path = QLineEdit()
        self.resume_path.setPlaceholderText('.pt 파일' if self.model == 'minivla' else 'checkpoints/040000 폴더')
        self.resume_browse = QPushButton('체크포인트 선택')
        self.resume_models = QComboBox()
        self.resume_models.addItem('등록 모델 선택 (또는 파일 선택)', None)
        self.form.addRow('Resume Training', self.resume_training)
        self.resume_panel = QWidget()
        resume_form = QFormLayout(self.resume_panel)
        resume_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        for label, widget in [('Resume Checkpoint', self.resume_path),
                              ('선택', self.resume_browse), ('등록 모델에서 Resume', self.resume_models)]:
            resume_form.addRow(label, widget)
        self.form.addRow(self.resume_panel)
        self.resume_panel.hide()
        self.resume_training.toggled.connect(self.resume_panel.setVisible)
        self.resume_training.toggled.connect(self.update_ready)
        self.resume_path.textChanged.connect(self.update_ready)
        self.resume_browse.clicked.connect(lambda: self.guard(self.browse_checkpoint))
        self.resume_models.activated.connect(lambda: self.guard(self.select_resume_model))

    def configure_options(self):
        self.options.allowed_keys = TRAIN_KEYS[self.model]
        self.options.minivla = self.model == 'minivla'
        values = dict(steps=50000, val_every=1000, save_freq=2500, early_stopping=False, batch_size=8) if self.model == 'minivla' else template('train')
        self.options.setPlainText(json.dumps({k:v for k,v in values.items() if k in TRAIN_KEYS[self.model]}))

    def change_model(self):
        if not self.initializing and not self.controller.busy:
            self.navigate.emit('train:'+self.architecture.currentData(), self.inputs.currentData())
            self.architecture.blockSignals(True)
            self.architecture.setCurrentIndex(self.architecture.findData(self.model))
            self.architecture.blockSignals(False)

    def refresh(self):
        old = self.resume_models.currentData()
        self.resume_models.blockSignals(True)
        self.resume_models.clear()
        self.resume_models.addItem('등록 모델 선택 (또는 파일 선택)', None)
        for record in self.registry.all():
            if record['type'] != 'policy' or record.get('status') != 'SUCCEEDED' or not Path(record['path']).exists():
                continue
            fmt = record.get('format')
            model = 'minivla' if fmt == 'so101.minivla-policy/1' else fmt.get('type') if isinstance(fmt, dict) else None
            if model == self.model:
                self.resume_models.addItem(artifact_description(record), record['id'])
        self.resume_models.setCurrentIndex(max(0, self.resume_models.findData(old)))
        self.resume_models.blockSignals(False)
        super().refresh()

    def input_changed(self):
        if self.initializing or self.updating:
            return
        if self.model == 'minivla':
            self.validation_candidates = tuple(r for r in self.registry.all() if r['type'] == 'dataset'
                and r.get('status', 'SUCCEEDED') == 'SUCCEEDED' and Path(r['path']).exists()
                and isinstance(r.get('format'), dict) and partition(r).get('split') == 'val')
        super().input_changed()

    def apply_validation_choices(self, owner, records):
        if owner != self.key or self.validation_input is None:
            return
        selected = self.validation_input.currentData()
        self.validation_input.blockSignals(True)
        self.validation_input.clear()
        self.validation_input.addItem('별도 Validation 선택 필요 / Legacy는 고급 설정', None)
        for record in records:
            self.validation_input.addItem(artifact_description(record), record['id'])
            self.validation_input.setItemData(self.validation_input.count()-1, record['path']+'\nID: '+record['id'], Qt.ItemDataRole.ToolTipRole)
        if selected and self.validation_input.findData(selected) < 0:
            self.validation_input.addItem('호환되지 않거나 누락된 Validation · #'+selected, selected)
        self.validation_input.setCurrentIndex(max(0, self.validation_input.findData(selected)))
        self.validation_input.blockSignals(False)

    def validation_imported(self, record):
        if self.validation_input.findData(record['id']) < 0:
            self.validation_input.addItem(artifact_description(record), record['id'])
        self.validation_input.setCurrentIndex(self.validation_input.findData(record['id']))
        self.input_changed()

    def collect_options(self):
        opts = super().collect_options()
        opts = {k:v for k,v in opts.items() if k in TRAIN_KEYS[self.model] or k in ('backend', 'python', 'mode')}
        opts['architecture'] = self.model
        if self.model == 'minivla':
            if self.validation_input.currentData():
                opts['validation_dataset'] = self.validation_input.currentData()
            if not opts.get('model_config'):
                from ....learning.minivla.config import MODEL_ROOT
                opts['model_config'] = str(MODEL_ROOT/'configs/default.json')
        if self.resume_training.isChecked():
            if not self.resume_path.text().strip():
                raise ValueError('이어서 학습할 checkpoint를 선택하세요.')
            opts['resume_checkpoint'] = self.resume_path.text().strip()
        return opts

    def browse_checkpoint(self, path=None):
        if not path:
            path = QFileDialog.getOpenFileName(self, 'MiniVLA 재개 체크포인트 선택', '', 'PyTorch (*.pt)')[0] if self.model == 'minivla' else QFileDialog.getExistingDirectory(self, '체크포인트 폴더 선택 (예: checkpoints/040000)')
        if not path:
            return
        self.resume_path.setText(str(path))
        root = Path(path).resolve()
        try:
            opts = json.loads(self.options.toPlainText())
            if root.suffix == '.pt':
                model_root = minivla_root(root)
                saved = json.loads((model_root/'config.json').read_text())
                opts.update(steps=saved['steps'], batch_size=saved.get('batch_size', 1), model_config=str(model_root/'config.json'))
            else:
                if root.name == 'pretrained_model':
                    root = root.parent
                saved = json.loads((root/'pretrained_model/train_config.json').read_text())
                if saved['policy']['type'] != self.model:
                    raise ValueError('선택한 checkpoint의 Model Architecture가 다릅니다.')
                opts['steps'] = saved['steps']
            self.options.setPlainText(json.dumps(opts, ensure_ascii=False))
        except (OSError, KeyError):
            pass  # Existing readiness validation explains incomplete checkpoint files.
        self.resume_training.setChecked(True)

    def select_resume_model(self):
        if self.resume_models.currentData():
            root = Path(self.registry.get(self.resume_models.currentData())['path'])
            self.browse_checkpoint(root/('last.pt' if (root/'last.pt').is_file() else 'best.pt') if self.model == 'minivla' else root)

    def started(self, active):
        super().started(active)
        if active['owner'] == self.key and self.metrics:
            self.metrics._training_metrics_cache = None
            self.last_metrics_poll = 0
            self.chart.reload()
            self.training_status.setText('Loss / Validation: 미측정 · 첫 학습 기록 대기')
            self.metrics_source.setText('현재 실행 · 첫 학습 기록 대기')

    def poll(self):
        super().poll()
        if not self.metrics or time.monotonic()-self.last_metrics_poll < 1.5:
            return
        self.last_metrics_poll = time.monotonic()
        self.chart.reload()
        text = 'Loss / Validation: 미측정 · 첫 학습 기록 대기'
        if self.chart.rows:
            from ...presentation import training_status_text
            text = training_status_text(self.chart.rows[-1])
        path = self.chart.path
        current = (self.controller.active or {}).get('owner') == self.key and self.monitor.session is not None
        self.metrics_source.setText(('현재 실행' if current else '기존 학습 기록 · 선택한 Train 입력의 최근 저장 모델') +
                                   '\n' + str(path) if path else '현재 실행의 첫 학습 기록 대기')
        if path:
            try:
                status = json.loads((path.parent/'training_status.json').read_text())
                text += f"\n저장된 step {status.get('step','?')}/{status.get('target_steps','?')}"
            except (OSError, ValueError):
                pass
            existing = [p.name for p in (path.parent/'best.pt', path.parent/'last.pt') if p.is_file()]
            if existing:
                text += ' · Checkpoint: ' + ', '.join(existing)
        self.training_status.setText(text)

    def deactivate(self):
        super().deactivate()
        if self.metrics and self.metrics._loss_dialog is not None:
            self.metrics._loss_dialog.close()
