"""Typed workflow settings backed by the existing option schema."""
import json
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
                             QLineEdit, QSpinBox, QToolButton, QPushButton, QTableWidget,
                             QTableWidgetItem, QHeaderView)
from .option_help import FIELDS, KEYS

LABELS = {'steps': '총 학습 step', 'batch_size': 'Batch 크기', 'num_workers': '데이터 로더 workers',
          'save_freq': '체크포인트 저장 간격', 'num_successful_demos': '목표 성공 데이터 수',
          'episodes': 'Task당 평가 에피소드', 'max_steps': '에피소드 최대 step',
          'output': '저장 경로', 'base_model': 'Pretrained 모델', 'generation_seed': '고정 생성 seed',
          'device': '실행 장치', 'repo_id': 'Dataset / Hub ID', 'policy_options': '모델 추가 옵션',
          'rename_map': '카메라 키 매핑'}


class SettingsForm(QWidget):
    textChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.stage = 'source'
        self.editors = {}
        self.extra = {}
        self.invalid = None

    def set_stage(self, stage):
        self.stage = stage

    def setPlainText(self, text):
        values = json.loads(text)
        if not isinstance(values, dict):
            self.invalid = text
            self.textChanged.emit()
            return
        self.invalid = None
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.editors = {}
        keys = [k for k in KEYS[self.stage] if k != 'architecture']
        self.extra = {k: v for k, v in values.items() if k not in keys}
        for key in keys:
            kind, default, description = FIELDS[key]
            card = QWidget(); card.setObjectName('settingCard'); card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            box = QVBoxLayout(card)
            row = QHBoxLayout(); box.addLayout(row)
            enabled = QCheckBox(LABELS.get(key, key))
            enabled.setToolTip(key)
            enabled.setChecked(key in values)
            row.addWidget(enabled, 1)
            help_button = QToolButton(); help_button.setText('설명'); help_button.setCheckable(True)
            row.addWidget(help_button)
            help_label = QLabel(description); help_label.setWordWrap(True); help_label.hide()
            help_button.toggled.connect(help_label.setVisible)
            value = values.get(key, default)
            if kind == '정수':
                editor = QSpinBox()
                editor.setRange(0 if key in ('num_workers', 'generation_seed') else 1, 2147483647)
                editor.setValue(value if value is not None else 0)
                editor.valueChanged.connect(self.textChanged)
            elif kind == 'true/false':
                editor = QCheckBox('사용')
                editor.setChecked(bool(value))
                editor.toggled.connect(self.textChanged)
            elif kind == '객체':
                editor = QTableWidget(0, 2)
                editor.setHorizontalHeaderLabels(['항목 / 원본 키', '값 / 대상 키'])
                editor.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
                editor.setMinimumHeight(115)
                for name, entry in (value or {}).items():
                    r = editor.rowCount(); editor.insertRow(r)
                    editor.setItem(r, 0, QTableWidgetItem(name))
                    editor.setItem(r, 1, QTableWidgetItem(entry if isinstance(entry, str) else json.dumps(entry)))
                editor.itemChanged.connect(self.textChanged)
                buttons = QHBoxLayout(); box.addLayout(buttons)
                add = QPushButton('+ 항목'); remove = QPushButton('선택 삭제')
                buttons.addWidget(add); buttons.addWidget(remove)
                add.clicked.connect(lambda _, e=editor: e.insertRow(e.rowCount()))
                remove.clicked.connect(lambda _, e=editor: (e.removeRow(e.currentRow()), self.textChanged.emit()))
                enabled.toggled.connect(add.setEnabled); enabled.toggled.connect(remove.setEnabled)
                add.setEnabled(enabled.isChecked()); remove.setEnabled(enabled.isChecked())
            else:
                editor = QLineEdit('' if value is None else str(value))
                if value is None:
                    editor.setPlaceholderText('자동 기본값 사용 · 지정하려면 위 항목 체크')
                editor.textChanged.connect(self.textChanged)
            editor.setEnabled(enabled.isChecked())
            enabled.toggled.connect(editor.setEnabled)
            enabled.toggled.connect(self.textChanged)
            box.addWidget(editor); box.addWidget(help_label)
            self.layout_.addWidget(card)
            self.editors[key] = enabled, editor, kind
        self.textChanged.emit()

    def toPlainText(self):
        if self.invalid is not None:
            return self.invalid
        values = dict(self.extra)
        for key, (enabled, editor, kind) in self.editors.items():
            if not enabled.isChecked():
                continue
            if kind == '정수':
                value = editor.value()
            elif kind == 'true/false':
                value = editor.isChecked()
            elif kind == '객체':
                value = {}
                for r in range(editor.rowCount()):
                    name = editor.item(r, 0); entry = editor.item(r, 1)
                    if not name or not name.text().strip():
                        if entry and entry.text().strip():
                            raise ValueError(f'{key}: 항목 이름을 입력하세요.')
                        continue
                    name = name.text().strip()
                    if name in value:
                        raise ValueError(f'{key}: 중복 항목 {name}')
                    raw = entry.text() if entry else ''
                    if key == 'rename_map':
                        value[name] = raw
                    else:
                        try: value[name] = json.loads(raw)
                        except ValueError: value[name] = raw
            else:
                value = editor.text()
            values[key] = value
        return json.dumps(values, ensure_ascii=False)
