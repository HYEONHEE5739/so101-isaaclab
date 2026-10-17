"""Registry selection and metadata filtering shared by every task screen."""
from pathlib import Path
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QComboBox, QLineEdit,
                             QLabel, QPushButton, QFileDialog, QToolButton, QPlainTextEdit, QGridLayout)
from ..artifacts import artifact_name, artifact_description, browse_directory
from ..operations import STAGES


def partition(record):
    return record.get('source_partition') or record.get('provenance', {}).get('source_partition') or {}


def compatible_input(record, stage, context, role=None):
    if record['type'] not in STAGES[stage][0]:
        return False
    if record.get('status', 'SUCCEEDED') != 'SUCCEEDED' or not Path(record['path']).exists():
        return False
    workspace = context.workspace_identity()
    if record['type'] != 'workspace' and workspace and record.get('workspace') and record['workspace'] != workspace:
        return False
    cross_engine = (stage in ('train', 'sim_eval', 'real_eval', 'hf_dataset', 'hf_policy', 'convert')
                    or stage in ('replay', 'annotate', 'datagen') and
                    (record.get('format') in ('so101.common-episode/2', 'so101.canonical/1')
                     or record.get('env_args', {}).get('workspace_path')))
    if record['type'] != 'workspace' and not cross_engine and record.get('backend', 'isaac') != context.backend.currentData():
        return False
    return not (role == 'train' and partition(record).get('split') == 'val')


def lineage_text(registry, record):
    lines, seen = [], set()
    def visit(row, depth=0):
        if row['id'] in seen:
            return
        seen.add(row['id'])
        lines.append('  ' * depth + artifact_name(row) + ' #' + row['id'])
        for identifier in row.get('parents', []):
            try:
                visit(registry.get(identifier), depth + 1)
            except (OSError, ValueError):
                lines.append('  ' * (depth + 1) + '누락된 Parent: ' + identifier)
    visit(record)
    return '\n'.join(lines)


class ArtifactPicker(QWidget):
    changed = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, stage, context, parent=None):
        super().__init__(parent)
        self.stage, self.context, self.registry = stage, context, context.registry
        self.role = None
        self.preferred_id = None
        self.initialized = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel('Train Dataset' if stage == 'train' else '입력 Artifact')
        layout.addWidget(self.label)
        self.combo = QComboBox()
        self.combo.setMinimumContentsLength(16)
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        layout.addWidget(self.combo)
        toggle = QToolButton()
        toggle.setText('검색 / 종류 / Task')
        toggle.setCheckable(True)
        layout.addWidget(toggle)
        filters = QWidget()
        row = QVBoxLayout(filters)
        row.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText('저장 이름 / ID 검색')
        self.kind_filter = QComboBox()
        self.kind_filter.addItem('전체 종류', None)
        for kind in sorted(STAGES[stage][0]):
            self.kind_filter.addItem(kind, kind)
        self.task = QLineEdit()
        self.task.setPlaceholderText('Task 필터')
        for widget in (self.search, self.kind_filter, self.task):
            row.addWidget(widget)
        filters.hide()
        toggle.toggled.connect(filters.setVisible)
        layout.addWidget(filters)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        metadata_toggle = QToolButton()
        metadata_toggle.setText('입력 Metadata / 경로 / Lineage 상세')
        metadata_toggle.setCheckable(True)
        layout.addWidget(metadata_toggle)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(180)
        self.details.setMinimumWidth(0)
        self.details.hide()
        metadata_toggle.toggled.connect(self.details.setVisible)
        layout.addWidget(self.details)
        self.fixed_kind = next(iter(STAGES[stage][0])) if len(STAGES[stage][0]) == 1 else None
        self.kind = None
        if self.fixed_kind is None:
            self.kind = QComboBox()
            self.kind.addItems(sorted(STAGES[stage][0]))
            layout.addWidget(self.kind)
        buttons = QGridLayout()
        layout.addLayout(buttons)
        for index, (title, callback) in enumerate([('파일 등록', lambda: self.browse(False)), ('폴더 등록', lambda: self.browse(True)),
                                ('폴더 열기', self.open_folder), ('새로고침', self.refresh)]):
            button = QPushButton(title)
            button.clicked.connect(lambda _, fn=callback: self.guard(fn))
            buttons.addWidget(button, index//2, index%2)
        self.search.textChanged.connect(self.refresh)
        self.kind_filter.currentIndexChanged.connect(self.refresh)
        self.task.textChanged.connect(self.refresh)
        self.combo.currentIndexChanged.connect(self._selected)

    def guard(self, fn):
        try:
            fn()
        except Exception as exc:
            self.error.emit(str(exc))

    def candidates(self):
        return [record for record in self.registry.all() if compatible_input(record, self.stage, self.context, self.role)]

    def refresh(self):
        selected = self.preferred_id
        records = self.candidates()
        self.combo.blockSignals(True)
        self.combo.clear()
        for record in records:
            if self.kind_filter.currentData() and record['type'] != self.kind_filter.currentData():
                continue
            if self.search.text().casefold() not in (artifact_name(record) + ' ' + record['id']).casefold():
                continue
            if self.task.text().casefold() not in ','.join(t['task_id'] for t in record.get('tasks', [])).casefold():
                continue
            self.combo.addItem(artifact_description(record), record['id'])
            self.combo.setItemData(self.combo.count()-1, record['path'] + '\nID: ' + record['id'], Qt.ItemDataRole.ToolTipRole)
        if not self.initialized and self.combo.count():
            selected = self.combo.itemData(self.combo.count()-1)
            self.preferred_id = selected
            self.initialized = True
        self.combo.setCurrentIndex(self.combo.findData(selected))
        self.combo.blockSignals(False)
        self.update_details()
        self.changed.emit()

    def _selected(self):
        self.preferred_id = self.combo.currentData()
        self.initialized = True
        self.update_details()
        self.changed.emit()

    def update_details(self):
        if self.combo.currentData() is None:
            message = '선택한 ID가 현재 필터·Workspace·Backend 조건에 맞지 않습니다: ' + self.preferred_id if self.preferred_id else '입력 Artifact를 선택하세요.'
            self.details.setPlainText(message)
            self.summary.setText(message)
            return
        record = self.registry.get(self.combo.currentData())
        self.details.setPlainText(artifact_description(record) + '\n' + record['path'] + '\n계보 (결과 → 입력):\n' + lineage_text(self.registry, record))
        split = partition(record)
        self.summary.setText('ID: #' + record['id'][:12] + ' · ' + record['type'] +
            (f" · {split['split']} Source {len(split['source_ids'])}" if split else ''))
        self.combo.setToolTip(record['path'])

    def select(self, identifier):
        record = self.registry.get(identifier)
        if not compatible_input(record, self.stage, self.context, self.role):
            raise ValueError('선택한 Artifact는 현재 작업·Workspace·Backend와 호환되지 않습니다: ' + identifier)
        self.search.clear()
        self.task.clear()
        self.kind_filter.setCurrentIndex(0)
        self.preferred_id, self.initialized = identifier, True
        self.refresh()
        if self.combo.currentData() != identifier:
            raise ValueError('Artifact 선택 실패: ' + identifier)

    def record(self):
        if self.combo.currentData() is None:
            raise ValueError('입력 데이터를 선택하세요.')
        record = self.registry.get(self.combo.currentData())
        if not compatible_input(record, self.stage, self.context, self.role):
            raise ValueError('선택한 입력이 더 이상 유효하지 않습니다. 목록을 새로고침하세요.')
        return record

    def browse(self, directory=False):
        kind = self.fixed_kind or self.kind.currentText()
        start = browse_directory(kind, registry=self.registry)
        if directory:
            path = QFileDialog.getExistingDirectory(self, 'Artifact 폴더 선택', start)
        else:
            path = QFileDialog.getOpenFileName(self, 'Artifact 파일 선택', start, 'Artifacts (*.json *.hdf5 *.h5)')[0]
        if not path:
            return
        if kind == 'policy' and not (Path(path) / 'project_provenance.json').is_file():
            from ..policy import import_checkpoint
            record = import_checkpoint(path, self.registry)
        else:
            record = self.registry.register(kind, path)
        if kind == 'workspace':
            self.context.select_workspace(path)
        if self.role == 'train' and partition(record).get('split') == 'val':
            # The TrainingScreen owns independent validation selection.
            self.validation_imported(record)
        else:
            self.select(record['id'])

    def validation_imported(self, record):
        raise ValueError('Validation 입력은 MiniVLA 화면에서 선택하세요.')

    def open_folder(self):
        path = Path(self.record()['path'])
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path if path.is_dir() else path.parent)))
