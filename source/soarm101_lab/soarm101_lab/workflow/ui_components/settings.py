"""Small settings components with one owner for each execution value."""
from PyQt6.QtWidgets import QWidget, QFormLayout, QLineEdit, QLabel, QPushButton, QToolButton, QFileDialog
from ..artifacts import suggested_name, output_location, validate_storage_name


def fold(layout, title):
    toggle = QToolButton()
    toggle.setText(title)
    toggle.setCheckable(True)
    panel = QWidget()
    form = QFormLayout(panel)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    panel.hide()
    toggle.toggled.connect(panel.setVisible)
    layout.addWidget(toggle)
    layout.addWidget(panel)
    return form


class OutputSettings(QWidget):
    def __init__(self, screen):
        super().__init__()
        self.screen = screen
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.name = QLineEdit(suggested_name(screen.stage))
        self.name.setPlaceholderText('결과물 이름')
        self.suggest = QPushButton('이름 추천')
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        self.suggest.clicked.connect(lambda: self.name.setText(suggested_name(screen.stage, screen.selected_record(False))))
        layout.addRow('저장 이름', self.name)
        layout.addRow(self.suggest)
        layout.addRow('예상 저장 위치', self.preview)
        self.name.textChanged.connect(screen.update_ready)

    def options(self, opts):
        continuing = self.screen.stage == 'train' and opts.get('resume_checkpoint') and opts.get('architecture') != 'minivla'
        self.name.setEnabled(not self.screen.controller.busy and not continuing)
        self.suggest.setEnabled(not self.screen.controller.busy and not continuing)
        if not continuing:
            opts['storage_name'] = validate_storage_name(self.name.text())
        return opts

    def update_preview(self, opts):
        try:
            output, _ = output_location(self.screen.stage, self.screen.selected_record(False) or {}, opts, self.screen.registry.root)
            self.preview.setText(str(output))
        except (ValueError, OSError, KeyError) as exc:
            self.preview.setText(str(exc))


class ResumeDataSettings(QWidget):
    def __init__(self, changed, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.path = QLineEdit()
        self.path.setPlaceholderText('선택: 기존 완료 데이터')
        form.addRow('Resume Data', self.path)
        for title, directory in [('HDF5 선택', False), ('데이터 폴더 선택', True)]:
            button = QPushButton(title)
            button.clicked.connect(lambda _, d=directory: self.browse(d))
            form.addRow(button)
        self.path.textChanged.connect(changed)

    def browse(self, directory=False):
        path = QFileDialog.getExistingDirectory(self, '이어 수집할 데이터 폴더') if directory else QFileDialog.getOpenFileName(self, '이어 수집할 HDF5', '', 'HDF5 (*.hdf5 *.h5)')[0]
        if path:
            self.path.setText(path)
