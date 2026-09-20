"""One shared semantic task selector for all workflow stages."""
import copy
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QLineEdit, QListWidget, QListWidgetItem, QLabel
from .tasks import catalog


class TaskSelector(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.definitions = {}; self.loading = False; self.inherited = None
        layout = QFormLayout(self); layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.preset = QComboBox(); self.pick = QComboBox(); self.target = QComboBox()
        self.instruction = QLineEdit()
        self.scope = QComboBox()
        for label, value in [('선택 task 1개', 'single'), ('여러 task 선택', 'multiple'), ('Workspace 전체 task', 'all')]:
            self.scope.addItem(label, value)
        self.multiple = QListWidget(); self.multiple.setMaximumHeight(120)
        self.summary = QLabel(); self.summary.setWordWrap(True)
        for label, widget in [('Task preset', self.preset), ('Pick object', self.pick), ('Target cup', self.target),
                              ('Language instruction', self.instruction), ('평가 범위', self.scope),
                              ('평가 task 목록', self.multiple), ('Task provenance', self.summary)]:
            layout.addRow(label, widget)
        self.preset.currentIndexChanged.connect(self.choose_preset)
        self.pick.currentTextChanged.connect(self.choose_pair); self.target.currentTextChanged.connect(self.choose_pair)
        self.instruction.textEdited.connect(self.edit_instruction)
        self.scope.currentIndexChanged.connect(lambda: self.changed.emit())
        self.multiple.itemChanged.connect(lambda: self.changed.emit())

    def configure(self, workspace, stage, inherited=None):
        previous = self.preset.currentData()
        previous_definitions = self.definitions
        self.loading = True
        self.definitions = {t['task_id']: t for t in catalog(workspace)}
        # Preserve edited text while the same workspace definition remains selected.
        for key, value in previous_definitions.items():
            if key in self.definitions and value['randomization_spec'] == self.definitions[key]['randomization_spec']:
                self.definitions[key]['language_instruction'] = value['language_instruction']
        editable = stage in ('source', 'sim_eval', 'real_eval')
        self.inherited = None if editable else copy.deepcopy(inherited or [])
        for t in inherited or []:
            if t['task_id'] in self.definitions:
                self.definitions[t['task_id']] = copy.deepcopy(t)
        self.preset.clear(); self.pick.clear(); self.target.clear(); self.multiple.clear()
        for key, t in self.definitions.items():
            self.preset.addItem(key, key)
            item = QListWidgetItem(key); item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if key == previous else Qt.CheckState.Unchecked)
            self.multiple.addItem(item)
        self.pick.addItems(list(dict.fromkeys(t['pick_object_id'] for t in self.definitions.values())))
        self.target.addItems(list(dict.fromkeys(t['target_object_id'] for t in self.definitions.values())))
        chosen = (inherited or [{}])[0].get('task_id') if not editable else previous
        self.preset.setCurrentIndex(max(0, self.preset.findData(chosen)))
        for widget in (self.preset, self.pick, self.target, self.instruction): widget.setEnabled(editable)
        for widget in (self.scope, self.multiple):
            widget.setVisible(stage in ('sim_eval', 'real_eval'))
            self.layout().labelForField(widget).setVisible(stage in ('sim_eval', 'real_eval'))
        self.summary.setText('명시적 객체 ID가 의미를 결정합니다. 문장은 파싱하지 않습니다.' if editable else
                             '입력 artifact에서 상속: ' + ', '.join(t['task_id'] for t in inherited or []))
        self.loading = False; self.choose_preset()

    def choose_preset(self):
        if self.loading or self.preset.currentData() not in self.definitions: return
        t = self.definitions[self.preset.currentData()]; self.loading = True
        self.pick.setCurrentText(t['pick_object_id']); self.target.setCurrentText(t['target_object_id'])
        self.instruction.setText(t['language_instruction']); self.loading = False
        self.changed.emit()

    def choose_pair(self):
        if self.loading: return
        key = self.pick.currentText() + '_to_' + self.target.currentText()
        if key in self.definitions:
            self.definitions[key]['language_instruction'] = f"Pick up the {self.pick.currentText().removeprefix('cube_')} cube and place it in cup {self.target.currentText().removeprefix('cup_').upper()}."
            self.preset.setCurrentIndex(self.preset.findData(key)); self.choose_preset()

    def edit_instruction(self, text):
        if not self.loading and self.preset.currentData() in self.definitions:
            self.definitions[self.preset.currentData()]['language_instruction'] = text
            self.changed.emit()

    def selection(self, evaluation=False):
        if self.inherited is not None: return copy.deepcopy(self.inherited)
        scope = self.scope.currentData() if evaluation else 'single'
        if scope == 'all': keys = list(self.definitions)
        elif scope == 'multiple':
            keys = [self.multiple.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.multiple.count())
                    if self.multiple.item(i).checkState() == Qt.CheckState.Checked]
        else: keys = [self.preset.currentData()]
        if not keys or any(k not in self.definitions for k in keys): raise ValueError('Task를 선택하세요')
        return [copy.deepcopy(self.definitions[k]) for k in keys]
