"""Source task editor and inherited task state for Workflow execution."""
import copy
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QLineEdit, QListWidget, QListWidgetItem, QLabel
from ..real2sim.workspaces.tasks import catalog, object_ids, active_objects, workspace_key


class TaskSelector(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.definitions = {}; self.loading = False; self.inherited = None; self.review = False
        layout = QFormLayout(self); layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.objects = QListWidget(); self.objects.setMaximumHeight(115)
        self.object_count = QLabel()
        layout.addRow('Scene objects', self.objects); layout.addRow('Active Objects', self.object_count)
        self.conflict = None; self.workspace = None
        self.objects.itemChanged.connect(self.choose_objects)
        self.preset = QComboBox(); self.pick = QComboBox(); self.target = QComboBox()
        self.instruction = QLineEdit()
        self.scope = QComboBox()
        for label, value in [('선택 task 1개', 'single'), ('여러 task 선택', 'multiple'), ('Workspace 전체 task', 'all')]:
            self.scope.addItem(label, value)
        self.multiple = QListWidget(); self.multiple.setMaximumHeight(120)
        for label, widget in [('Task preset', self.preset), ('Pick object', self.pick), ('Target cup', self.target),
                              ('Language instruction', self.instruction), ('평가 범위', self.scope),
                              ('평가 task 목록', self.multiple)]:
            layout.addRow(label, widget)
        self.preset.currentIndexChanged.connect(self.choose_preset)
        self.pick.currentTextChanged.connect(self.choose_pair); self.target.currentTextChanged.connect(self.choose_pair)
        self.instruction.textEdited.connect(self.edit_instruction)
        self.scope.currentIndexChanged.connect(lambda: None if self.loading else self.changed.emit())
        self.multiple.itemChanged.connect(lambda: None if self.loading else self.changed.emit())

    def configure(self, workspace, stage, inherited=None, active_object_ids=None):
        same_workspace = getattr(self,'workspace_identity',None)==workspace_key(workspace)
        self.workspace = workspace; self.stage = stage; self.inherited_definitions = inherited
        self.workspace_identity = workspace_key(workspace)
        self.review = workspace['task'].get('mode') == 'geometry_review'
        previous = self.preset.currentData() if same_workspace else None
        previous_definitions = self.definitions if same_workspace else {}
        self.loading = True
        self.objects.clear()
        active = object_ids(workspace) if active_object_ids is None else active_object_ids
        if set(active) - set(object_ids(workspace)):
            self.loading = False
            raise ValueError('Saved object IDs are not present in this Workspace')
        for name in object_ids(workspace):
            display = workspace['profile']['objects'][name].get('display_name', name)
            item = QListWidgetItem(display); item.setData(Qt.ItemDataRole.UserRole,name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name in active else Qt.CheckState.Unchecked)
            self.objects.addItem(item)
        self.object_count.setText(f'{len(active)} / {len(object_ids(workspace))}')
        self.definitions = {t['task_id']: t for t in catalog(workspace) if t['pick_object_id'] in active}
        self.conflict = None
        # Preserve edited text while the same workspace definition remains selected.
        for key, value in previous_definitions.items():
            if key in self.definitions and value['randomization_spec'] == self.definitions[key]['randomization_spec']:
                self.definitions[key]['language_instruction'] = value['language_instruction']
        editable = not self.review and stage in ('source', 'sim_eval', 'real_eval')
        self.objects.setEnabled(stage in ('source','sim_eval') and not self.review)
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
        if chosen and chosen not in self.definitions:
            self.conflict = 'Task conflicts with active objects: '+str(chosen)+'; select an explicit active Task'
        self.preset.setCurrentIndex(-1 if self.conflict else max(0,self.preset.findData(chosen)))
        for widget in (self.preset, self.pick, self.target, self.instruction): widget.setEnabled(editable)
        for widget in (self.scope, self.multiple):
            widget.setVisible(stage in ('sim_eval', 'real_eval'))
            self.layout().labelForField(widget).setVisible(stage in ('sim_eval', 'real_eval'))
        self.loading = False; self.choose_preset()

    def choose_preset(self):
        if self.loading or self.preset.currentData() not in self.definitions: return
        self.conflict = None
        t = self.definitions[self.preset.currentData()]; self.loading = True
        self.pick.setCurrentText(t['pick_object_id']); self.target.setCurrentText(t['target_object_id'])
        self.instruction.setText(t['language_instruction']); self.loading = False
        self.changed.emit()

    def choose_pair(self):
        if self.loading: return
        match = next((key for key,t in self.definitions.items() if t['pick_object_id']==self.pick.currentText()
                      and t['target_object_id']==self.target.currentText()),None)
        if match:
            self.preset.setCurrentIndex(self.preset.findData(match)); self.choose_preset()

    def object_selection(self):
        values = [self.objects.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.objects.count())
                  if self.objects.item(i).checkState()==Qt.CheckState.Checked]
        return active_objects(self.workspace, values)

    def choose_objects(self):
        if self.loading or self.workspace is None:return
        values = [self.objects.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.objects.count())
                  if self.objects.item(i).checkState()==Qt.CheckState.Checked]
        self.configure(self.workspace,self.stage,self.inherited_definitions,values)
        self.changed.emit()

    def edit_instruction(self, text):
        if not self.loading and self.preset.currentData() in self.definitions:
            self.definitions[self.preset.currentData()]['language_instruction'] = text
            self.changed.emit()

    def selection(self, evaluation=False):
        if self.review: return []
        self.object_selection()
        if self.conflict: raise ValueError(self.conflict)
        if self.inherited is not None: return copy.deepcopy(self.inherited)
        scope = self.scope.currentData() if evaluation else 'single'
        if scope == 'all': keys = list(self.definitions)
        elif scope == 'multiple':
            keys = [self.multiple.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.multiple.count())
                    if self.multiple.item(i).checkState() == Qt.CheckState.Checked]
        else: keys = [self.preset.currentData()]
        if not keys or any(k not in self.definitions for k in keys): raise ValueError('Task를 선택하세요')
        return [copy.deepcopy(self.definitions[k]) for k in keys]
