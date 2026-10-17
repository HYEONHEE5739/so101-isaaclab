"""Read-only Registry explorer; checkpoint files stay children of their policy record."""
import json
from pathlib import Path
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
                             QTreeWidget, QTreeWidgetItem, QPlainTextEdit, QPushButton, QLabel)
from ...artifacts import artifact_name, artifact_description
from ..artifact_picker import partition, lineage_text
from .sim_eval import checkpoint_paths

COLLECTIONS = dict(source={'workspace', 'source'}, generated={'annotated', 'datagen'}, dataset={'dataset'},
                   policy={'policy'}, checkpoint={'policy'}, evaluation={'sim_eval', 'real_eval'}, hub={'hf_dataset', 'hf_policy'})


class ArtifactExplorer(QWidget):
    navigate = pyqtSignal(str, object)

    def __init__(self, context, controller, parent=None):
        super().__init__(parent)
        self.context, self.controller, self.registry = context, controller, context.registry
        self.key, self.stage, self.collection = 'artifacts', None, 'source'
        layout = QVBoxLayout(self)
        self.filters = {}
        form = QFormLayout()
        for key, label in [('name', '이름 / Artifact ID'), ('type', 'Type'), ('task', 'Task'),
                           ('backend', 'Backend'), ('workspace', 'Workspace'), ('partition', 'Partition'), ('status', 'Status')]:
            editor = QLineEdit()
            editor.setPlaceholderText(label)
            editor.textChanged.connect(self.refresh)
            form.addRow(label, editor)
            self.filters[key] = editor
        layout.addLayout(form)
        refresh = QPushButton('Registry 새로고침')
        refresh.clicked.connect(self.refresh)
        layout.addWidget(refresh)
        body = QHBoxLayout()
        layout.addLayout(body, 1)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['결과물 / Checkpoint', 'Type', 'Status'])
        body.addWidget(self.tree, 1)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        body.addWidget(self.details, 2)
        self.tree.currentItemChanged.connect(self.show_details)
        actions = QHBoxLayout()
        layout.addLayout(actions)
        self.actions = {}
        for key, label in [('folder', 'Folder 열기'), ('lineage', 'Lineage 보기'), ('curation', 'Curation으로'),
                           ('train:smolvla', 'Train으로'), ('sim_eval', 'Sim Eval로'), ('real_eval', 'Real Eval로')]:
            button = QPushButton(label)
            button.clicked.connect(lambda _, action=key: self.action(action))
            actions.addWidget(button)
            self.actions[key] = button
        self.note = QLabel('읽기 전용 · 파일 이동/삭제 및 Registry 수정 없음')
        layout.addWidget(self.note)

    def activate(self):
        self.refresh()

    def deactivate(self):
        pass

    def set_collection(self, collection):
        self.collection = collection
        self.refresh()

    def refresh(self):
        old = self.tree.currentItem()
        selected = old.data(0, Qt.ItemDataRole.UserRole) if old else None
        self.tree.blockSignals(True)
        self.tree.clear()
        self.records = {record['id']: record for record in self.registry.all()}
        for record in self.records.values():
            if record['type'] not in COLLECTIONS[self.collection]:
                continue
            values = dict(name=artifact_name(record)+' '+record['id'], type=record['type'],
                task=','.join(t['task_id'] for t in record.get('tasks', [])), backend=record.get('backend', ''),
                workspace=json.dumps(record.get('workspace', {}), ensure_ascii=False),
                partition=partition(record).get('split', ''), status=record.get('status', ''))
            if any(editor.text().casefold() not in values[key].casefold() for key,editor in self.filters.items()):
                continue
            item = QTreeWidgetItem(self.tree, [artifact_description(record), record['type'], record.get('status', '')])
            payload = dict(artifact_id=record['id'])
            item.setData(0, Qt.ItemDataRole.UserRole, payload)
            if selected == payload:
                self.tree.setCurrentItem(item)
            if record['type'] == 'policy':
                root = Path(record['path'])
                for path in checkpoint_paths(root):
                    child = QTreeWidgetItem(item, [str(path.relative_to(root)), 'checkpoint (policy 내부)', '파일 존재'])
                    data = dict(payload, checkpoint_file=str(path))
                    child.setData(0, Qt.ItemDataRole.UserRole, data)
                    if selected == data:
                        self.tree.setCurrentItem(child)
                # ACT/SmolVLA checkpoints remain directories, never fabricated Registry records.
                if root.name == 'pretrained_model' and root.parent.parent.name == 'checkpoints':
                    checkpoints = root.parent.parent
                elif root.name == 'pretrained_model' and root.parent.parent.parent.name == 'checkpoints':
                    checkpoints = root.parent.parent.parent
                else:
                    checkpoints = root/'checkpoints'
                for path in sorted(checkpoints.glob('*/pretrained_model')):
                    child = QTreeWidgetItem(item, [str(path), 'checkpoint (LeRobot)', '폴더 존재'])
                    child.setData(0, Qt.ItemDataRole.UserRole, dict(payload, checkpoint_directory=str(path)))
                item.setExpanded(self.collection == 'checkpoint')
        self.tree.blockSignals(False)
        if self.tree.currentItem() is None and self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        self.show_details()

    def show_details(self, *args):
        item = self.tree.currentItem()
        payload = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        record = self.records.get(payload['artifact_id']) if payload else None
        for button in self.actions.values():
            button.setEnabled(bool(record))
        if not record:
            self.details.clear()
            return
        children = [r['id'] for r in self.records.values() if record['id'] in r.get('parents', [])]
        details = dict(record, display_name=artifact_name(record), exists=Path(record['path']).exists(),
            source_partition=partition(record), used_by=children, selected_checkpoint=payload,
            input_contracts={key:record.get(key) for key in ('coordinates', 'representation', 'workspace', 'provenance')},
            output_contract=record.get('execution', {}).get('plan'))
        self.details.setPlainText(json.dumps(details, indent=2, ensure_ascii=False))
        valid = record.get('status', 'SUCCEEDED') == 'SUCCEEDED' and Path(record['path']).exists()
        for key, kinds in [('curation', {'source','annotated','datagen'}), ('train:smolvla', {'dataset'}),
                           ('sim_eval', {'policy','hf_policy'}), ('real_eval', {'policy','hf_policy'})]:
            self.actions[key].setEnabled(valid and record['type'] in kinds)
        if payload.get('checkpoint_directory'):
            self.actions['sim_eval'].setEnabled(False)
            self.actions['real_eval'].setEnabled(False)
            self.note.setText('LeRobot 중간 Checkpoint는 평가 화면의 폴더 등록으로 먼저 검증하여 가져오세요.')
        else:
            self.note.setText('읽기 전용 · 파일 이동/삭제 및 Registry 수정 없음')

    def action(self, action):
        item = self.tree.currentItem()
        if item is None:
            return
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        record = self.registry.get(payload['artifact_id'])
        if action == 'folder':
            path = Path(payload.get('checkpoint_directory') or payload.get('checkpoint_file') or record['path'])
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path if path.is_dir() else path.parent)))
        elif action == 'lineage':
            children = [r['id'] for r in self.registry.all() if record['id'] in r.get('parents', [])]
            self.details.setPlainText(lineage_text(self.registry, record)+'\nChild IDs:\n'+'\n'.join(children))
        elif self.actions[action].isEnabled():
            self.navigate.emit(action, payload)

    def poll(self):
        pass
