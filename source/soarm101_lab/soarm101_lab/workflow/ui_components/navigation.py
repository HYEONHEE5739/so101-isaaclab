"""Task navigation, independent of the Operations stage ordering."""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem

GROUPS = (
    ('01. 데이터 준비', (('source', 'Source Demo'), ('replay', 'Replay'),
      ('annotate', 'Annotation'), ('datagen', 'DataGen'), ('curation', 'Data Curation'), ('convert', 'LeRobot 변환'))),
    ('02. 모델 학습', (('train:act', 'ACT'), ('train:smolvla', 'SmolVLA'), ('train:minivla', 'MiniVLA'))),
    ('03. 모델 평가', (('sim_eval', 'Sim Rollout'), ('real_eval', 'Real Rollout'))),
    ('04. 업로드', (('hf_dataset', 'Dataset 업로드'), ('hf_policy', 'Policy 업로드'))),
    ('05. 결과물 관리', (('artifacts:source', 'Workspace / Source'), ('artifacts:generated', 'Annotated / DataGen'),
      ('artifacts:dataset', 'LeRobot Dataset'), ('artifacts:policy', 'Model / Policy'),
      ('artifacts:checkpoint', 'Checkpoint'), ('artifacts:evaluation', 'Sim / Real Evaluation'),
      ('artifacts:hub', 'Hub 업로드 기록'))),
)


class WorkflowNavigation(QTreeWidget):
    selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('workflowNavigation')
        self.setHeaderHidden(True)
        self.setMinimumWidth(205)
        self.setMaximumWidth(260)
        self.items = {}
        for title, entries in GROUPS:
            group = QTreeWidgetItem(self, [title])
            group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            for key, label in entries:
                item = QTreeWidgetItem(group, [label])
                item.setData(0, Qt.ItemDataRole.UserRole, key)
                self.items[key] = item
            group.setExpanded(True)
        self.currentItemChanged.connect(self._selected)

    def _selected(self, item, previous):
        key = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if key:
            self.selected.emit(key)

    def select(self, key):
        self.blockSignals(True)
        self.setCurrentItem(self.items[key])
        self.blockSignals(False)
