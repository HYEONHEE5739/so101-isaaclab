"""One guided/independent workflow panel backed by Operations, not shell strings."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PyQt6.QtCore import QUrl, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QPixmap, QShortcut, QKeySequence
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
                            QPushButton, QLineEdit, QPlainTextEdit, QTextBrowser, QFileDialog, QMessageBox, QGridLayout, QScrollArea, QCheckBox, QSplitter, QToolButton, QDialog, QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem, QSizePolicy)
from .paths import validate_storage_name
from .paths import input_directory, discover_inputs
from .operations import Operations, STAGES, default_python
PATH_STAGES = frozenset(STAGES)
from ..real2sim.workspaces.package import ROOT, load, metadata
from ..real2sim.storage import atomic

LABELS = {'source': 'Source Demo',
          'replay': 'Replay', 'annotate': 'Annotation', 'datagen': 'Mimic / Datagen',
          'convert': 'LeRobot 변환', 'train': '모델 학습', 'validate': 'Offline Validation', 'hf_dataset': 'Dataset 업로드',
          'hf_policy': 'Policy 업로드', 'sim_eval': 'Sim 평가', 'real_eval': 'Real 평가 (모터 동작)'}


def stage_formats(stage):
    if stage=='source':return ('workspace',)
    if stage in ('sim_eval','real_eval','hf_policy','validate'):return ('lerobot-policy','minivla-policy','minivla-checkpoint')
    if stage in ('train','hf_dataset','merge'):return ('lerobot','canonical')
    return ('hdf5','canonical')



class WorkflowPanel(QWidget):
    workspaceChanged = pyqtSignal()

    @property
    def workspace_path(self):
        return self._workspace_path

    @workspace_path.setter
    def workspace_path(self, path):
        path = str(path)
        if path != self._workspace_path:
            self._workspace_path = path
            self.workspaceChanged.emit()

    def __init__(self, run_worker, parent=None, *, root=None):
        super().__init__(parent)
        from .theme import STYLE
        self.setStyleSheet(STYLE)
        self.run_worker = run_worker; self.root=Path(root or ROOT/'outputs/workflow'); self.service=Operations(root=self.root)
        self.busy = False
        self.task_setup_error = None
        self.input_folders = {}
        self.input_selections = {}
        self.inputs_stage = None
        self._workspace_path = ''
        self._display_workspace = None
        self.scene_cameras = ()
        self._preview_frames = {}
        self.worker_python = default_python()
        self._ready_generation = 0
        self._ready_future = None
        self._ready_request = None
        self._ready_executor = None
        self._discovery_generation=0;self._discovery_request=None;self._discovery_future=None
        self._checkpoint_generation=0;self._checkpoint_future=None
        self._input_generation=0;self._input_request=None;self._input_future=None
        self._path_details=None;self._applying_context=False
        self._ready_timer = QTimer(self)
        self._ready_timer.setSingleShot(True)
        self._ready_timer.setInterval(200)
        self._ready_timer.timeout.connect(self._start_ready_check)
        self.saved_stage_options = {}
        self.resume_inputs = {}
        self.storage_names = {}
        self.options_context = None
        self.configured_stage = None
        from .view import CameraView, LogTail
        self.log_tail = LogTail(); self.last_result = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12); outer.setSpacing(16)
        header = QWidget(); header.setObjectName('dashboardHeader')
        header_layout = QHBoxLayout(header)
        title_box = QVBoxLayout()
        brand = QLabel('SO-101  /  WORKSPACE'); brand.setObjectName('eyebrow')
        self.dashboard_title = QLabel('Source Demo'); self.dashboard_title.setObjectName('dashboardTitle')
        title_box.addWidget(brand); title_box.addWidget(self.dashboard_title)
        header_layout.addLayout(title_box, 1)
        settings_toggle = QPushButton('실행 설정'); settings_toggle.setCheckable(True); settings_toggle.setChecked(True)
        header_layout.addWidget(settings_toggle)
        outer.addWidget(header)
        body = QHBoxLayout(); body.setSpacing(16); outer.addLayout(body, 1)
        self.navigation_panel = QWidget(); self.navigation_panel.setObjectName('navigationPanel')
        self.navigation_panel.setFixedWidth(200)
        nav_layout = QVBoxLayout(self.navigation_panel)
        nav_title = QLabel('WORKFLOW'); nav_title.setObjectName('eyebrow'); nav_layout.addWidget(nav_title)
        self.navigation = QListWidget(); self.navigation.setObjectName('stageNavigation')
        nav_layout.addWidget(self.navigation, 1)
        curate = QPushButton('Data Curation')
        curate.clicked.connect(self.open_curation)
        nav_layout.addWidget(curate)
        nav_note = QLabel('단계를 선택하고\n입력을 확인한 뒤 실행하세요.'); nav_note.setWordWrap(True); nav_note.setObjectName('mutedLabel')
        nav_layout.addWidget(nav_note)
        body.addWidget(self.navigation_panel)
        self.splitter = QSplitter(Qt.Orientation.Horizontal); body.addWidget(self.splitter, 1)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setMinimumWidth(280)
        self.splitter.addWidget(scroll)
        viewer = QWidget(); display = QVBoxLayout(viewer); self.splitter.addWidget(viewer)
        self.splitter.setStretchFactor(0, 0); self.splitter.setStretchFactor(1, 1)
        self.splitter.insertWidget(0, viewer)
        self.splitter.setStretchFactor(0, 1); self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([850, 390])
        scroll.setObjectName('settingsPanel'); viewer.setObjectName('workspaceCanvas')
        scroll.setMinimumWidth(320)
        settings_toggle.toggled.connect(scroll.setVisible)
        display.setContentsMargins(0, 0, 0, 0); display.setSpacing(12)
        content = QWidget(); scroll.setWidget(content)
        layout = QVBoxLayout(content)
        settings_title = QLabel('실행 설정'); settings_title.setObjectName('sectionTitle'); layout.addWidget(settings_title)
        self.section = QComboBox(); self.section.addItems(['Sim 작업 파이프라인', '실제 로봇 평가'])
        nav_layout.insertWidget(1, self.section)
        form = QFormLayout(); form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); layout.addLayout(form)
        self.main_form = form
        self.storage_name = QLineEdit()
        self.storage_name.setPlaceholderText('필수 · 예: 테스트_1')
        self.storage_name.setToolTip('확장자 없이 한글/영문/숫자/_/-로 입력하세요. 현재 단계의 결과 폴더와 목록 이름에 적용하며 기존 이름은 덮어쓰지 않습니다.')
        self.storage_name.setClearButtonEnabled(True)
        form.addRow('저장 이름 (필수)', self.storage_name)
        self.learning_output=QLineEdit();self.learning_output.setPlaceholderText('생략하면 기본 저장 영역 / Resume 원래 위치')
        self.learning_output.textChanged.connect(self.update_ready)
        form.addRow('학습·검증 출력 부모 폴더',self.learning_output)
        self.storage_name.textChanged.connect(self.update_ready)
        self.mode = QComboBox()
        self.mode.addItem('Operator — 통합 UI에서 영상/조작', 'operator')
        self.mode.addItem('Developer — 시뮬레이터 창 함께 실행', 'developer')
        form.addRow('실행 모드', self.mode)
        self.real_enabled = QCheckBox('REAL 카메라 영상 연결 (로봇 모터 연결 아님)')
        form.addRow(self.real_enabled)
        self.real_enabled.hide()
        self.real_side = QLineEdit('/dev/cam_side'); self.real_wrist = QLineEdit('/dev/cam_wrist')
        self.real_side.hide(); self.real_wrist.hide()
        self.stage = QComboBox()
        for k in ('source', 'replay', 'annotate', 'datagen', 'convert', 'train', 'validate', 'sim_eval', 'hf_dataset', 'hf_policy'): self.stage.addItem(LABELS[k], k)
        form.addRow('단계', self.stage)
        self.stage.hide(); form.labelForField(self.stage).hide()
        self.inputs = QComboBox(); self.inputs.setMinimumContentsLength(16); self.inputs.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon); self.input_label=QLabel('사용할 데이터 / 모델');form.addRow(self.input_label, self.inputs)
        self.data_select = QPushButton('데이터 선택')
        self.data_select.clicked.connect(lambda: self.guard(self.browse_data))
        form.addRow(self.data_select)
        self.merge_button = QPushButton('변환된 LeRobot 데이터셋 병합')
        self.merge_button.clicked.connect(lambda: self.guard(self.select_merge_datasets))
        form.addRow(self.merge_button)
        from .task_ui import TaskSelector
        self.task_selector = TaskSelector(); layout.addWidget(self.task_selector)
        self.task_selector.changed.connect(self.objects_changed)
        self.task_selector.layout().setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        advanced = QToolButton(); advanced.setText('실행 설정 접기 / 펼치기'); advanced.setCheckable(True)
        layout.addWidget(advanced)
        self.primary = QWidget(); primary_form = QFormLayout(self.primary)
        primary_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.port = QLineEdit('/dev/so101_leader'); primary_form.addRow('Leader port', self.port)
        self.episodes = QSpinBox(); self.episodes.setRange(1, 100000)
        self.episodes.setToolTip('저장할 전체 에피소드 목표입니다. 이어 수집은 기존 개수를 포함합니다. 기존 20개에 5개를 추가하려면 25를 입력하세요.')
        primary_form.addRow('총 목표 Episodes (기존 포함)', self.episodes)
        self.episode_time = QDoubleSpinBox(); self.episode_time.setRange(.1, 3600); self.episode_time.setValue(20); self.episode_time.setSuffix(' s')
        self.reset_time = QDoubleSpinBox(); self.reset_time.setRange(0, 600); self.reset_time.setValue(5); self.reset_time.setSuffix(' s')
        primary_form.addRow('Episode Time', self.episode_time); primary_form.addRow('Reset Time', self.reset_time)
        self.annotation_mode = QComboBox(); self.annotation_mode.addItems(['자동 Annotation', '수동 Annotation'])
        primary_form.addRow('Annotation 방식', self.annotation_mode)
        self.architecture = QComboBox()
        self.architecture.addItem('SmolVLA — smolvla_base 파인튜닝', 'smolvla')
        self.architecture.addItem('ACT', 'act')
        primary_form.addRow('학습 모델', self.architecture)
        self.architecture.currentIndexChanged.connect(self.change_model)
        self.validation_input=QComboBox();self.validation_input.addItem('별도 검증 데이터 없음',None)
        self.validation_input.setToolTip('선택한 Val은 학습 완료 후 별도 Offline Validation에서 평가합니다. Train 전체를 사용합니다.')
        primary_form.addRow('Validation Dataset (선택)',self.validation_input)
        self.validation_browse=QPushButton('Validation Dataset 폴더 선택')
        primary_form.addRow('',self.validation_browse)
        self.validation_browse.clicked.connect(self.browse_validation)
        self.validation_input.currentIndexChanged.connect(self.update_ready)
        self.checkpoint_choice=QComboBox();primary_form.addRow('평가 Checkpoint (MiniVLA)',self.checkpoint_choice)
        self.checkpoint_choice.currentIndexChanged.connect(self.update_ready)

        self.resume_training = QCheckBox('이어서 학습')
        self.resume_training.setToolTip('모델·optimizer·step 복원. MiniVLA는 새 실행 폴더에 저장하고 ACT/SmolVLA는 기존 학습 폴더를 사용합니다.')
        primary_form.addRow('학습 방식', self.resume_training)
        self.resume_path = QLineEdit()
        self.resume_path.setPlaceholderText('MiniVLA: .pt 파일 · ACT/SmolVLA: checkpoint 폴더')
        self.resume_browse = QPushButton('체크포인트 선택')
        primary_form.addRow('Resume checkpoint', self.resume_path)
        primary_form.addRow('폴더 선택', self.resume_browse)
        self.resume_browse.clicked.connect(self.browse_checkpoint)
        self.resume_models=QComboBox();primary_form.addRow('저장 모델에서 Resume',self.resume_models)
        self.resume_models.activated.connect(self.select_resume_model)
        self.resume_training.toggled.connect(self.update_ready)
        self.resume_path.textChanged.connect(self.update_ready)
        self.resume_data = QLineEdit()
        self.resume_data.setClearButtonEnabled(True)
        self.resume_data.setPlaceholderText('선택 사항: 기존 HDF5 파일 또는 데이터셋 폴더')
        self.resume_data_browse = QPushButton('이어 수집할 데이터 선택 (HDF5 / 폴더)')
        primary_form.addRow('기존 완료 데이터', self.resume_data)
        primary_form.addRow('이어 수집', self.resume_data_browse)
        self.resume_data.textChanged.connect(self.update_ready)
        self.resume_data_browse.clicked.connect(lambda: self.guard(self.browse_resume_data))
        self.eval_seed_mode = QComboBox()
        self.eval_seed_mode.addItem('고정 seed — 모델 비교용', 'fixed')
        self.eval_seed_mode.addItem('실행마다 랜덤 seed', 'random')
        self.eval_seed = QSpinBox()
        self.eval_seed.setRange(0, 2147483647)
        primary_form.addRow('평가 배치 seed 방식', self.eval_seed_mode)
        primary_form.addRow('평가 seed', self.eval_seed)
        self.eval_seed_mode.currentIndexChanged.connect(
            lambda: self.eval_seed.setEnabled(self.eval_seed_mode.currentData() == 'fixed'))
        self.eval_seed_mode.currentIndexChanged.connect(self.update_ready)
        self.eval_seed.valueChanged.connect(self.update_ready)
        layout.insertWidget(layout.count() - 1, self.primary)
        self.port.textChanged.connect(self.update_ready); self.episodes.valueChanged.connect(self.update_ready)
        self.annotation_mode.currentIndexChanged.connect(self.configure_stage)
        self.episode_time.valueChanged.connect(self.update_ready); self.reset_time.valueChanged.connect(self.update_ready)
        self.advanced_panel = QWidget(); options_layout = QVBoxLayout(self.advanced_panel)
        options_layout.setContentsMargins(0, 0, 0, 0)
        self.options_help = QTextBrowser(); self.options_help.setMinimumHeight(200)
        self.options_help.hide() # retained schema reference for tooling
        from .settings_form import SettingsForm
        self.options = SettingsForm()
        options_layout.addWidget(QLabel('체크한 항목을 적용합니다 · 해제하면 자동 기본값'))
        options_layout.addWidget(self.options)
        restore = QPushButton('설정을 기본값으로 복원')
        restore.clicked.connect(self.reset_options); options_layout.addWidget(restore)
        self.advanced_panel.show(); advanced.setChecked(True)
        advanced.toggled.connect(self.advanced_panel.setVisible); layout.addWidget(self.advanced_panel)
        layout.addStretch()
        self.status = QLabel(); self.status.setWordWrap(True); display.addWidget(self.status)
        self.run = QPushButton('선택 단계 실행'); self.run.setObjectName('primaryAction'); self.run.clicked.connect(lambda: self.guard(self.execute)); display.addWidget(self.run)
        self.stop = QPushButton('현재 workflow 중지'); self.stop.setObjectName('stopAction'); self.stop.clicked.connect(self.service.cancel); display.addWidget(self.stop)
        self.media_splitter = QSplitter(Qt.Orientation.Vertical); display.addWidget(self.media_splitter, 1)
        images = QWidget(); grid = QGridLayout(images); self.media_splitter.addWidget(images)
        self.images = images; self.image_grid = grid
        self.feeds = {}; self.feed_titles = {}
        for index, name in enumerate(('sim_side', 'sim_wrist', 'real_side', 'real_wrist', 'sim_overview')):
            label = CameraView(name.upper() + ' — 영상 없음')
            title = QLabel(name.upper().replace('_', ' ')); title.setMaximumHeight(24); self.feed_titles[name] = title
            grid.addWidget(title, 0, index % 2); grid.addWidget(label, 1, index % 2)
            if name.startswith('real_'): label.hide(); title.hide()
            self.feeds[name] = label
        grid.setRowStretch(1, 1)
        self.stage_summary = QLabel(); self.stage_summary.setWordWrap(True); display.insertWidget(3, self.stage_summary)
        self.task_state = QLabel(); self.task_state.setWordWrap(True)
        self.task_state.setSizePolicy(QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Minimum)
        display.insertWidget(4, self.task_state)
        log_panel = QWidget(); log_layout = QVBoxLayout(log_panel); self.media_splitter.addWidget(log_panel)
        log_tools = QHBoxLayout(); log_layout.addLayout(log_tools)
        log_tools.addWidget(QLabel('진행 / 오류 로그'))
        self.raw_log = QCheckBox('전체 로그'); log_tools.addWidget(self.raw_log)
        for title, fn in [('원본 로그 열기', self.open_log), ('결과 상세', self.show_result)]:
            button = QPushButton(title); button.clicked.connect(fn); log_tools.addWidget(button)
        self.training_loss_button = QPushButton('학습 Loss 그래프')
        self.training_loss_button.clicked.connect(self.show_training_loss)
        log_tools.addWidget(self.training_loss_button)
        self.result = QPlainTextEdit(); self.result.setReadOnly(True); self.result.setMaximumBlockCount(4000)
        log_layout.addWidget(self.result)
        self.media_splitter.setStretchFactor(0, 3); self.media_splitter.setStretchFactor(1, 1)
        self.media_splitter.setSizes([500, 180])
        self.preview_status = QLabel('SIM SIDE / WRIST · 영상과 진행 로그를 보면서 조작하세요.')
        self.preview_status.setWordWrap(True); display.addWidget(self.preview_status)
        controls = QHBoxLayout(); display.insertLayout(3, controls); self.controls = {}
        for command, title in [('save', '저장 / 다음 (→)'), ('discard', '리셋 / 폐기 (←)'), ('quit', '기록 종료'),
                               ('resume', '재생'), ('pause', '일시정지'), ('step', '한 스텝'),
                               ('mark', 'Subtask 표시'), ('skip', 'Episode 건너뛰기')]:
            button = QPushButton(title); button.setEnabled(False)
            button.clicked.connect(lambda _, c=command: self.guard(lambda: self.control(c)))
            controls.addWidget(button); self.controls[command] = button
        # Reorganize existing controls into a compact action bar and rounded cards.
        display.removeWidget(self.status); display.removeWidget(self.run); display.removeWidget(self.stop)
        action_card = QWidget(); action_card.setObjectName('actionCard')
        action_layout = QHBoxLayout(action_card)
        action_layout.addWidget(self.status, 1); action_layout.addWidget(self.run); action_layout.addWidget(self.stop)
        display.insertWidget(0, action_card)
        images.setObjectName('previewCard'); log_panel.setObjectName('logCard')
        self.stage_summary.setObjectName('mutedLabel')
        self.task_state.setObjectName('taskBadge')
        from .view import AnnotationProgress
        self.annotation_progress=AnnotationProgress()
        display.insertWidget(4,self.annotation_progress)
        self.visible_feeds=();self.job_status_view=None
        self.result.setObjectName('console')
        for card in (header, self.navigation_panel, action_card, images, log_panel):
            card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content.setObjectName('settingsContent')
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.status.setStyleSheet('background: transparent; padding: 6px;')
        for label in self.feeds.values():
            label.setStyleSheet('background: #141b23; border-radius: 14px; color: #8b9aaa;')
        self.inputs.currentIndexChanged.connect(self.update_input_tooltip)
        self.validation_input.currentIndexChanged.connect(self.remember_inputs)
        self.navigation.currentRowChanged.connect(self.select_navigation)
        self.sync_navigation()
        self.timer = QTimer(self); self.timer.timeout.connect(self.poll_preview); self.timer.start(33)
        self.shortcuts = []
        for key, command in [('Right', 'save'), ('Left', 'discard')]:
            shortcut = QShortcut(QKeySequence(key), viewer)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(lambda c=command: self.controls[c].click() if self.controls[c].isEnabled() else None)
            self.shortcuts.append(shortcut)
        viewer.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.section.currentIndexChanged.connect(self.change_section)
        settings = self.root / 'ui.json'
        if settings.exists():
            saved = json.loads(settings.read_text()); self.workspace_path = saved.get('workspace', '')
            self.worker_python = saved.get('python', default_python())
            self.mode.setCurrentIndex(max(0, self.mode.findData(saved.get('mode', 'operator'))))
        self.stage.currentIndexChanged.connect(self.changed)
        self.inputs.currentIndexChanged.connect(self.sync_tasks)
        self.options.textChanged.connect(self.update_ready)
        for combo in self.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
        self.setup_backends(header_layout)
        self.changed()

    def setup_backends(self, header):
        self.backend = QComboBox()
        self.backend.addItem('Isaac Sim / Isaac Lab', 'isaac')
        self.backend.addItem('MuJoCo', 'mujoco')
        header.insertWidget(1, self.backend)
        self.backend_pythons = {'isaac': self.worker_python, 'mujoco': default_python()}
        saved_path = self.root / 'ui.json'
        saved = json.loads(saved_path.read_text()) if saved_path.exists() else {}
        self.object_selections = saved.get('object_selections', {})
        self.input_folders = saved.get('input_folders', {})
        self.input_selections = saved.get('input_selections', {})
        self.backend_pythons.update(saved.get('backend_pythons', {}))
        self.backend.setCurrentIndex(max(0, self.backend.findData(saved.get('backend', 'isaac'))))
        self.previous_backend = self.backend.currentData()
        self.worker_python = self.backend_pythons[self.previous_backend]
        self.backend.currentIndexChanged.connect(self.change_backend)

    def is_mujoco(self):
        return hasattr(self, 'backend') and self.backend.currentData() == 'mujoco'

    def change_backend(self):
        if self.busy: return
        self.backend_pythons[self.previous_backend] = self.worker_python
        self.previous_backend = self.backend.currentData()
        self.worker_python = self.backend_pythons[self.previous_backend]
        self.refresh()
        self.save_backend_settings()

    def save_backend_settings(self):
        self.backend_pythons[self.backend.currentData()] = self.worker_python
        atomic(self.root / 'ui.json', {
            'workspace': self.workspace_path, 'python': self.worker_python, 'mode': self.mode.currentData(),
            'backend': self.backend.currentData(), 'backend_pythons': self.backend_pythons,
            'object_selections': getattr(self,'object_selections',{}), 'input_folders':self.input_folders,
            'input_selections':self.input_selections})

    def guard(self, fn):
        try: fn()
        except Exception as exc: QMessageBox.warning(self, 'Workflow', str(exc))

    def open_curation(self):
        from .curation_ui import CurationDialog
        dialog=CurationDialog(self.root,self,workspace=self.task_selector.workspace_identity if self.task_selector.workspace is not None else None)
        selected=self.inputs.currentData()
        if selected and self.stage.currentData() in ('replay','annotate','datagen','convert'):
            QTimer.singleShot(0,lambda:dialog.load_data(selected))
        else:QTimer.singleShot(0,dialog.open_latest)
        dialog.exec()
        if getattr(dialog,'saved_collection',None) and self.stage.currentData() in ('replay','annotate','convert'):
            self.input_folders[self.stage.currentData()]=dialog.saved_collection
            self.save_backend_settings()
        self.refresh()

    def remember_inputs(self):
        stage=self.stage.currentData()
        if self.inputs_stage!=stage:return
        selected={('path' if stage in PATH_STAGES else 'artifact'):self.inputs.currentData()}
        if stage in ('train','validate'):selected['validation']=self.validation_input.currentData()
        if self.input_selections.get(stage)!=selected:
            self.input_selections[stage]=selected
            if hasattr(self,'backend'):self.save_backend_settings()

    def update_input_tooltip(self):
        self.inputs.setToolTip(self.inputs.currentData() or '')
        self.refresh_training_choices();self.remember_inputs()

    def sync_navigation(self):
        self.navigation.blockSignals(True)
        self.navigation.clear()
        for i in range(self.stage.count()):
            item = QListWidgetItem(f'{i + 1:02d}   {self.stage.itemText(i)}')
            self.navigation.addItem(item)
        self.navigation.setCurrentRow(self.stage.currentIndex())
        self.navigation.blockSignals(False)

    def select_navigation(self, row):
        if row >= 0:
            self.stage.setCurrentIndex(row)

    def changed(self):
        stage = self.stage.currentData()
        if not stage: return
        previous = self.configured_stage
        if previous != stage:
            if previous:self.storage_names[previous] = self.storage_name.text()
            self.storage_name.blockSignals(True)
            self.storage_name.setText(self.storage_names.get(stage, ''))
            self.storage_name.blockSignals(False)
        if previous in ('source', 'datagen'):
            self.resume_inputs[previous] = self.resume_data.text()
        if previous != stage:
            self.resume_data.blockSignals(True)
            self.resume_data.setText(self.resume_inputs.get(stage, ''))
            self.resume_data.blockSignals(False)
        if self.configured_stage and self.configured_stage != stage:
            self.saved_stage_options[self.options_context] = json.loads(self.options.toPlainText())
        self.configured_stage = stage
        self.sync_navigation()
        self.dashboard_title.setText(LABELS.get(stage, stage))
        self.configure_stage()
        self.learning_output.setVisible(stage in ('train','validate'));self.main_form.labelForField(self.learning_output).setVisible(stage in ('train','validate'))
        self.primary.setVisible(stage in ('source', 'annotate', 'train', 'validate', 'datagen', 'sim_eval'))
        primary_form = self.primary.layout()
        validation_visible=stage in ('train','validate')
        self.validation_browse.setVisible(validation_visible)
        self.validation_input.setVisible(validation_visible)
        self.resume_models.setVisible(stage=='train');primary_form.labelForField(self.resume_models).setVisible(stage=='train')
        primary_form.labelForField(self.validation_input).setVisible(validation_visible)
        for widget, visible in ((self.port, stage == 'source'), (self.episodes, stage == 'source'), (self.episode_time, stage == 'source'), (self.reset_time, stage == 'source'), (self.annotation_mode, stage == 'annotate'), (self.architecture, stage == 'train'), (self.resume_training, stage == 'train'), (self.resume_path, stage == 'train'), (self.resume_browse, stage == 'train'), (self.resume_data, stage in ('source', 'datagen')), (self.resume_data_browse, stage in ('source', 'datagen')), (self.eval_seed_mode, stage == 'sim_eval'), (self.eval_seed, stage == 'sim_eval')):
            widget.setVisible(visible); primary_form.labelForField(widget).setVisible(visible)
        from .option_help import help_html
        self.options_help.setHtml(help_html(stage))
        self.reset_options(restore_saved=True)
        self.refresh()

    def refresh(self):
        self._input_generation+=1;self._input_request=None
        self._refresh_episode_inputs()

    def _refresh_episode_inputs(self):
        self._discovery_generation+=1;self._discovery_request=None
        stage=self.stage.currentData()
        selected=self.inputs.currentData() if self.inputs_stage==stage else self.input_selections.get(stage,{}).get('path')
        if not selected and stage=='source':selected=self.workspace_path
        self.inputs_stage=stage
        folder=self.input_folders.get(stage) or str(input_directory(stage,self.root,workspace_root=ROOT/'outputs/real2sim/environments'))
        self.data_select.setToolTip('입력 저장소: '+folder)
        self.inputs.blockSignals(True);self.inputs.clear();self.inputs.addItem('Dataset 목록을 읽는 중…',None);self.inputs.blockSignals(False)
        self.task_selector.workspace=None;self._path_details=None;self.task_setup_error=None
        self._input_request=(self._input_generation,stage,folder,selected)
        self.update_ready();self._poll_episode_inputs();self._request_model_discovery()

    def _poll_episode_inputs(self):
        if self._input_future is not None:
            generation,stage,selected,future=self._input_future
            if not future.done():return
            self._input_future=None
            try:items=future.result();error=''
            except Exception as exc:items=[];error=str(exc)
            if generation==self._input_generation and self.stage.currentData()==stage:
                self.inputs.blockSignals(True);self.inputs.clear();self.inputs.addItem('입력 Path 선택',None)
                for item in items:
                    if item['format'] not in stage_formats(stage):continue
                    self.inputs.addItem(item['name'],item['path'])
                    self.inputs.setItemData(self.inputs.count()-1,item['path']+'\n형식 후보 · Worker에서 실제 계약 검사',Qt.ItemDataRole.ToolTipRole)
                index=self.inputs.findData(selected) if selected else (1 if self.inputs.count()>1 else 0)
                self.inputs.setCurrentIndex(max(0,index));self.inputs.blockSignals(False)
                if stage in ('train','validate'):
                    old=self.input_selections.get(stage,{}).get('validation')
                    self.validation_input.blockSignals(True);self.validation_input.clear();self.validation_input.addItem('별도 검증 데이터 없음',None)
                    for item in items:
                        if item['format']=='lerobot':self.validation_input.addItem(item['name'],item['path'])
                    if old and self.validation_input.findData(old)<0:self.validation_input.addItem(Path(old).name,old)
                    self.validation_input.setCurrentIndex(max(0,self.validation_input.findData(old)));self.validation_input.blockSignals(False)
                self.update_input_tooltip();self.sync_tasks()
                if error:self.status.setText('NOT_READY — '+error)
        if self._input_future is None and self._input_request is not None:
            generation,stage,folder,selected=self._input_request;self._input_request=None
            self._ensure_executor()
            def scan():
                category='workspace' if stage=='source' else 'policy' if stage in ('sim_eval','real_eval','hf_policy','validate') else 'dataset'
                items=discover_inputs(folder,category=category)
                if selected and not any(item['path']==selected for item in items):
                    items.extend(discover_inputs(selected,category=category))
                return items
            self._input_future=(generation,stage,selected,self._ready_executor.submit(scan))

    def _ensure_executor(self):
        if self._ready_executor is None:
            self._ready_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='workflow-readiness')
            executor=self._ready_executor
            self.destroyed.connect(lambda:executor.shutdown(wait=False,cancel_futures=True))

    def _request_model_discovery(self):
        self._discovery_generation+=1
        self._discovery_request=(self._discovery_generation,self.root/'models') if self.stage.currentData()=='train' else None
        self._poll_model_discovery()

    def _poll_model_discovery(self):
        if self._discovery_future is not None:
            generation,future=self._discovery_future
            if not future.done():return
            self._discovery_future=None
            try:items=future.result();error=''
            except Exception as exc:items=[];error=str(exc)
            if generation==self._discovery_generation and self.stage.currentData()=='train':
                old=self.resume_models.currentData();self.resume_models.blockSignals(True)
                self.resume_models.clear();self.resume_models.addItem('저장 모델 선택 (또는 파일 선택)',None)
                for item in items:
                    self.resume_models.addItem(item['name'],item['path'])
                    self.resume_models.setItemData(self.resume_models.count()-1,item['path']+'\n형식 후보 · 실행 시 별도 검증',Qt.ItemDataRole.ToolTipRole)
                self.resume_models.setCurrentIndex(max(0,self.resume_models.findData(old)))
                self.resume_models.setToolTip(error);self.resume_models.blockSignals(False)
        if self._discovery_future is None and self._discovery_request is not None:
            generation,folder=self._discovery_request;self._discovery_request=None
            self._ensure_executor()
            self._discovery_future=(generation,self._ready_executor.submit(discover_inputs,folder,category='policy'))

    def _poll_checkpoint_metadata(self):
        if self._checkpoint_future is None:return
        generation,path,future=self._checkpoint_future
        if not future.done():return
        self._checkpoint_future=None
        if generation!=self._checkpoint_generation or self.resume_path.text().strip()!=path or self.stage.currentData()!='train':return
        try:saved=future.result()
        except Exception as exc:
            self.resume_path.setToolTip(str(exc));self.status.setText('NOT_READY — '+str(exc));self.run.setEnabled(False);return
        if saved['architecture']=='minivla':
            self.status.setText('NOT_READY — MiniVLA 신규 학습/학습 Resume는 비활성화되어 있습니다. 기존 추론은 지원합니다.');self.run.setEnabled(False);return
        self.architecture.setCurrentIndex(self.architecture.findData(saved['architecture']))
        opts=json.loads(self.options.toPlainText())
        if saved.get('steps') is not None:opts['steps']=saved['steps']
        self.options.setPlainText(json.dumps(opts,indent=2,ensure_ascii=False))
        self.resume_training.setChecked(True)
        self.update_ready()



    def browse_data(self):
        if self.busy:raise ValueError('Workflow 실행이 끝난 뒤 데이터를 선택하세요.')
        directory=input_directory(self.stage.currentData(),self.root)
        if self.stage.currentData() in ('source','sim_eval','real_eval','hf_policy','train','hf_dataset'):
            folder=QFileDialog.getExistingDirectory(self,'데이터 선택 · '+LABELS[self.stage.currentData()],str(directory))
            if folder:self.select_data(folder)
        else:
            from .curation_ui import DatasetPicker
            picker=DatasetPicker(self,str(directory),allow_collection=True)
            if picker.exec()==QDialog.DialogCode.Accepted:self.select_data(picker.selectedFiles()[0])

    def select_data(self, folder):
        if self.busy:raise ValueError('Workflow 실행이 끝난 뒤 데이터를 선택하세요.')
        from .paths import input_format
        folder=str(Path(folder).expanduser().resolve());stage=self.stage.currentData()
        self.input_folders[stage]=folder;self.inputs_stage=None
        self.input_selections[stage]={'path':folder if input_format(folder) in stage_formats(stage) else None}
        if stage=='source':self.workspace_path=folder
        self.refresh();self.save_backend_settings()

    def select_workspace(self, path):
        """Select a published Workspace; background preflight owns loading/Hash."""
        if self.busy:raise ValueError('Workflow 실행이 끝난 뒤 Scene을 선택하세요.')
        self.workspace_path=str(Path(path).expanduser().resolve());self._display_workspace=None
        if self.stage.currentData()=='source':
            scope=self.input_folders.get('source')
            if scope and not Path(self.workspace_path).is_relative_to(Path(scope)):
                self.input_folders.pop('source',None)
            self.input_selections['source']={'path':self.workspace_path};self.inputs_stage=None
        self.refresh();self.save_backend_settings()

    def update_ready(self):
        """Snapshot widgets cheaply; file/asset validation never runs in the input signal."""
        if self._applying_context:return
        self._ready_generation += 1
        self._ready_timer.stop()
        self._ready_request = None
        self.run.setEnabled(False)
        self.merge_button.setEnabled(not self.busy)
        self.data_select.setEnabled(not self.busy)
        self.storage_name.setEnabled(not self.busy)
        self.resume_data.setEnabled(not self.busy)
        self.resume_data_browse.setEnabled(not self.busy)
        if self.busy:
            self.status.setText('RUNNING — 완료 후 output 검증'+('' if self.stage.currentData() in PATH_STAGES else ' 및 등록')); return
        try:
            live = getattr(self.window(), 'bus', None)
            if self.stage.currentData()=='train' and self._checkpoint_future is not None:
                generation,path,_future=self._checkpoint_future
                if generation==self._checkpoint_generation and path==self.resume_path.text().strip():
                    raise ValueError('Checkpoint 설정을 읽는 중입니다.')
            if self.stage.currentData() in ('source', 'real_eval') and live and live.owner_alive() and live.status().get('backend')=='calibration':
                raise ValueError('Precision Calibration owns the motor ports; End Calibration Session first')
            if self.inputs.currentData() is None and not self.workspace_path:
                raise ValueError('입력 데이터를 선택하세요.')
            if self.stage.currentData() in ('source', 'replay', 'annotate', 'datagen', 'sim_eval', 'real_eval') and self.task_setup_error:
                raise ValueError(self.task_setup_error)
            artifact=self.inputs.currentData()
            opts = self.stage_options(require_name=False)
            self._ready_request = (self._ready_generation, self.stage.currentData(), artifact, self.workspace_path, opts)
        except Exception as exc:
            self.status.setText('NOT_READY — '+str(exc)); return
        self.status.setText('CHECKING — 실행 설정 확인 중')
        self._ready_timer.start()

    def _start_ready_check(self):
        if self.busy or self._ready_future is not None or self._ready_request is None:return
        generation,stage,artifact,workspace,options = self._ready_request
        self._ready_request = None
        self._ensure_executor()
        # The worker receives only plain data, and calls the same unchanged read-only validation.
        def check():
            context=None
            try:
                header_path=artifact if stage=='source' else workspace
                if stage in ('source','sim_eval','real_eval') and header_path:
                    w=load(header_path)
                    context={'workspace':w,'tasks':[],'scene':None}
                if artifact is None:raise ValueError('입력 데이터를 선택하세요.')
                plan=self.service.plan(stage,artifact,workspace,options,dry=True)
                valid=None
                return ('NOT_READY' if plan.get('ready') is False else 'READY'),plan.get('reason','입력 계약 확인 · Worker에서 실행 시 재검증'),valid,plan.get('context')
            except (ValueError,KeyError,OSError,TypeError) as exc:return 'NOT_READY',str(exc),[],context
        future = self._ready_executor.submit(check)
        self._ready_future = (generation,future)

    def _poll_ready_check(self):
        if self._ready_future is None:return
        generation,future = self._ready_future
        if not future.done():return
        self._ready_future = None
        try:
            result=future.result();state,reason,valid=result[:3];context=result[3] if len(result)>3 else None
        except Exception as exc:state,reason,valid,context = 'NOT_READY',str(exc),None,None
        if generation == self._ready_generation and not self.busy:
            if context is not None and context.get('workspace') is not None and self.task_selector.workspace is None:
                self._applying_context=True
                try:
                    self._path_details=context;w=context['workspace'];self._display_workspace=w
                    previous_workspace=self.workspace_path
                    self.workspace_path=str(w['root']);self.scene_cameras=tuple(w['profile']['cameras'])
                    self.configure_stage()
                    ids=(context.get('scene') or {}).get('active_object_ids')
                    if ids is None:
                        from ..real2sim.workspaces.tasks import workspace_key
                        ids=self.object_selections.get(json.dumps(workspace_key(w),sort_keys=True))
                    self.task_selector.configure(w,self.stage.currentData(),[] if self.stage.currentData()=='source' else context['tasks'],ids)
                    self.task_selector.setEnabled(True)
                    if previous_workspace==self.workspace_path:self.workspaceChanged.emit()
                    self.save_backend_settings()
                finally:self._applying_context=False
            if state=='READY' and not self.storage_name.text().strip():
                reason='저장 이름을 입력하세요.'
            self.status.setText(state+(' — '+reason if reason else ''))
            self.run.setEnabled(state=='READY')
        # A newer edit replaces the pending snapshot; never queue every intermediate keystroke.
        if self._ready_request is not None and not self._ready_timer.isActive():self._ready_timer.start(0)

    def closeEvent(self,event):
        self._input_generation+=1;self._input_request=None;self._input_future=None
        self._discovery_generation+=1;self._discovery_request=None;self._discovery_future=None
        self._checkpoint_generation+=1;self._checkpoint_future=None
        self._ready_generation += 1
        self._ready_request = None
        self._ready_timer.stop()
        if self._ready_executor is not None:
            self._ready_executor.shutdown(wait=False,cancel_futures=True)
            self._ready_executor = None
        self._ready_future = None
        super().closeEvent(event)

    def select_merge_datasets(self):
        dialog=DatasetMergeDialog(self.root,self)
        if dialog.exec()==QDialog.DialogCode.Accepted:
            paths=dialog.selected_paths()
            self.execute(stage='merge',artifact=paths[0],options={'merge_datasets':paths,'storage_name':dialog.name.text().strip()})

    def execute(self, *, stage=None, artifact=None, options=None):
        from ..real2sim.storage import atomic
        if self.busy:
            raise ValueError('현재 Workflow가 완료된 후 실행하세요.')
        stage = stage or self.stage.currentData()
        name=(options or {}).get('storage_name') if options is not None else self.storage_name.text().strip()
        if not name:
            self.storage_name.setFocus()
            raise ValueError('저장 이름을 입력하세요.')
        if stage in ('source', 'replay', 'annotate', 'datagen', 'sim_eval', 'real_eval') and self.task_setup_error:
            raise ValueError(self.task_setup_error)
        artifact=artifact or self.inputs.currentData()
        live = getattr(self.window(), 'bus', None)
        if stage in ('source', 'replay', 'annotate', 'datagen', 'sim_eval', 'real_eval') and live and live.owner_alive():
            raise ValueError('Real/Sim을 Disconnect한 뒤 Workflow를 실행하세요. 장비·카메라 중복 연결을 방지합니다.')
        opts = dict(options) if options is not None else self.stage_options()
        from .paths import validate_storage_name
        opts['storage_name']=validate_storage_name(name)
        opts.pop('output',None)
        opts['python'] = self.worker_python; opts['mode'] = self.mode.currentData()
        workspace = self.workspace_path
        self.save_backend_settings()
        self.service.session = None
        from .view import LogTail
        self.log_tail = LogTail(); self.result.clear(); self.last_result = None
        if self.job_status_view is not None:self.job_status_view.set_status({})
        self.section.setEnabled(False); self.stage.setEnabled(False)
        self.backend.setEnabled(False)
        self.navigation.setEnabled(False)
        self.active_stage = stage
        self.active_manual = stage == 'annotate' and not opts.get('auto', True)
        self.busy = True; self.task_selector.setEnabled(False); self.update_ready()
        def operation(notify):
            try:
                return {'ok': True, 'artifact': self.service.run(stage, artifact, workspace, opts, notify)}
            except Exception as exc:
                return {'ok': False, 'error': str(exc)}
        def done(result):
            self.busy = False; self.task_selector.setEnabled(True); self.last_result = result
            self.poll_log()
            self.section.setEnabled(True); self.stage.setEnabled(True)
            self.backend.setEnabled(True)
            self.navigation.setEnabled(True)
            if result['ok'] and result['artifact'].get('no_artifact'):
                self.result.appendPlainText('중지됨 · 완료된 성공 데이터 0개')
                self.refresh()
                return
            self.result.appendPlainText((('중지됨 · 완료 Episode 보존: ' if result['artifact'].get('status')=='STOPPED' else '완료: ') + result['artifact']['path']) if result['ok'] else ('오류: ' + result['error']))
            if result['ok']:
                a = result['artifact']
                if a.get('type') == 'workspace': self.workspace_path = a['path']
            self.refresh()
            if result['ok']:
                desired=artifact
                index = self.inputs.findData(desired)
                if index >= 0: self.inputs.setCurrentIndex(index)
        try: self.run_worker(operation, done)
        except Exception:
            self.busy = False; self.task_selector.setEnabled(True); self.section.setEnabled(True); self.stage.setEnabled(True); self.backend.setEnabled(True); self.navigation.setEnabled(True); self.update_ready(); raise

    def control(self, command):
        from .presentation import send
        if self.busy and self.service.session:
            send(self.service.session, command)

    def poll_preview(self):
        self._poll_episode_inputs();self._poll_model_discovery();self._poll_checkpoint_metadata()
        self._poll_ready_check()
        self.poll_log()
        root = self.service.session
        if getattr(self, 'active_stage', None) != self.stage.currentData(): root = None
        status = {}
        if root and (root / 'status.json').exists():
            try:
                status = json.loads((root / 'status.json').read_text())
            except (OSError, ValueError):
                pass
        active = self.busy and status.get('state') in ('RUNNING', 'PAUSED')
        stage = getattr(self, 'active_stage', None)
        manual = getattr(self, 'active_manual', False)
        for name, button in self.controls.items():
            from .stages import controls
            allowed = name in controls(stage, manual)
            button.setEnabled(bool(active and allowed))
        for name in self.visible_feeds:
            label=self.feeds[name]
            stamp = status.get('frames', {}).get(name, 0)
            fresh = active and time.time() - stamp < 2
            # A paused image is explicitly a frozen preview, not a live camera frame.
            paused = stamp > 0 and name.startswith('sim_') and root is not None
            if root and (fresh or paused):
                frame=(str(root),stamp)
                if self._preview_frames.get(name)==frame:continue
                pix = QPixmap(str(root / (name + '.jpg')))
                if not pix.isNull():
                    label.setPixmap(pix);self._preview_frames[name]=frame
                    continue
            self._preview_frames.pop(name,None)
            text=name.upper() + (' — 새 프레임 대기' if active and stamp else ' — 연결된 영상 없음')
            if label.text()!=text:label.setText(text)
        if self.job_status_view is not None:self.job_status_view.set_status(status,self.last_result if root else None)
        if self.task_state.isVisible():
            signals = status.get('task_signals', {})
            word = lambda value: '충족' if value is True else '미충족' if value is False else '판정 없음'
            task_text=('접근: ' + ('TCP 거리 %.1f mm (성공 임계값 없음)' % signals['tcp_distance_mm'] if signals.get('tcp_distance_mm') is not None else '측정 없음')
                + '   |   Grasp (휴리스틱): ' + word(signals.get('grasp')) + '   |   작업 성공 조건: ' + word(signals.get('place'))
                + '\nEpisode 결과: ' + status.get('episode_outcome', '진행 중' if active else '대기')
                + '   ' + status.get('outcome_reason', '')
                + (f" · 시도 {status['attempts']} / 성공 {status['successes']} / 실패 {status['failures']} · 성공률 " + (f"{status['success_rate'] * 100:.1f}%" if status.get('success_rate') is not None else "—") if 'attempts' in status else '')
                + ('\n' + ' · '.join(f"{dict(ik_failure='IK 계산 탈락',planning_failure='계획 실패',simulation_failure='시뮬레이션 실패',grasp_failure='집기 실패',place_failure='배치 실패',timeout='시간 초과').get(key,key)}: {value}" for key,value in status['failure_breakdown'].items()) if status.get('failure_breakdown') else '')
                + ('\n' + ' | '.join(f"Env {key}: {value['state']} · 성공 {value['successes']}" for key,value in status['env_status'].items()) if status.get('env_status') else ''))
            outcome = status.get('episode_outcome', '')
            color = '#dff3e3' if outcome.startswith('성공') else '#ffe4d9' if outcome.startswith('실패') else '#e9eef5'
            self.task_state.setStyleSheet(f'QLabel {{ background: {color}; color: #17202a; padding: 8px; }}')
            if (stage=='datagen' and 'target_successes' in status) or (stage=='sim_eval' and status.get('workflow_phase')=='evaluation'):
                from .presentation import datagen_status_text
                task_text=datagen_status_text(status)
            self.task_state.setText(task_text)
            # The media splitter must not compress a multi-env summary to one line.
            self.task_state.setMinimumHeight(max(self.task_state.heightForWidth(self.task_state.width()),
                self.task_state.fontMetrics().lineSpacing()*(task_text.count('\n')+1)+20))
        if self.busy and not self.preview_status.isHidden():
            text = status.get('state', 'STARTING — 환경/정책 준비 중')
            if stage=='datagen' and status.get('workflow_phase'):
                phase={'preparing':'환경 준비 중','source_validation':'Source 사전 검증 중','generation':'데이터 생성 중'}[status['workflow_phase']]
                self.status.setText(f'{text} · {phase}')
            if status.get('frames') and time.time() - max(status['frames'].values()) >= 2:
                text += ' · 마지막 프레임 (새 프레임 대기)'
            self.preview_status.setText(f"{text} · {status.get('task_id', '')} · step {status.get('steps', 0)} · "
                                       f"episode {status.get('episode_index', '-')}/{status.get('episode_total', '-')} · action {status.get('action_index', '-')} · marks {status.get('marks', [])}"
                                       + '\n' + ' · '.join(str(v) for k, v in status.items() if k.endswith('_camera_error') or k == 'leader_limit_status'))
        elif root and not self.preview_status.isHidden():
            self.preview_status.setText('실행 종료 · 마지막 SIM 프레임 표시 · 진행 로그에서 결과를 확인하세요.')


    def sync_tasks(self):
        self._path_details=None;self.task_setup_error=None;self.task_selector.workspace=None
        self.task_selector.setEnabled(False);self.update_ready()

    def objects_changed(self):
        selector = self.task_selector
        if not selector.loading and selector.workspace is not None and hasattr(self,'object_selections') and selector.stage in ('source','sim_eval'):
            key = json.dumps(selector.workspace_identity,sort_keys=True)
            self.object_selections[key] = [selector.objects.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(selector.objects.count()) if selector.objects.item(i).checkState()==Qt.CheckState.Checked]
            self.save_backend_settings()
        self.update_ready()

    def stage_options(self, *, require_name=True):
        opts = json.loads(self.options.toPlainText())
        if not isinstance(opts, dict):
            raise ValueError('실행 설정은 {"항목": 값} 형식의 JSON 객체여야 합니다.')
        stage = self.stage.currentData()
        opts.pop('storage_name',None)
        opts.pop('output',None)
        if stage == 'real_eval':
            # Readiness only validates this request. Hardware is opened exclusively
            # by the worker launched through the explicit Real evaluation start.
            opts['enable_motion'] = True
        if stage=='datagen':
            from ..simulation.ik import validate_solver,validate_failure_policy
            opts['ik_solver']=validate_solver(opts.get('ik_solver','scipy_ls'))
            opts['ik_failure_policy']=validate_failure_policy(opts.get('ik_failure_policy','strict'))
        if stage=='sim_eval' and self.checkpoint_choice.currentData():opts['checkpoint_file']=self.checkpoint_choice.currentData()
        opts.update(backend=self.backend.currentData() if hasattr(self, 'backend') else 'isaac', python=self.worker_python, mode=self.mode.currentData())
        if stage == 'source':
            opts.update(port=self.port.text(), episodes=self.episodes.value(), episode_time_s=self.episode_time.value(), reset_time_s=self.reset_time.value())
        if stage in ('source', 'datagen') and self.resume_data.text().strip():
            opts['resume_data'] = self.resume_data.text().strip()
        if stage == 'sim_eval':
            opts.update(eval_seed_mode=self.eval_seed_mode.currentData(), eval_seed=self.eval_seed.value())
        if stage == 'train':
            from .option_help import TRAIN_KEYS
            opts = {k: v for k, v in opts.items() if k in TRAIN_KEYS[self.architecture.currentData()]
                    or k in ('backend', 'python', 'mode')}
            opts['architecture'] = self.architecture.currentData()
            opts.pop('validation_dataset',None)
            if self.validation_input.currentData():opts['validation_dataset']=self.validation_input.currentData()
            opts.pop('resume_checkpoint', None)
            if self.resume_training.isChecked():
                if not self.resume_path.text().strip():
                    raise ValueError('이어서 학습할 checkpoint를 선택하세요.')
                opts['resume_checkpoint'] = self.resume_path.text().strip()
        if stage=='validate':opts['validation_dataset']=self.validation_input.currentData()
        if stage == 'annotate':
            opts['auto'] = self.annotation_mode.currentIndex() == 0
        if stage in ('source','sim_eval','replay','annotate','datagen') and self.task_selector.workspace is not None:
            opts['active_object_ids'] = self.task_selector.object_selection()
        if stage in ('source', 'sim_eval', 'real_eval') and self.task_selector.workspace is not None:
            opts['tasks'] = self.task_selector.selection(evaluation=stage != 'source')
        name=self.storage_name.text().strip()
        if not name and require_name:raise ValueError('저장 이름을 입력하세요.')
        if name:
            from .paths import validate_storage_name
            opts['storage_name']=validate_storage_name(name)
            if stage in ('train','validate') and self.learning_output.text().strip():opts['output']=str(Path(self.learning_output.text().strip()).expanduser()/name)
        return opts

    def select_resume_model(self):
        if not self.resume_models.currentData():return
        self.browse_checkpoint(self.resume_models.currentData())

    def browse_resume_data(self):
        from .curation_ui import DatasetPicker
        from .data_resume import inspect_resume_input
        stage = self.stage.currentData()
        if stage not in ('source', 'datagen') or self.busy:
            return
        directory = self.root / ('sources' if stage == 'source' else 'datagen')
        if self.resume_data.text().strip():
            current = Path(self.resume_data.text().strip()).expanduser()
            if current.is_dir():directory = current
            elif current.is_file():directory = current.parent
        picker = DatasetPicker(self, str(directory), allow_evaluation=False)
        picker.setWindowTitle('이어 생성할 이전 DataGen 결과 · dataset.hdf5' if stage=='datagen' else '이어 수집할 완료 데이터 · HDF5 파일 또는 데이터셋 폴더')
        if picker.exec() == QDialog.DialogCode.Accepted:
            path = picker.selectedFiles()[0]
            self.resume_data.setText(str(Path(path).resolve()))

    def browse_checkpoint(self,path=None):
        if path:pass
        else:
            path = QFileDialog.getExistingDirectory(self, '체크포인트 폴더 선택 (예: checkpoints/040000)')
        if not path:
            return
        self.resume_path.setText(path)
        from .policy import checkpoint_metadata
        if self._checkpoint_future is not None:self._checkpoint_future[2].cancel()
        self._checkpoint_generation+=1;self._ensure_executor()
        self._checkpoint_future=(self._checkpoint_generation,path,self._ready_executor.submit(checkpoint_metadata,path))
        self.resume_training.setChecked(True)
        self.update_ready()

    def reset_options(self, checked=False, restore_saved=False):
        from .option_help import template, TRAIN_KEYS
        self.options.set_stage(self.stage.currentData())
        stage = self.stage.currentData()
        model = self.architecture.currentData() if stage == 'train' else None
        self.options_context = (stage, model)
        self.options.allowed_keys = TRAIN_KEYS[model]-{'validation_dataset'} if model else None
        values = dict(self.saved_stage_options.get(self.options_context, template(stage))) if restore_saved else template(stage)
        if stage == 'real_eval':
            defaults = self.real_evaluation_defaults()
            if not restore_saved or self.options_context not in self.saved_stage_options:
                values.update(defaults)
            else:
                for key, value in defaults.items():
                    values.setdefault(key, value)
        if model:
            values = {k: v for k, v in values.items() if k in TRAIN_KEYS[model]}
        self.options.setPlainText(json.dumps(values, indent=2, ensure_ascii=False))

    def real_evaluation_defaults(self):
        """Reuse saved/live connection settings without opening any devices."""
        window = self.window()
        path = Path(getattr(window, 'workspace', ROOT / 'outputs/real2sim')) / 'ui_settings.json'
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError):
            saved = {}
        fields = getattr(window, 'fields', {})
        defaults = {}
        for option, field in (('port', 'follower_port'), ('follower_id', 'follower_id'),
                              ('calibration_dir', 'follower_calibration_dir'),
                              ('side_device', 'side_camera'), ('wrist_device', 'wrist_camera')):
            value = fields[field].text().strip() if field in fields else saved.get(field, '')
            if value:
                defaults[option] = value
        return defaults

    def change_model(self):
        self._checkpoint_generation+=1
        if not hasattr(self, 'options') or self.stage.currentData() != 'train':
            return
        if self.options_context:
            self.saved_stage_options[self.options_context] = json.loads(self.options.toPlainText())
        self.reset_options(restore_saved=True)
        if hasattr(self,'validation_input'):
            visible=True
            self.validation_input.setVisible(visible);self.primary.layout().labelForField(self.validation_input).setVisible(visible)
            self.refresh()
        self.update_ready()

    def browse_validation(self):
        path=QFileDialog.getExistingDirectory(self,'Validation LeRobot Dataset',str(self.root/'datasets'))
        if not path:return
        if self.validation_input.findData(path)<0:self.validation_input.addItem(Path(path).name,path)
        self.validation_input.setCurrentIndex(self.validation_input.findData(path))
        self.remember_inputs();self.update_ready()

    def change_section(self):
        groups = [('source', 'replay', 'annotate', 'datagen', 'convert', 'train', 'validate', 'sim_eval', 'hf_dataset', 'hf_policy'),
                  ('real_eval',)]
        self.stage.blockSignals(True); self.stage.clear()
        for key in groups[self.section.currentIndex()]: self.stage.addItem(LABELS[key], key)
        self.stage.blockSignals(False); self.changed()

    def poll_log(self):
        if not self.service.session: return
        path = self.service.session.parent / 'process.log'
        for line in self.log_tail.read(path):
            if self.raw_log.isChecked() or self.log_tail.relevant(line):
                self.result.appendPlainText(line if self.raw_log.isChecked() else self.log_tail.summary(line))

    def open_log(self):
        if self.service.session:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.service.session.parent / 'process.log')))


    def dataset_description(artifact):
        workspace=artifact.get('workspace') or {}
        return str(artifact['path'])+f" · {workspace.get('workspace_id','?')}/v{workspace.get('version','?')}"

    def refresh_training_choices(self):
        if not hasattr(self,'checkpoint_choice'):return
        from .paths import input_format
        path=self.inputs.currentData();stage=self.stage.currentData()
        mini=bool(stage=='sim_eval' and path and input_format(path) in ('minivla-policy','minivla-checkpoint'))
        self.checkpoint_choice.setVisible(mini);self.primary.layout().labelForField(self.checkpoint_choice).setVisible(mini)
        old=self.checkpoint_choice.currentData();self.checkpoint_choice.blockSignals(True);self.checkpoint_choice.clear()
        if mini:
            from .policy import minivla_root
            root=minivla_root(path)
            for file in [root/'best.pt',root/'last.pt',*sorted((root/'checkpoints').glob('step_*.pt')),*sorted(root.glob('step_*.pt'))]:
                if file.is_file():self.checkpoint_choice.addItem(str(file.relative_to(root)),str(file))
            self.checkpoint_choice.setCurrentIndex(max(0,self.checkpoint_choice.findData(old or path)))
        self.checkpoint_choice.blockSignals(False)

    def training_metrics_path(self):
        """Resolve the current run's plan, including custom outputs and pre-metric startup."""
        if self.stage.currentData() not in ('train','sim_eval','real_eval','hf_policy'):return None
        session=self.service.session
        if self.busy and getattr(self,'active_stage',None)=='train' and not session:return None
        identifier=self.inputs.currentData()
        key=(identifier,str(session),self.busy)
        cached=getattr(self,'_training_metrics_cache',None)
        if cached and cached[0]==key:return cached[1]
        def remember(path):
            self._training_metrics_cache=(key,path)
            return path
        def from_run(run):
            try:
                state=json.loads((run/'state.json').read_text());plan=state.get('plan',{})
                if state.get('stage')!='train':return None
                if not (self.busy and getattr(self,'active_stage',None)=='train') and state.get('source')!=identifier:return None
                options=plan.get('options',{})
                if options.get('architecture') not in (None,'minivla'):return None
                return Path(options.get('resume_output') or Path(plan['output'])/'policy')/'metrics.jsonl'
            except (OSError,ValueError,KeyError,TypeError):return None
        if session and self.busy and getattr(self,'active_stage',None)=='train':
            path=from_run(Path(session).parent)
            return remember(path) if path else None
        if identifier:
            from .policy import minivla_root
            root=minivla_root(identifier)
            if (root/'metrics.jsonl').is_file():return remember(root/'metrics.jsonl')
        if session:
            path=from_run(Path(session).parent)
            if path:return remember(path)
        if identifier:
            for run in sorted((self.root/'runs').glob('*_train_*'),reverse=True):
                path=from_run(run)
                if path and path.is_file():return remember(path)
        return None

    def show_training_loss(self):
        self._training_metrics_cache=None
        existing=getattr(self,'_loss_dialog',None)
        if existing is not None:
            existing.show();existing.raise_();existing.activateWindow();return
        dialog=QDialog(self);self._loss_dialog=dialog
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle('학습 Loss · 자동 연결 / 실시간 갱신');dialog.resize(1000,600)
        layout=QVBoxLayout(dialog)
        chart=TrainingLossPlot(None,resolver=self.training_metrics_path);chart.setObjectName('trainingLossPlot');layout.addWidget(chart)
        source=QLabel();source.setWordWrap(True);layout.addWidget(source)
        models=QComboBox();models.setObjectName('lossModelChoice');models.addItem('현재 입력 / 실행에 자동 연결',None)
        for item in discover_inputs(self.root/'models',category='policy'):
            root=Path(item['path']).parent if Path(item['path']).suffix=='.pt' else Path(item['path'])
            if (root/'metrics.jsonl').is_file() and models.findData(str(root))<0:models.addItem(item['name'],str(root))
        layout.addWidget(models)
        def select_model():
            identifier=models.currentData()
            chart.resolver=None if identifier else self.training_metrics_path
            chart.path=Path(identifier)/'metrics.jsonl' if identifier else None
            refresh()
        models.activated.connect(select_model)
        manual=QPushButton('다른 학습 기록 열기 (선택 사항)');layout.addWidget(manual)
        def open_record():
            path=QFileDialog.getOpenFileName(dialog,'metrics.jsonl 파일 선택','','JSONL (*.jsonl)')[0]
            if path:chart.resolver=None;chart.path=Path(path);refresh()
        manual.clicked.connect(open_record)
        def refresh():
            chart.reload()
            source.setText('기록: '+str(chart.path) if chart.path else '학습을 시작하면 자동 연결됩니다. 파일을 고를 필요가 없습니다.')
        def closed():
            if getattr(self,'_loss_dialog',None) is dialog:self._loss_dialog=None
        dialog.destroyed.connect(closed)
        timer=QTimer(dialog);timer.timeout.connect(refresh);timer.start(1500)
        refresh();dialog.show()

    def show_result(self):
        dialog = QDialog(self); dialog.setWindowTitle('실행 결과 상세'); dialog.resize(900, 650)
        layout = QVBoxLayout(dialog); text = QPlainTextEdit(); text.setReadOnly(True)
        text.setPlainText(json.dumps(self.last_result, indent=2, ensure_ascii=False)); layout.addWidget(text)
        dialog.exec()

    def configure_stage(self):
        from .view import CameraView
        from .stages import feeds, controls, SIM_STAGES, PREVIEW_STAGES, INSPECTION_STAGES
        if not hasattr(self, 'feeds'): return
        stage = self.stage.currentData()
        self.merge_button.setVisible(stage == 'convert')
        if stage in ('source','datagen'):
            generation=stage=='datagen'
            self.resume_data.setPlaceholderText('새로 생성하면 비워 두세요 · 이어 생성은 이전 DataGen dataset.hdf5' if generation else '선택 사항: 기존 Source HDF5 파일 또는 데이터셋 폴더')
            self.primary.layout().labelForField(self.resume_data).setText('기존 DataGen 결과 (선택)' if generation else '기존 Source 데이터 (선택)')
            self.resume_data_browse.setText('이어 생성할 DataGen 결과 선택' if generation else '이어 수집할 데이터 선택 (HDF5 / 폴더)')
        self.training_loss_button.setVisible(stage == 'train')
        if stage != 'train' and getattr(self, '_loss_dialog', None) is not None:
            self._loss_dialog.close()
            self._loss_dialog = None
        self.input_label.setText('Train Dataset' if stage=='train' else '사용할 데이터 / 모델')
        visible = feeds(stage)
        cameras = getattr(self, 'scene_cameras', ('side','wrist'))
        if stage == 'real_eval': visible = tuple('real_'+n for n in cameras)
        if stage in PREVIEW_STAGES:
            visible = tuple('sim_'+n for n in cameras)
            if stage in INSPECTION_STAGES: visible = ('sim_overview', *visible)
        for name in visible:
            if name not in self.feeds:
                self.feeds[name] = CameraView(name.upper() + ' — 영상 없음')
                self.feed_titles[name] = QLabel(name.upper().replace('_',' '))
                self.feed_titles[name].setMaximumHeight(24)

        self.visible_feeds=visible
        self.images.setVisible(bool(visible)); self.preview_status.setVisible(bool(visible))
        self.job_status_view={'annotate':self.annotation_progress}.get(stage)
        self.annotation_progress.setVisible(self.job_status_view is not None)
        self.annotation_progress.set_status({})
        for name, label in self.feeds.items():
            show = name in visible
            label.setVisible(show); self.feed_titles[name].setVisible(show)
            self.image_grid.removeWidget(label); self.image_grid.removeWidget(self.feed_titles[name])
            if not self.busy: label.setText(name.upper() + ' — 실행 대기')
        for i, name in enumerate(visible):
            if len(visible) == 3:
                row, col, span = (0, 0, 1) if i == 0 else (2 * (i - 1), 1, 1)
            else: row, col, span = 0, i, 1
            self.image_grid.addWidget(self.feed_titles[name], row, col, 1, span)
            self.image_grid.addWidget(self.feeds[name], row + 1, col, 3 if len(visible) == 3 and i == 0 else 1, span)
        self.image_grid.setColumnStretch(0, 2 if len(visible) == 3 else 1)
        self.image_grid.setColumnStretch(1, 1)
        self.image_grid.setRowStretch(1, 1)
        self.image_grid.setRowStretch(3, 1 if len(visible) == 3 else 0)
        allowed = controls(stage, self.annotation_mode.currentIndex() == 1)
        for name, button in self.controls.items(): button.setVisible(name in allowed)
        self.task_state.setVisible(stage in INSPECTION_STAGES)
        self.task_state.setMinimumHeight(0)
        self.task_state.setText('접근 / Grasp / 작업 성공 조건: 실행 대기')
        self.task_selector.setVisible(stage in ('source', 'sim_eval', 'real_eval'))
        self.mode.setVisible(stage in SIM_STAGES); self.main_form.labelForField(self.mode).setVisible(stage in SIM_STAGES)
        descriptions = {'source': 'Leader로 조작 · 저장/다음 또는 리셋 · Episode/Reset Time은 초 단위',
                        'replay': '기록된 episode 재현 · 카메라/관절 오차 확인',
                        'annotate': 'Source 처리 진행률 · 저장 성공/실패 · 환경별 상태 · 자동/수동 Annotation',
                        'datagen': '생성 동작 preview · grasp/place 신호 및 실행 로그',
                        'convert': 'HDF5 → LeRobot · episode 변환 진행과 결과 경로',
                        'train': '학습 step / loss / checkpoint는 아래 실시간 로그에서 확인',
                        'sim_eval': 'Task별 평가 · Overview / Side / Wrist · episode 성공/실패',
                        'real_eval': '실제 카메라 및 실행 로그 · 실제 성공은 자동 판정하지 않음'}
        self.stage_summary.setText(descriptions.get(stage, LABELS.get(stage, '')))
        actions = {'source': '데모 기록 시작', 'replay': '데모 재생', 'annotate': 'Annotation 시작',
                   'datagen': '데이터 생성 시작', 'convert': '데이터셋 변환', 'train': '학습 시작',
                   'sim_eval': 'Sim 평가 시작', 'real_eval': 'Real 평가 시작'}
        self.run.setText(actions.get(stage, '선택 작업 실행'))
        self.update_ready()


class DatasetMergeDialog(QDialog):
    """Path selection with background native compatibility checks."""
    def __init__(self,root,parent=None):
        super().__init__(parent);self.root=Path(root);self.extra=[];self.generation=0;self.future=None
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='merge-input')
        self.destroyed.connect(lambda:self.executor.shutdown(wait=False,cancel_futures=True))
        self.setWindowTitle('LeRobot 데이터셋 병합');self.resize(760,450)
        layout=QVBoxLayout(self)
        text=QLabel('병합할 변환본을 2개 이상 체크하세요. 미분할끼리, Train끼리, Val끼리 병합합니다.\n원본과 각 Task·분할 상태를 보존하고 새 데이터셋으로 저장합니다. 목록 순서대로 이어 붙입니다.')
        text.setWordWrap(True);layout.addWidget(text)
        self.datasets=QListWidget();layout.addWidget(self.datasets,1)
        add=QPushButton('데이터셋 폴더 추가');layout.addWidget(add);add.clicked.connect(self.add_folder)
        self.name=QLineEdit();form=QFormLayout();form.addRow('병합 데이터셋 이름',self.name);layout.addLayout(form)
        self.summary=QLabel();self.summary.setWordWrap(True);layout.addWidget(self.summary)
        self.location=QLabel();self.location.setWordWrap(True);layout.addWidget(self.location)
        buttons=QHBoxLayout();layout.addLayout(buttons)
        cancel=QPushButton('취소');cancel.clicked.connect(self.reject);buttons.addWidget(cancel)
        self.start=QPushButton('새 데이터셋으로 병합');self.start.setObjectName('primaryAction');self.start.clicked.connect(self.accept);buttons.addWidget(self.start)
        self.datasets.itemChanged.connect(self.update_selection);self.name.textChanged.connect(self.update_selection)
        self.timer=QTimer(self);self.timer.timeout.connect(self.poll);self.timer.start(30)
        self.refresh()

    def refresh(self):
        selected=set(self.selected_paths());self.datasets.blockSignals(True);self.datasets.clear()
        items=discover_inputs(self.root/'datasets',category='dataset')
        paths=list(dict.fromkeys([v['path'] for v in items if v['format']=='lerobot']+self.extra))
        for path in paths:
            item=QListWidgetItem(path);item.setData(Qt.ItemDataRole.UserRole,path)
            item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if path in selected else Qt.CheckState.Unchecked)
            self.datasets.addItem(item)
        self.datasets.blockSignals(False);self.update_selection()

    def selected_paths(self):
        return [self.datasets.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.datasets.count()) if self.datasets.item(i).checkState()==Qt.CheckState.Checked]

    def update_selection(self):
        from .paths import output_location,validate_storage_name
        self.generation+=1;generation=self.generation;self.start.setEnabled(False);self.location.clear()
        try:
            paths=self.selected_paths();name=validate_storage_name(self.name.text().strip())
            output,_=output_location('merge',paths[0] if paths else '',{'storage_name':name},self.root)
            self.location.setText('저장 위치: '+str(output))
            if output.exists():raise ValueError('이미 사용 중인 이름입니다.')
            def check():
                from .worker_request import stage_input
                from .lerobot_merge import validate_inputs
                return validate_inputs([stage_input('merge',path) for path in paths])
            self.future=(generation,self.executor.submit(check));self.summary.setText('입력 계약 확인 중…')
        except (ValueError,KeyError,OSError) as exc:self.summary.setText(str(exc))

    def poll(self):
        if not self.future or not self.future[1].done():return
        generation,future=self.future;self.future=None
        if generation!=self.generation:return
        try:
            summary=future.result();self.summary.setText(f"총 {summary['episodes']} episodes / {summary['frames']} frames");self.start.setEnabled(True)
        except Exception as exc:self.summary.setText(str(exc))

    def add_folder(self):
        folder=QFileDialog.getExistingDirectory(self,'변환된 LeRobot 데이터셋 폴더',str(self.root/'datasets'))
        if folder:self.extra.append(str(Path(folder).resolve()));self.refresh()


class TrainingLossPlot(QWidget):
    """Incremental real measurements; attach before the trainer creates its first record."""
    def __init__(self,path=None,parent=None,*,resolver=None):
        super().__init__(parent)
        from .view import LogTail
        self.path=Path(path) if path else None;self.resolver=resolver;self.tail=LogTail()
        self.rows=[];self.target=None;self.setMinimumSize(600,350)
    def reload(self):
        import math
        if self.resolver:
            path=self.resolver();self.path=Path(path) if path else None
        if self.path!=self.tail.path:
            from .view import LogTail
            self.tail=LogTail();self.rows=[];self.target=None
        if self.path and self.path.is_file():
            try:
                if self.path.stat().st_size<self.tail.offset:
                    from .view import LogTail
                    self.tail=LogTail();self.rows=[]
                size=self.path.stat().st_size
                while self.tail.offset<size:
                    for line in self.tail.read(self.path):
                        try:
                            row=json.loads(line);loss=row.get('training_loss',row.get('loss'))
                            if not math.isfinite(float(loss)) or not isinstance(row.get('step'),int):continue
                            row['loss']=float(loss)
                            validation=row.get('validation_loss',row.get('validation_action_loss'))
                            row.pop('validation_action_loss',None)
                            if validation is not None and math.isfinite(float(validation)):row['validation_action_loss']=float(validation)
                            self.rows.append(row)
                        except (ValueError,TypeError,AttributeError):pass
                config=json.loads((self.path.parent/'config.json').read_text());self.target=config.get('steps')
            except (OSError,ValueError,TypeError):pass
        self.update()
    def paintEvent(self,event):
        from PyQt6.QtGui import QPainter,QColor,QPen
        from PyQt6.QtCore import QPointF
        p=QPainter(self);p.fillRect(self.rect(),QColor('#17212b'));p.setPen(QColor('#eeeeee'))
        train=[(r['step'],r['loss']) for r in self.rows]
        val=[(r['step'],r['validation_action_loss']) for r in self.rows if 'validation_action_loss' in r]
        best=min(val,key=lambda v:v[1]) if val else None
        p.drawText(20,25,f"Step {train[-1][0] if train else 0} / {self.target or '?'} · Best Val {best if best else '미측정'}")
        p.drawText(20,48,'Train: 청록 선 · Validation: 주황 점 (episode당 최대 8개 고정 시점 평균)')
        if not train:
            p.drawText(20,90,'첫 학습 step 기록을 기다리는 중 · 1.5초마다 자동 갱신')
            return
        points=train+val;xmax=max(x for x,y in points);ymax=max(y for x,y in points);ymin=min(y for x,y in points)
        def point(x,y):return QPointF(60+(self.width()-90)*x/max(xmax,1),self.height()-40-(self.height()-120)*(y-ymin)/max(ymax-ymin,1e-9))
        p.drawText(8,85,f'{ymax:.3g}');p.drawText(8,self.height()-40,f'{ymin:.3g}')
        p.drawText(60,self.height()-12,'0');p.drawText(self.width()-65,self.height()-12,str(xmax))
        p.setPen(QPen(QColor('#4ed6ba'),1))
        for a,b in zip(train,train[1:]):p.drawLine(point(*a),point(*b))
        if len(train)==1:p.drawPoint(point(*train[0]))
        p.setPen(QPen(QColor('#f3b563'),5))
        for v in val:p.drawPoint(point(*v))
        p.end()
