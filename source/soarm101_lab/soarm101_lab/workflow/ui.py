"""One guided/independent workflow panel backed by Operations, not shell strings."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PyQt6.QtCore import QUrl, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QPixmap, QShortcut, QKeySequence
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
                            QPushButton, QLineEdit, QPlainTextEdit, QTextBrowser, QFileDialog, QMessageBox, QGridLayout, QScrollArea, QCheckBox, QSplitter, QToolButton, QDialog, QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem, QSizePolicy)
from .artifacts import Registry, browse_directory, artifact_description, episode_input_kind, OUTPUT_FOLDERS
from .operations import Operations, STAGES, default_python
from ..real2sim.workspaces.package import ROOT, load, metadata
from ..real2sim.storage import atomic

LABELS = {'source': 'Source Demo',
          'replay': 'Replay', 'annotate': 'Annotation', 'datagen': 'Mimic / Datagen',
          'convert': 'LeRobot 변환', 'train': '모델 학습', 'hf_dataset': 'Dataset 업로드',
          'hf_policy': 'Policy 업로드', 'sim_eval': 'Sim 평가', 'real_eval': 'Real 평가 (모터 동작)'}



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

    def __init__(self, run_worker, parent=None):
        super().__init__(parent)
        from .theme import STYLE
        self.setStyleSheet(STYLE)
        self.run_worker = run_worker; self.registry = Registry(); self.service = Operations(self.registry)
        self.busy = False
        self.task_setup_error = None
        self.input_folders = {}
        self.input_selections = {}
        self.inputs_stage = None
        self._workspace_path = ''
        self.worker_python = default_python()
        self._ready_generation = 0
        self._ready_future = None
        self._ready_request = None
        self._ready_executor = None
        self._ready_timer = QTimer(self)
        self._ready_timer.setSingleShot(True)
        self._ready_timer.setInterval(200)
        self._ready_timer.timeout.connect(self._start_ready_check)
        self.saved_stage_options = {}
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
        for k in ('source', 'replay', 'annotate', 'datagen', 'convert', 'train', 'sim_eval', 'hf_dataset', 'hf_policy'): self.stage.addItem(LABELS[k], k)
        form.addRow('단계', self.stage)
        self.stage.hide(); form.labelForField(self.stage).hide()
        self.inputs = QComboBox(); self.inputs.setMinimumContentsLength(16); self.inputs.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon); self.input_label=QLabel('사용할 데이터 / 모델');form.addRow(self.input_label, self.inputs)
        self.input_browse = QPushButton('입력 폴더 열기')
        self.input_browse.clicked.connect(lambda: self.guard(self.browse))
        self.collection_browse = QPushButton('데이터 묶음 선택')
        self.collection_browse.clicked.connect(lambda: self.guard(self.browse_collection))
        input_buttons=QHBoxLayout();input_buttons.addWidget(self.input_browse);input_buttons.addWidget(self.collection_browse)
        form.addRow(input_buttons)
        from .task_ui import TaskSelector
        self.task_selector = TaskSelector(); layout.addWidget(self.task_selector)
        self.task_selector.changed.connect(self.objects_changed)
        self.task_selector.layout().setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        advanced = QToolButton(); advanced.setText('실행 설정 접기 / 펼치기'); advanced.setCheckable(True)
        layout.addWidget(advanced)
        self.primary = QWidget(); primary_form = QFormLayout(self.primary)
        primary_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.port = QLineEdit('/dev/so101_leader'); primary_form.addRow('Leader port', self.port)
        self.episodes = QSpinBox(); self.episodes.setRange(1, 100000); primary_form.addRow('Episodes', self.episodes)
        self.episode_time = QDoubleSpinBox(); self.episode_time.setRange(.1, 3600); self.episode_time.setValue(20); self.episode_time.setSuffix(' s')
        self.reset_time = QDoubleSpinBox(); self.reset_time.setRange(0, 600); self.reset_time.setValue(5); self.reset_time.setSuffix(' s')
        primary_form.addRow('Episode Time', self.episode_time); primary_form.addRow('Reset Time', self.reset_time)
        self.annotation_mode = QComboBox(); self.annotation_mode.addItems(['자동 Annotation', '수동 Annotation'])
        primary_form.addRow('Annotation 방식', self.annotation_mode)
        self.architecture = QComboBox()
        self.architecture.addItem('SmolVLA — smolvla_base 파인튜닝', 'smolvla')
        self.architecture.addItem('ACT', 'act')
        self.architecture.addItem('MiniVLA · Qwen3-VL + Flow-DiT', 'minivla')
        primary_form.addRow('학습 모델', self.architecture)
        self.architecture.currentIndexChanged.connect(self.change_model)
        self.validation_input=QComboBox();self.validation_input.addItem('별도 검증 데이터 없음 (Train 전용 입력은 선택 필요)',None)
        self.validation_input.setToolTip('Train 전용 데이터는 별도의 Validation LeRobot 데이터를 선택하세요. 목록에 없으면 입력 폴더 열기로 Val 폴더를 등록하세요. Legacy 자동 분리는 고급 설정에 있습니다.')
        primary_form.addRow('Validation Dataset (MiniVLA)',self.validation_input)
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
        self.resume_models=QComboBox();primary_form.addRow('등록 모델에서 Resume',self.resume_models)
        self.resume_models.activated.connect(self.select_resume_model)
        self.resume_training.toggled.connect(self.update_ready)
        self.resume_path.textChanged.connect(self.update_ready)
        self.resume_data = QLineEdit()
        self.resume_data.setPlaceholderText('선택 사항: 이어 수집할 기존 HDF5')
        self.resume_data_browse = QPushButton('이어 수집할 HDF5 선택')
        primary_form.addRow('기존 완료 데이터', self.resume_data)
        primary_form.addRow('이어 수집', self.resume_data_browse)
        self.resume_data.textChanged.connect(self.update_ready)
        self.resume_data_browse.clicked.connect(self.browse_resume_data)
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
        settings = self.registry.root / 'ui.json'
        if settings.exists():
            saved = json.loads(settings.read_text()); self.workspace_path = saved.get('workspace', '')
            self.worker_python = saved.get('python', default_python())
            self.mode.setCurrentIndex(max(0, self.mode.findData(saved.get('mode', 'operator'))))
        self.stage.currentIndexChanged.connect(self.changed)
        self.inputs.currentIndexChanged.connect(self.sync_tasks)
        self.options.textChanged.connect(self.update_ready)
        # Register the selected imported workspace without starting any worker.
        try:
            if self.workspace_path:
                root = load(self.workspace_path)['root']
                if not any(a['path'] == str(root) for a in self.registry.all()):
                    self.registry.register('workspace', root)
        except Exception as exc:
            self.result.setPlainText(str(exc))
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
        saved_path = self.registry.root / 'ui.json'
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
        atomic(self.registry.root / 'ui.json', {
            'workspace': self.workspace_path, 'python': self.worker_python, 'mode': self.mode.currentData(),
            'backend': self.backend.currentData(), 'backend_pythons': self.backend_pythons,
            'object_selections': getattr(self,'object_selections',{}), 'input_folders':self.input_folders,
            'input_selections':self.input_selections})

    def guard(self, fn):
        try: fn()
        except Exception as exc: QMessageBox.warning(self, 'Workflow', str(exc))

    def open_curation(self):
        from .curation_ui import CurationDialog
        workspace = self.task_selector.workspace_identity if self.task_selector.workspace is not None else None
        dialog = CurationDialog(self.registry, self, workspace=workspace)
        selected=self.registry.get(self.inputs.currentData()) if self.inputs.currentData() else {}
        if selected.get('type') in ('source','annotated','datagen'):
            QTimer.singleShot(0,lambda:dialog.load_data(selected['path']))
        else:QTimer.singleShot(0, dialog.open_latest)
        dialog.exec()
        if getattr(dialog,'saved_collection',None) and self.stage.currentData() in ('replay','annotate','convert'):
            self.input_folders[self.stage.currentData()]=dialog.saved_collection
            self.save_backend_settings()
        self.refresh()

    def remember_inputs(self):
        stage=self.stage.currentData()
        if self.inputs_stage!=stage:return
        selected={'artifact':self.inputs.currentData()}
        if stage=='train':selected['validation']=self.validation_input.currentData()
        if self.input_selections.get(stage)!=selected:
            self.input_selections[stage]=selected
            if hasattr(self,'backend'):self.save_backend_settings()

    def update_input_tooltip(self):
        self.inputs.setToolTip('')
        identifier = self.inputs.currentData()
        if identifier:
            try:
                artifact=self.registry.get(identifier)
                self.inputs.setToolTip(artifact['path'])
                self.refresh_training_choices()
            except (OSError, ValueError):
                pass
        self.remember_inputs()

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
        if self.configured_stage and self.configured_stage != stage:
            self.saved_stage_options[self.options_context] = json.loads(self.options.toPlainText())
        self.configured_stage = stage
        self.sync_navigation()
        self.dashboard_title.setText(LABELS.get(stage, stage))
        self.configure_stage()
        self.primary.setVisible(stage in ('source', 'annotate', 'train', 'datagen', 'sim_eval'))
        primary_form = self.primary.layout()
        validation_visible=stage=='train' and self.architecture.currentData()=='minivla'
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
        stage=self.stage.currentData();saved=self.input_selections.get(stage,{})
        selected=self.inputs.currentData() if self.inputs_stage==stage else saved.get('artifact')
        validation_id=self.validation_input.currentData() if self.inputs_stage==stage else saved.get('validation')
        self.inputs_stage=stage
        self.inputs.blockSignals(True); self.inputs.clear()
        allowed = STAGES[stage][0]
        folder=self.input_folders.get(stage)
        if folder:
            path=Path(folder).resolve();label=path.name
            for bucket in set(OUTPUT_FOLDERS.values()):
                store=(self.registry.root/bucket).resolve()
                if path.is_relative_to(store):
                    label=bucket if path==store else bucket+' / '+path.name
                    break
        self.collection_browse.setText('묶음: '+label if folder else '데이터 묶음 선택')
        self.collection_browse.setToolTip(folder or str(self.registry.collection_directory(stage)))
        try: workspace = metadata(load(self.workspace_path)) if self.workspace_path else None
        except Exception: workspace = None
        if stage!='source':
            try:workspace=self.registry.get(selected).get('workspace') if selected else None
            except (OSError,ValueError):workspace=None
        self.validation_input.blockSignals(True);self.validation_input.clear();self.validation_input.addItem('별도 검증 데이터 없음 (Train 전용 입력은 선택 필요)',None)
        for a in self.registry.all():
            if a.get('status', 'SUCCEEDED') != 'SUCCEEDED' or not Path(a['path']).exists():
                continue
            if not self.in_input_folder(a):continue
            partition=a.get('provenance',{}).get('source_partition') or a.get('source_partition')
            if stage=='train' and a['type']=='dataset' and partition and partition['split']=='val':
                self.validation_input.addItem(f'Val Source {len(partition["source_ids"])} · {Path(a["path"]).name}',a['id'])
            engine_match = a.get('backend', 'isaac') == self.backend.currentData() if hasattr(self, 'backend') else True
            if self.stage.currentData() in ('replay','annotate','datagen') and (a.get('format') in ('so101.common-episode/2','so101.canonical/1') or a.get('env_args',{}).get('workspace_path')):
                engine_match=True
            if a['type'] in allowed and (a['type'] == 'workspace' or engine_match or self.stage.currentData() in ('train', 'sim_eval', 'hf_dataset', 'hf_policy')) and (folder or a['type'] == 'workspace' or not workspace or not a.get('workspace') or a['workspace'] == workspace):
                label = self.input_description(a)
                if self.stage.currentData()=='train' and self.architecture.currentData()=='minivla':
                    if partition and partition.get('split')=='val':continue
                    label=self.input_description(a)
                self.inputs.addItem(label, a['id'])
                self.inputs.setItemData(self.inputs.count() - 1, label+'\nID: '+a['id']+'\n'+a['path'], Qt.ItemDataRole.ToolTipRole)
        index = self.inputs.findData(selected)
        if self.stage.currentData() == 'source' and index<0:
            index = next((i for i in range(self.inputs.count())
                          if self.registry.get(self.inputs.itemData(i))['path'] == self.workspace_path), index)
        self.inputs.setCurrentIndex(index if index >= 0 else self.inputs.count() - 1)
        vi=self.validation_input.findData(validation_id);self.validation_input.setCurrentIndex(max(0,vi));self.validation_input.blockSignals(False)
        self.inputs.blockSignals(False); self.refresh_training_choices(); self.update_input_tooltip();self.sync_tasks(); self.update_ready()
        old=self.resume_models.currentData();self.resume_models.blockSignals(True);self.resume_models.clear();self.resume_models.addItem('등록 모델 선택 (또는 파일 선택)',None)
        for a in self.registry.all():
            if a['type']=='policy' and a.get('status')=='SUCCEEDED' and Path(a['path']).exists():
                self.resume_models.addItem(artifact_description(a),a['id'])
        self.resume_models.setCurrentIndex(max(0,self.resume_models.findData(old)));self.resume_models.blockSignals(False)

    def in_input_folder(self, artifact):
        folder=self.input_folders.get(self.stage.currentData())
        return not folder or Path(artifact['path']).resolve().is_relative_to(Path(folder).resolve())

    def input_description(self, artifact):
        folder=self.input_folders.get(self.stage.currentData())
        if folder:
            relative=Path(artifact['path']).resolve().relative_to(Path(folder).resolve())
            partition=(artifact.get('source_partition') or artifact.get('provenance',{}).get('source_partition') or {}).get('split')
            role={'train':'Train','val':'Val'}.get(partition)
            return (role+' · ' if role else '')+str(relative)
        return self.dataset_description(artifact) if self.stage.currentData()=='train' else artifact_description(artifact)

    def browse_collection(self):
        if self.busy:raise ValueError('Workflow 실행이 끝난 뒤 데이터 묶음을 선택하세요.')
        stage=self.stage.currentData();current=self.inputs.currentData()
        artifact=self.registry.get(current) if current else {}
        directory=self.input_folders.get(stage) or self.registry.collection_directory(stage,artifact)
        folder=QFileDialog.getExistingDirectory(self,'현재 단계의 데이터 묶음 폴더 선택',str(directory))
        if folder:self.select_collection(folder)

    def select_collection(self, folder):
        if self.busy:raise ValueError('Workflow 실행이 끝난 뒤 데이터 묶음을 선택하세요.')
        folder=str(Path(folder).expanduser().resolve())
        self.registry.discover_collection(folder,STAGES[self.stage.currentData()][0])
        self.input_folders[self.stage.currentData()]=folder
        self.inputs.setCurrentIndex(-1)
        self.refresh();self.save_backend_settings()

    def select_workspace(self, path):
        """Share the chosen Workspace with Real/Sim without running a stage."""
        if self.busy:raise ValueError('Workflow 실행이 끝난 뒤 Scene을 선택하세요.')
        w=load(path)
        self.workspace_path = str(w['root'])
        if self.stage.currentData()=='source':
            artifact=next((a for a in self.registry.all() if a['type']=='workspace' and a['path']==str(w['root'])),None)
            if artifact:
                if not self.in_input_folder(artifact):self.input_folders.pop('source',None)
                self.input_selections['source']={'artifact':artifact['id']};self.inputs_stage=None
        self.refresh()
        self.save_backend_settings()

    def update_ready(self):
        """Snapshot widgets cheaply; file/asset validation never runs in the input signal."""
        self._ready_generation += 1
        self._ready_timer.stop()
        self._ready_request = None
        self.run.setEnabled(False)
        if self.busy:
            self.status.setText('RUNNING — 완료 후 output 검증 및 등록'); return
        try:
            live = getattr(self.window(), 'bus', None)
            if self.stage.currentData() in ('source', 'real_eval') and live and live.owner_alive() and live.status().get('backend')=='calibration':
                raise ValueError('Precision Calibration owns the motor ports; End Calibration Session first')
            if self.inputs.currentData() is None:
                raise ValueError('입력 데이터를 선택하세요.')
            if self.stage.currentData() in ('source', 'replay', 'annotate', 'datagen', 'sim_eval', 'real_eval') and self.task_setup_error:
                raise ValueError(self.task_setup_error)
            artifact = self.registry.get(self.inputs.currentData())
            opts = self.stage_options()
            self._ready_request = (self._ready_generation, self.stage.currentData(), artifact, self.workspace_path, opts)
        except Exception as exc:
            self.status.setText('NOT_READY — '+str(exc)); return
        self.status.setText('CHECKING — 실행 설정 확인 중')
        self._ready_timer.start()

    def _start_ready_check(self):
        if self.busy or self._ready_future is not None or self._ready_request is None:return
        generation,stage,artifact,workspace,options = self._ready_request
        self._ready_request = None
        if self._ready_executor is None:
            self._ready_executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='workflow-readiness')
            executor = self._ready_executor
            self.destroyed.connect(lambda:executor.shutdown(wait=False,cancel_futures=True))
        # The worker receives only plain data, and calls the same unchanged read-only validation.
        future = self._ready_executor.submit(self.service.readiness,stage,artifact,workspace,options)
        self._ready_future = (generation,future)

    def _poll_ready_check(self):
        if self._ready_future is None:return
        generation,future = self._ready_future
        if not future.done():return
        self._ready_future = None
        try:state,reason = future.result()
        except Exception as exc:state,reason = 'NOT_READY',str(exc)
        if generation == self._ready_generation and not self.busy:
            self.status.setText(state+(' — '+reason if reason else ''))
            self.run.setEnabled(state=='READY')
        # A newer edit replaces the pending snapshot; never queue every intermediate keystroke.
        if self._ready_request is not None and not self._ready_timer.isActive():self._ready_timer.start(0)

    def closeEvent(self,event):
        self._ready_generation += 1
        self._ready_request = None
        self._ready_timer.stop()
        if self._ready_executor is not None:
            self._ready_executor.shutdown(wait=False,cancel_futures=True)
            self._ready_executor = None
        self._ready_future = None
        super().closeEvent(event)

    def browse(self):
        if self.busy:
            raise ValueError('Workflow 실행이 끝난 뒤 입력을 선택하세요.')
        stage = self.stage.currentData()
        if stage == 'source':
            return self.browse_workspace()
        allowed = STAGES[stage][0]
        current = self.registry.get(self.inputs.currentData()) if self.inputs.currentData() else {}
        kind = current.get('type') or {'replay':'source', 'annotate':'source', 'datagen':'annotated',
            'convert':'datagen', 'train':'dataset', 'hf_dataset':'dataset',
            'sim_eval':'policy', 'real_eval':'policy', 'hf_policy':'policy'}[stage]
        directory = browse_directory(kind, current['path'], registry=self.registry) if current else str(
            self.input_folders.get(stage) or self.registry.collection_directory(stage))
        if stage in ('train', 'hf_dataset', 'sim_eval', 'real_eval', 'hf_policy'):
            path = QFileDialog.getExistingDirectory(self, LABELS[stage]+' 입력 폴더 선택', directory)
        else:
            path, _ = QFileDialog.getOpenFileName(self, LABELS[stage]+' 입력 선택', directory,
                'Episode 데이터 (*.hdf5 *.h5 manifest.json project_provenance.json)')
            if path and Path(path).name in ('manifest.json', 'project_provenance.json'):
                path = str(Path(path).parent)
        if path:
            known = next((a for a in reversed(self.registry.all()) if a['type'] in allowed
                          and a.get('status', 'SUCCEEDED') == 'SUCCEEDED'
                          and Path(a['path']).resolve() == Path(path).resolve()), None)
            if known:
                artifact = known
                kind = artifact['type']
            elif stage in ('replay', 'annotate', 'datagen', 'convert'):
                kind = episode_input_kind(path)
                if kind not in allowed:
                    raise ValueError('현재 단계의 입력 형식이 아닙니다. '+LABELS[stage]+'에 맞는 데이터를 선택하세요.')
                artifact = self.registry.register(kind, path)
            elif stage in ('sim_eval', 'real_eval', 'hf_policy'):
                kind = 'policy'
                if (Path(path) / 'project_provenance.json').is_file():
                    artifact = self.registry.register(kind, path)
                else:
                    from .policy import import_checkpoint
                    artifact = import_checkpoint(path, self.registry)
            else:
                kind = 'dataset'
                artifact = self.registry.register(kind, path)
            is_validation=(kind=='dataset' and self.stage.currentData()=='train' and self.architecture.currentData()=='minivla'
                and (artifact.get('provenance',{}).get('source_partition') or {}).get('split')=='val')
            if not self.in_input_folder(artifact):self.input_folders.pop(stage,None)
            if not is_validation:
                self.workspace_path = str(self.input_workspace(artifact)['root'])
            self.refresh()
            if is_validation:
                index=self.validation_input.findData(artifact['id'])
                if index<0:raise ValueError('선택한 Val 데이터가 현재 Train과 호환되지 않습니다. Train 입력·Workspace·원본 분할 계약을 확인하세요.')
                self.validation_input.setCurrentIndex(index)
            else:
                index=self.inputs.findData(artifact['id'])
                if index<0:raise ValueError('선택한 입력은 현재 단계·Backend와 호환되지 않습니다.')
                self.inputs.setCurrentIndex(index)

    def browse_workspace(self):
        path = QFileDialog.getExistingDirectory(self, 'Workspace 또는 Builder export 폴더 선택', browse_directory('workspace', self.workspace_path))
        if path:
            if (Path(path)/'scene_manifest.yaml').is_file():
                import subprocess
                result = subprocess.run([self.worker_python, str(ROOT/'scripts/real2sim/workflow.py'),
                    'import-builder', path], capture_output=True, text=True, timeout=180)
                if result.returncode:
                    raise RuntimeError('Builder 가져오기 실패: ' + result.stderr[-3000:])
                path = result.stdout.strip().splitlines()[-1]
            self.registry.register('workspace', path);self.select_workspace(path)


    def execute(self):
        from ..real2sim.storage import atomic
        stage = self.stage.currentData()
        if stage in ('source', 'replay', 'annotate', 'datagen', 'sim_eval', 'real_eval') and self.task_setup_error:
            raise ValueError(self.task_setup_error)
        artifact = self.registry.get(self.inputs.currentData())
        live = getattr(self.window(), 'bus', None)
        if stage in ('source', 'replay', 'annotate', 'datagen', 'sim_eval', 'real_eval') and live and live.owner_alive():
            raise ValueError('Real/Sim을 Disconnect한 뒤 Workflow를 실행하세요. 장비·카메라 중복 연결을 방지합니다.')
        opts = self.stage_options(); opts['python'] = self.worker_python; opts['mode'] = self.mode.currentData()
        workspace = self.workspace_path
        self.save_backend_settings()
        self.service.session = None
        from .view import LogTail
        self.log_tail = LogTail(); self.result.clear(); self.last_result = None
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
            self.result.appendPlainText(('완료: ' + result['artifact']['path']) if result['ok'] else ('오류: ' + result['error']))
            if result['ok']:
                a = result['artifact']
                if a['type'] == 'workspace': self.workspace_path = a['path']
            self.refresh()
            if result['ok']:
                desired = artifact['id'] if stage == 'replay' else result['artifact']['id']
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
        for name, label in self.feeds.items():
            stamp = status.get('frames', {}).get(name, 0)
            fresh = active and time.time() - stamp < 2
            # A paused image is explicitly a frozen preview, not a live camera frame.
            paused = stamp > 0 and name.startswith('sim_') and root is not None
            if root and (fresh or paused):
                pix = QPixmap(str(root / (name + '.jpg')))
                if not pix.isNull():
                    label.setPixmap(pix)
                    continue
            label.setText(name.upper() + (' — 새 프레임 대기' if active and stamp else ' — 연결된 영상 없음'))
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
            if stage=='datagen' and 'target_successes' in status:
                from .presentation import datagen_status_text
                task_text=datagen_status_text(status)
            self.task_state.setText(task_text)
            # The media splitter must not compress a multi-env summary to one line.
            self.task_state.setMinimumHeight(max(self.task_state.heightForWidth(self.task_state.width()),
                self.task_state.fontMetrics().lineSpacing()*(task_text.count('\n')+1)+20))
        if self.busy:
            text = status.get('state', 'STARTING — 환경/정책 준비 중')
            if stage=='datagen' and status.get('workflow_phase'):
                phase={'preparing':'환경 준비 중','source_validation':'Source 사전 검증 중','generation':'데이터 생성 중'}[status['workflow_phase']]
                self.status.setText(f'{text} · {phase}')
            if status.get('frames') and time.time() - max(status['frames'].values()) >= 2:
                text += ' · 마지막 프레임 (새 프레임 대기)'
            self.preview_status.setText(f"{text} · {status.get('task_id', '')} · step {status.get('steps', 0)} · "
                                       f"episode {status.get('episode_index', '-')}/{status.get('episode_total', '-')} · action {status.get('action_index', '-')} · marks {status.get('marks', [])}"
                                       + '\n' + ' · '.join(str(v) for k, v in status.items() if k.endswith('_camera_error') or k == 'leader_limit_status'))
        elif root:
            self.preview_status.setText('실행 종료 · 마지막 SIM 프레임 표시 · 진행 로그에서 결과를 확인하세요.')

    def input_workspace(self, artifact):
        """Resolve the selected input's Scene without a second editable path."""
        if artifact.get('type') == 'workspace':
            return load(artifact['path'])
        identity = artifact.get('workspace')
        candidates = [self.workspace_path]
        if identity:
            candidates.extend(a['path'] for a in self.registry.all()
                              if a['type'] == 'workspace' and a.get('workspace') == identity
                              and a.get('status', 'SUCCEEDED') == 'SUCCEEDED')
            provenance = artifact.get('provenance', {})
            candidates.extend((artifact.get('env_args', {}).get('workspace_path'),
                               provenance.get('workspace_path'),
                               provenance.get('source_env_args', {}).get('workspace_path')))
        for path in dict.fromkeys(p for p in candidates if p):
            try:
                workspace = load(path)
                if not identity or metadata(workspace) == identity:
                    return workspace
            except (ValueError, OSError):
                continue
        raise ValueError('입력에 해당하는 Scene을 찾을 수 없습니다. Source Demo의 입력 폴더 열기에서 해당 Scene을 등록하세요.')

    def sync_tasks(self):
        from .tasks import from_artifact
        self.task_setup_error = None
        try:
            artifact = self.registry.get(self.inputs.currentData()) if self.inputs.currentData() else {}
            previous = self.workspace_path
            w = self.input_workspace(artifact)
            self.workspace_path = str(w['root'])
            self.scene_cameras = tuple(w['profile']['cameras'])
            self.configure_stage()
            from .tasks import workspace_key, recorded_scene
            key = json.dumps(workspace_key(w),sort_keys=True)
            selected = getattr(self,'object_selections',{}).get(key)
            if self.stage.currentData() in ('replay','annotate','datagen'):
                scene = recorded_scene(artifact,w,required=False)
                if scene:selected = scene['active_object_ids']
            self.task_selector.configure(w, self.stage.currentData(), from_artifact(artifact, w), selected)
            if previous != self.workspace_path and hasattr(self, 'backend'):
                self.save_backend_settings()
        except (ValueError, OSError, TypeError, KeyError) as exc:
            self.task_selector.workspace = None
            self.task_setup_error = str(exc)
        self.update_ready()

    def objects_changed(self):
        selector = self.task_selector
        if not selector.loading and selector.workspace is not None and hasattr(self,'object_selections') and selector.stage in ('source','sim_eval'):
            key = json.dumps(selector.workspace_identity,sort_keys=True)
            self.object_selections[key] = [selector.objects.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(selector.objects.count()) if selector.objects.item(i).checkState()==Qt.CheckState.Checked]
            self.save_backend_settings()
        self.update_ready()

    def stage_options(self):
        opts = json.loads(self.options.toPlainText())
        if not isinstance(opts, dict):
            raise ValueError('실행 설정은 {"항목": 값} 형식의 JSON 객체여야 합니다.')
        stage = self.stage.currentData()
        if stage=='datagen':
            from ..simulation.ik import validate_solver,validate_failure_policy
            opts['ik_solver']=validate_solver(opts.get('ik_solver','scipy_ls'))
            opts['ik_failure_policy']=validate_failure_policy(opts.get('ik_failure_policy','strict'))
        if stage=='sim_eval' and self.checkpoint_choice.currentData():opts['checkpoint_file']=self.checkpoint_choice.currentData()
        opts.update(backend=self.backend.currentData() if hasattr(self, 'backend') else 'isaac', python=self.worker_python, mode=self.mode.currentData())
        if stage == 'source': opts.update(port=self.port.text(), episodes=self.episodes.value(), episode_time_s=self.episode_time.value(), reset_time_s=self.reset_time.value())
        if stage in ('source', 'datagen') and self.resume_data.text().strip():
            opts['resume_data'] = self.resume_data.text().strip()
        if stage == 'sim_eval':
            opts.update(eval_seed_mode=self.eval_seed_mode.currentData(), eval_seed=self.eval_seed.value())
        if stage == 'train':
            from .option_help import TRAIN_KEYS
            opts = {k: v for k, v in opts.items() if k in TRAIN_KEYS[self.architecture.currentData()]
                    or k in ('backend', 'python', 'mode')}
            opts['architecture'] = self.architecture.currentData()
            if opts['architecture']=='minivla' and self.validation_input.currentData():opts['validation_dataset']=self.validation_input.currentData()
            if opts['architecture'] == 'minivla' and not opts.get('model_config'):
                from ..learning.minivla.config import MODEL_ROOT
                opts['model_config'] = str(MODEL_ROOT / 'configs/default.json')
            opts.pop('resume_checkpoint', None)
            if self.resume_training.isChecked():
                if not self.resume_path.text().strip():
                    raise ValueError('이어서 학습할 checkpoint를 선택하세요.')
                opts['resume_checkpoint'] = self.resume_path.text().strip()
        if stage == 'annotate':
            opts['auto'] = self.annotation_mode.currentIndex() == 0
        if stage in ('source','sim_eval','replay','annotate','datagen') and self.task_selector.workspace is not None:
            opts['active_object_ids'] = self.task_selector.object_selection()
        if stage in ('source', 'sim_eval', 'real_eval'):
            opts['tasks'] = self.task_selector.selection(evaluation=stage != 'source')
        return opts

    def select_resume_model(self):
        if not self.resume_models.currentData():return
        record=self.registry.get(self.resume_models.currentData());root=Path(record['path'])
        mini=record.get('format')=='so101.minivla-policy/1'
        if mini:self.architecture.setCurrentIndex(self.architecture.findData('minivla'))
        else:self.architecture.setCurrentIndex(self.architecture.findData(record.get('format',{}).get('type','smolvla')))
        self.browse_checkpoint(str(root/('last.pt' if (root/'last.pt').is_file() else 'best.pt')) if mini else str(root))

    def browse_resume_data(self):
        if self.is_mujoco():
            path = QFileDialog.getExistingDirectory(self, '이어 수집할 완료 데이터셋')
        else:
            path, _ = QFileDialog.getOpenFileName(self, '이어 수집할 완료 HDF5', '', 'HDF5 (*.hdf5 *.h5)')
        if path:
            self.resume_data.setText(path)

    def browse_checkpoint(self,path=None):
        if path:pass
        elif self.architecture.currentData() == 'minivla':
            path, _ = QFileDialog.getOpenFileName(self, 'MiniVLA 재개 체크포인트 선택', '', 'PyTorch (*.pt)')
        else:
            path = QFileDialog.getExistingDirectory(self, '체크포인트 폴더 선택 (예: checkpoints/040000)')
        if not path:
            return
        self.resume_path.setText(path)
        root = Path(path).resolve()
        if root.suffix == '.pt':
            try:
                from .policy import minivla_root
                model_root=minivla_root(root)
                saved = json.loads((model_root / 'config.json').read_text())
                opts = json.loads(self.options.toPlainText())
                opts['steps'] = saved['steps']
                opts['batch_size'] = saved.get('batch_size',1)
                opts['model_config'] = str(model_root / 'config.json')
                self.options.setPlainText(json.dumps(opts, indent=2, ensure_ascii=False))
            except (OSError, ValueError, KeyError): pass
            self.resume_training.setChecked(True)
            return
        if root.name == 'pretrained_model':
            root = root.parent
        try:
            saved = json.loads((root / 'pretrained_model/train_config.json').read_text())
            self.architecture.setCurrentIndex(self.architecture.findData(saved['policy']['type']))
            opts = json.loads(self.options.toPlainText())
            opts['steps'] = saved['steps']
            self.options.setPlainText(json.dumps(opts, indent=2, ensure_ascii=False))
        except (OSError, ValueError, KeyError):
            pass  # readiness validation explains incomplete checkpoints
        self.resume_training.setChecked(True)

    def reset_options(self, checked=False, restore_saved=False):
        from .option_help import template, TRAIN_KEYS
        self.options.set_stage(self.stage.currentData())
        stage = self.stage.currentData()
        model = self.architecture.currentData() if stage == 'train' else None
        self.options_context = (stage, model)
        self.options.allowed_keys = TRAIN_KEYS[model] if model else None
        self.options.minivla=model=='minivla'
        values = dict(self.saved_stage_options.get(self.options_context, template(stage))) if restore_saved else template(stage)
        if model=='minivla' and (not restore_saved or self.options_context not in self.saved_stage_options):
            values={'steps':50000,'val_every':1000,'save_freq':2500,'early_stopping':False,'batch_size':8}
        if model:
            values = {k: v for k, v in values.items() if k in TRAIN_KEYS[model]}
        self.options.setPlainText(json.dumps(values, indent=2, ensure_ascii=False))

    def change_model(self):
        if not hasattr(self, 'options') or self.stage.currentData() != 'train':
            return
        if self.options_context:
            self.saved_stage_options[self.options_context] = json.loads(self.options.toPlainText())
        self.reset_options(restore_saved=True)
        if hasattr(self,'validation_input'):
            visible=self.architecture.currentData()=='minivla'
            self.validation_input.setVisible(visible);self.primary.layout().labelForField(self.validation_input).setVisible(visible)
            self.refresh()
        self.update_ready()

    def change_section(self):
        groups = [('source', 'replay', 'annotate', 'datagen', 'convert', 'train', 'sim_eval', 'hf_dataset', 'hf_policy'),
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

    @staticmethod
    def dataset_description(artifact):
        workspace=artifact.get('workspace') or {}
        return artifact_description(artifact)+f" · {workspace.get('workspace_id','?')}/v{workspace.get('version','?')}"

    def refresh_training_choices(self):
        if not hasattr(self,'checkpoint_choice'):return
        identifier=self.inputs.currentData()
        if not identifier:return
        artifact=self.registry.get(identifier);stage=self.stage.currentData()
        is_policy=stage=='sim_eval' and artifact.get('format')=='so101.minivla-policy/1'
        self.checkpoint_choice.setVisible(is_policy);self.primary.layout().labelForField(self.checkpoint_choice).setVisible(is_policy)
        old=self.checkpoint_choice.currentData();self.checkpoint_choice.blockSignals(True);self.checkpoint_choice.clear()
        if is_policy:
            root=Path(artifact['path'])
            for path in [root/'best.pt',root/'last.pt',*sorted((root/'checkpoints').glob('step_*.pt')),*sorted(root.glob('step_*.pt'))]:
                if path.is_file():self.checkpoint_choice.addItem(str(path.relative_to(root)),str(path))
            index=self.checkpoint_choice.findData(old);self.checkpoint_choice.setCurrentIndex(max(0,index))
        self.checkpoint_choice.blockSignals(False)
        if stage!='train' or self.architecture.currentData()!='minivla':return
        old=self.validation_input.currentData();self.validation_input.blockSignals(True);self.validation_input.clear()
        self.validation_input.addItem('별도 Validation 선택 필요 / Legacy는 고급 설정',None)
        from .episodes import validate_training_inputs
        for candidate in self.registry.all():
            if not self.in_input_folder(candidate):continue
            if candidate['type']!='dataset' or candidate.get('status','SUCCEEDED')!='SUCCEEDED' or not isinstance(candidate.get('format'),dict):continue
            if (candidate.get('provenance',{}).get('source_partition') or {}).get('split')!='val':continue
            try:validate_training_inputs(artifact,{'validation_artifact':candidate})
            except (ValueError,KeyError,OSError,TypeError):continue
            self.validation_input.addItem(self.dataset_description(candidate),candidate['id'])
            self.validation_input.setItemData(self.validation_input.count()-1,self.dataset_description(candidate)+'\nID: '+candidate['id']+'\n'+candidate['path'],Qt.ItemDataRole.ToolTipRole)
        self.validation_input.setCurrentIndex(max(0,self.validation_input.findData(old)));self.validation_input.blockSignals(False)
        if isinstance(artifact.get('format'),dict):self.inputs.setItemText(self.inputs.currentIndex(),self.input_description(artifact))

    def training_metrics_path(self):
        """Resolve the current run's plan, including custom outputs and pre-metric startup."""
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
            try:
                artifact=self.registry.get(identifier);root=Path(artifact['path'])
                if artifact['type']=='policy' and (root/'metrics.jsonl').is_file():return remember(root/'metrics.jsonl')
            except (OSError,ValueError,KeyError):pass
        if identifier:
            for record in reversed(self.registry.all()):
                path=Path(record['path'])/'metrics.jsonl'
                if record['type']=='policy' and record.get('status')=='SUCCEEDED' and identifier in record.get('parents',[]) and path.is_file():return remember(path)
        if session:
            path=from_run(Path(session).parent)
            if path:return remember(path)
        if identifier:
            for run in sorted((self.registry.root/'runs').glob('*_train_*'),reverse=True):
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
        for record in self.registry.all():
            if record['type']=='policy' and record.get('status')=='SUCCEEDED' and (Path(record['path'])/'metrics.jsonl').is_file():
                models.addItem(artifact_description(record),record['id'])
        layout.addWidget(models)
        def select_model():
            identifier=models.currentData()
            chart.resolver=None if identifier else self.training_metrics_path
            chart.path=Path(self.registry.get(identifier)['path'])/'metrics.jsonl' if identifier else None
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
        from .stages import feeds, controls, SIM_STAGES, INSPECTION_STAGES
        if not hasattr(self, 'feeds'): return
        stage = self.stage.currentData()
        self.training_loss_button.setVisible(stage == 'train')
        if stage != 'train' and getattr(self, '_loss_dialog', None) is not None:
            self._loss_dialog.close()
            self._loss_dialog = None
        self.input_label.setText('Train Dataset' if stage=='train' else '사용할 데이터 / 모델')
        visible = feeds(stage)
        cameras = getattr(self, 'scene_cameras', ('side','wrist'))
        if stage == 'real_eval': visible = tuple('real_'+n for n in cameras)
        if stage in SIM_STAGES:
            visible = tuple('sim_'+n for n in cameras)
            if stage in INSPECTION_STAGES: visible = ('sim_overview', *visible)
        for name in visible:
            if name not in self.feeds:
                self.feeds[name] = CameraView(name.upper() + ' — 영상 없음')
                self.feed_titles[name] = QLabel(name.upper().replace('_',' '))
                self.feed_titles[name].setMaximumHeight(24)

        self.images.setVisible(bool(visible)); self.preview_status.setVisible(bool(visible))
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
        self.task_selector.setVisible(stage == 'source')
        self.mode.setVisible(stage in SIM_STAGES); self.main_form.labelForField(self.mode).setVisible(stage in SIM_STAGES)
        descriptions = {'source': 'Leader로 조작 · 저장/다음 또는 리셋 · Episode/Reset Time은 초 단위',
                        'replay': '기록된 episode 재현 · 카메라/관절 오차 확인',
                        'annotate': 'Overview로 동작 확인 · grasp 및 place 판정 · 자동/수동 annotation',
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
