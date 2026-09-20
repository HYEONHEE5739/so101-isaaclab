"""One guided/independent workflow panel backed by Operations, not shell strings."""
import json
import time
from pathlib import Path
from PyQt6.QtCore import QUrl, QTimer, Qt
from PyQt6.QtGui import QDesktopServices, QPixmap, QShortcut, QKeySequence
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
                            QPushButton, QLineEdit, QPlainTextEdit, QTextBrowser, QFileDialog, QMessageBox, QGridLayout, QScrollArea, QCheckBox, QSplitter, QToolButton, QDialog, QSpinBox, QDoubleSpinBox)
from .artifacts import Registry, browse_directory
from .operations import Operations, STAGES, default_python
from ..real2sim.workspaces.package import ROOT, load, metadata

LABELS = {'accept': 'Revision 승인', 'publish': 'Publish Workspace', 'source': 'Source Demo',
          'replay': 'Replay', 'annotate': 'Annotation', 'datagen': 'Mimic / Datagen',
          'convert': 'LeRobot 변환', 'train': 'ACT / SmolVLA 학습', 'hf_dataset': 'Dataset 업로드',
          'hf_policy': 'Policy 업로드', 'sim_eval': 'Sim 평가', 'real_eval': 'Real 평가 (모터 동작)'}



class WorkflowPanel(QWidget):
    def __init__(self, run_worker, parent=None):
        super().__init__(parent)
        self.run_worker = run_worker; self.registry = Registry(); self.service = Operations(self.registry)
        self.busy = False
        from .view import CameraView, LogTail
        self.log_tail = LogTail(); self.last_result = None
        outer = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Orientation.Horizontal); outer.addWidget(self.splitter)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setMinimumWidth(280)
        self.splitter.addWidget(scroll)
        viewer = QWidget(); display = QVBoxLayout(viewer); self.splitter.addWidget(viewer)
        self.splitter.setStretchFactor(0, 0); self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([350, 900])
        content = QWidget(); scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.addWidget(QLabel('Workspace 선택 → Source Demo → Replay → Annotation'))
        self.section = QComboBox(); self.section.addItems(['Sim 작업 파이프라인', '환경 관리', '실제 로봇 평가'])
        layout.addWidget(self.section)
        form = QFormLayout(); form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); layout.addLayout(form)
        self.main_form = form
        self.workspace = QLineEdit(); form.addRow('Workspace ID / 경로', self.workspace)
        self.python = QLineEdit(default_python()); form.addRow('실행 Python (Isaac / LeRobot)', self.python)
        self.mode = QComboBox()
        self.mode.addItem('Operator — 통합 UI에서 영상/조작', 'operator')
        self.mode.addItem('Developer — Isaac 창 함께 실행', 'developer')
        form.addRow('실행 모드', self.mode)
        self.real_enabled = QCheckBox('REAL 카메라 영상 연결 (로봇 모터 연결 아님)')
        form.addRow(self.real_enabled)
        self.real_enabled.hide()
        self.real_side = QLineEdit('/dev/cam_side'); self.real_wrist = QLineEdit('/dev/cam_wrist')
        self.real_side.hide(); self.real_wrist.hide()
        self.stage = QComboBox()
        for k in ('source', 'replay', 'annotate', 'datagen', 'convert', 'train', 'sim_eval', 'hf_dataset', 'hf_policy'): self.stage.addItem(LABELS[k], k)
        form.addRow('단계', self.stage)
        self.inputs = QComboBox(); form.addRow('등록된 입력 artifact', self.inputs)
        self.kind = QComboBox(); form.addRow('수동 선택 파일 유형', self.kind)
        row = QGridLayout(); layout.addLayout(row)
        self.browse_buttons = []
        for index, (title, fn) in enumerate( [('입력 파일/폴더 선택', self.browse), ('입력 폴더 열기', self.open_folder),
                          ('Workspace 선택', self.browse_workspace), ('Artifact 새로고침', self.refresh)]):
            b = QPushButton(title); b.clicked.connect(lambda _, f=fn: self.guard(f)); row.addWidget(b, index // 2, index % 2); self.browse_buttons.append(b)
        from .task_ui import TaskSelector
        self.task_selector = TaskSelector(); layout.addWidget(self.task_selector)
        self.task_selector.changed.connect(self.update_ready)
        advanced = QToolButton(); advanced.setText('실행 설정 · JSON 예시 / 항목 설명'); advanced.setCheckable(True)
        layout.addWidget(advanced)
        self.primary = QWidget(); primary_form = QFormLayout(self.primary)
        self.port = QLineEdit('/dev/so101_leader'); primary_form.addRow('Leader port', self.port)
        self.episodes = QSpinBox(); self.episodes.setRange(1, 100000); primary_form.addRow('Episodes', self.episodes)
        self.episode_time = QDoubleSpinBox(); self.episode_time.setRange(.1, 3600); self.episode_time.setValue(20); self.episode_time.setSuffix(' s')
        self.reset_time = QDoubleSpinBox(); self.reset_time.setRange(0, 600); self.reset_time.setValue(5); self.reset_time.setSuffix(' s')
        primary_form.addRow('Episode Time', self.episode_time); primary_form.addRow('Reset Time', self.reset_time)
        self.annotation_mode = QComboBox(); self.annotation_mode.addItems(['자동 Annotation', '수동 Annotation'])
        primary_form.addRow('Annotation 방식', self.annotation_mode)
        layout.insertWidget(layout.count() - 1, self.primary)
        self.port.textChanged.connect(self.update_ready); self.episodes.valueChanged.connect(self.update_ready)
        self.annotation_mode.currentIndexChanged.connect(self.configure_stage)
        self.episode_time.valueChanged.connect(self.update_ready); self.reset_time.valueChanged.connect(self.update_ready)
        self.advanced_panel = QWidget(); options_layout = QVBoxLayout(self.advanced_panel)
        options_layout.setContentsMargins(0, 0, 0, 0)
        self.options_help = QTextBrowser(); self.options_help.setMinimumHeight(200)
        options_layout.addWidget(self.options_help)
        self.options = QPlainTextEdit(); self.options.setMinimumHeight(200)
        options_layout.addWidget(QLabel('실행 JSON — 아래 기본값에서 필요한 값만 수정'))
        options_layout.addWidget(self.options)
        restore = QPushButton('현재 단계 JSON을 기본 예시로 되돌리기')
        restore.clicked.connect(self.reset_options); options_layout.addWidget(restore)
        self.advanced_panel.hide()
        advanced.toggled.connect(self.advanced_panel.setVisible); layout.addWidget(self.advanced_panel)
        layout.addStretch()
        self.status = QLabel(); self.status.setWordWrap(True); display.addWidget(self.status)
        self.run = QPushButton('선택 단계 Run'); self.run.clicked.connect(lambda: self.guard(self.execute)); display.addWidget(self.run)
        self.stop = QPushButton('현재 workflow 중지'); self.stop.clicked.connect(self.service.cancel); display.addWidget(self.stop)
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
        self.task_state = QLabel(); self.task_state.setWordWrap(True); display.insertWidget(4, self.task_state)
        log_panel = QWidget(); log_layout = QVBoxLayout(log_panel); self.media_splitter.addWidget(log_panel)
        log_tools = QHBoxLayout(); log_layout.addLayout(log_tools)
        log_tools.addWidget(QLabel('진행 / 오류 로그'))
        self.raw_log = QCheckBox('전체 로그'); log_tools.addWidget(self.raw_log)
        for title, fn in [('원본 로그 열기', self.open_log), ('결과 상세', self.show_result)]:
            button = QPushButton(title); button.clicked.connect(fn); log_tools.addWidget(button)
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
        self.timer = QTimer(self); self.timer.timeout.connect(self.poll_preview); self.timer.start(100)
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
            saved = json.loads(settings.read_text()); self.workspace.setText(saved.get('workspace', ''))
            self.python.setText(saved.get('python', default_python()))
            self.mode.setCurrentIndex(max(0, self.mode.findData(saved.get('mode', 'operator'))))
        elif (ROOT / 'outputs/real2sim/environments/workspace_001/v1/manifest.json').exists():
            self.workspace.setText(str(ROOT / 'outputs/real2sim/environments/workspace_001/v1'))
        self.stage.currentIndexChanged.connect(self.changed)
        self.inputs.currentIndexChanged.connect(self.sync_tasks)
        self.options.textChanged.connect(self.update_ready)
        self.workspace.editingFinished.connect(self.refresh)
        # Existing selected revision/workspace can be imported without executing any stage.
        try:
            settings = ROOT / 'outputs/real2sim/ui_settings.json'
            if settings.exists():
                path = json.loads(settings.read_text()).get('profile')
                if path and not any(a['path'] == str(Path(path).resolve()) for a in self.registry.all()):
                    self.registry.register('revision', path)
            if self.workspace.text():
                root = load(self.workspace.text())['root']
                if not any(a['path'] == str(root) for a in self.registry.all()):
                    self.registry.register('workspace', root)
        except Exception as exc:
            self.result.setPlainText(str(exc))
        for combo in self.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
        self.changed()

    def guard(self, fn):
        try: fn()
        except Exception as exc: QMessageBox.warning(self, 'Workflow', str(exc))

    def changed(self):
        stage = self.stage.currentData()
        if not stage: return
        self.configure_stage()
        self.primary.setVisible(stage in ('source', 'annotate'))
        primary_form = self.primary.layout()
        for widget, visible in ((self.port, stage == 'source'), (self.episodes, stage == 'source'), (self.episode_time, stage == 'source'), (self.reset_time, stage == 'source'), (self.annotation_mode, stage == 'annotate')):
            widget.setVisible(visible); primary_form.labelForField(widget).setVisible(visible)
        from .option_help import help_html
        self.options_help.setHtml(help_html(stage))
        self.reset_options()
        self.kind.clear(); self.kind.addItems(sorted(STAGES[stage][0])); self.refresh()

    def refresh(self):
        selected = self.inputs.currentData(); self.inputs.blockSignals(True); self.inputs.clear()
        allowed = STAGES[self.stage.currentData()][0]
        try: workspace = metadata(load(self.workspace.text())) if self.workspace.text() else None
        except Exception: workspace = None
        for a in self.registry.all():
            if a['type'] in allowed and (not workspace or not a.get('workspace') or a['workspace'] == workspace):
                self.inputs.addItem(a['type'] + ' · ' + a['path'], a['id'])
        index = self.inputs.findData(selected)
        self.inputs.setCurrentIndex(index if index >= 0 else self.inputs.count() - 1)
        self.inputs.blockSignals(False); self.sync_tasks(); self.update_ready()

    def update_ready(self):
        if self.busy:
            self.status.setText('RUNNING — 완료 후 output 검증 및 등록'); self.run.setEnabled(False); return
        try:
            artifact = self.registry.get(self.inputs.currentData())
            opts = self.stage_options()
            state, reason = self.service.readiness(self.stage.currentData(), artifact, self.workspace.text(), opts)
        except Exception as exc:
            state, reason = 'NOT_READY', str(exc)
        self.status.setText(state + (' — ' + reason if reason else '')); self.run.setEnabled(state == 'READY')

    def browse(self):
        kind = self.kind.currentText(); directory = browse_directory(kind, registry=self.registry)
        if kind in ('workspace', 'dataset', 'policy'):
            path = QFileDialog.getExistingDirectory(self, 'Artifact 선택', directory)
        else:
            path, _ = QFileDialog.getOpenFileName(self, 'Artifact 선택', directory, 'Artifacts (*.json *.hdf5 *.h5)')
        if path:
            artifact = self.registry.register(kind, path)
            if kind == 'workspace': self.workspace.setText(artifact['path'])
            self.refresh(); self.inputs.setCurrentIndex(self.inputs.findData(artifact['id']))

    def browse_workspace(self):
        path = QFileDialog.getExistingDirectory(self, 'manifest.json이 있는 version 폴더 선택', browse_directory('workspace', self.workspace.text()))
        if path:
            w = load(path); self.workspace.setText(str(w['root'])); self.registry.register('workspace', path); self.refresh()

    def open_folder(self):
        a = self.registry.get(self.inputs.currentData()); p = Path(a['path'])
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p if p.is_dir() else p.parent)))

    def execute(self):
        from ..real2sim.storage import atomic
        stage = self.stage.currentData(); artifact = self.registry.get(self.inputs.currentData())
        opts = self.stage_options(); opts['python'] = self.python.text(); opts['mode'] = self.mode.currentData()
        workspace = self.workspace.text()
        atomic(self.registry.root / 'ui.json', {'workspace': workspace, 'python': self.python.text(), 'mode': self.mode.currentData()})
        self.service.session = None
        from .view import LogTail
        self.log_tail = LogTail(); self.result.clear(); self.last_result = None
        self.section.setEnabled(False); self.stage.setEnabled(False)
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
            self.result.appendPlainText(('완료: ' + result['artifact']['path']) if result['ok'] else ('오류: ' + result['error']))
            if result['ok']:
                a = result['artifact']
                if a['type'] == 'workspace': self.workspace.setText(a['path'])
            self.refresh()
            if result['ok']:
                desired = artifact['id'] if stage == 'replay' else result['artifact']['id']
                index = self.inputs.findData(desired)
                if index >= 0: self.inputs.setCurrentIndex(index)
        try: self.run_worker(operation, done)
        except Exception:
            self.busy = False; self.task_selector.setEnabled(True); self.section.setEnabled(True); self.stage.setEnabled(True); self.update_ready(); raise

    def control(self, command):
        from .presentation import send
        if self.busy and self.service.session:
            send(self.service.session, command)

    def poll_preview(self):
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
            self.task_state.setText('접근: ' + ('TCP 거리 %.1f mm (성공 임계값 없음)' % signals['tcp_distance_mm'] if signals.get('tcp_distance_mm') is not None else '측정 없음')
                + '   |   Grasp (휴리스틱): ' + word(signals.get('grasp')) + '   |   컵 안 정지: ' + word(signals.get('place'))
                + '\nEpisode 결과: ' + status.get('episode_outcome', '진행 중' if active else '대기')
                + '   ' + status.get('outcome_reason', '')
                + (f" · 시도 {status['attempts']} / 성공 {status['successes']} / 실패 {status['failures']}" if 'attempts' in status else ''))
            outcome = status.get('episode_outcome', '')
            color = '#dff3e3' if outcome.startswith('성공') else '#ffe4d9' if outcome.startswith('실패') else '#e9eef5'
            self.task_state.setStyleSheet(f'QLabel {{ background: {color}; color: #17202a; padding: 8px; }}')
        if self.busy:
            text = status.get('state', 'STARTING — 환경/정책 준비 중')
            if status.get('frames') and time.time() - max(status['frames'].values()) >= 2:
                text += ' · 마지막 프레임 (새 프레임 대기)'
            self.preview_status.setText(f"{text} · {status.get('task_id', '')} · step {status.get('steps', 0)} · "
                                       f"episode {status.get('episode_index', '-')}/{status.get('episode_total', '-')} · action {status.get('action_index', '-')} · marks {status.get('marks', [])}"
                                       + '\n' + ' · '.join(str(v) for k, v in status.items() if k.endswith('_camera_error')))
        elif root:
            self.preview_status.setText('실행 종료 · 마지막 SIM 프레임 표시 · 진행 로그에서 결과를 확인하세요.')

    def sync_tasks(self):
        from .tasks import from_artifact
        try:
            w = load(self.workspace.text())
            artifact = self.registry.get(self.inputs.currentData()) if self.inputs.currentData() else {}
            self.task_selector.configure(w, self.stage.currentData(), from_artifact(artifact, w))
        except (ValueError, OSError, TypeError, KeyError):
            self.task_selector.summary.setText('유효한 workspace와 입력 artifact를 선택하세요.')
        self.update_ready()

    def stage_options(self):
        opts = json.loads(self.options.toPlainText())
        if not isinstance(opts, dict):
            raise ValueError('실행 설정은 {"항목": 값} 형식의 JSON 객체여야 합니다.')
        stage = self.stage.currentData()
        if stage == 'source': opts.update(port=self.port.text(), episodes=self.episodes.value(), episode_time_s=self.episode_time.value(), reset_time_s=self.reset_time.value())
        if stage == 'annotate': opts['auto'] = self.annotation_mode.currentIndex() == 0
        if stage in ('source', 'sim_eval', 'real_eval'):
            opts['tasks'] = self.task_selector.selection(evaluation=stage != 'source')
        return opts

    def reset_options(self):
        from .option_help import template
        self.options.setPlainText(json.dumps(template(self.stage.currentData()), indent=2, ensure_ascii=False))

    def change_section(self):
        groups = [('source', 'replay', 'annotate', 'datagen', 'convert', 'train', 'sim_eval', 'hf_dataset', 'hf_policy'),
                  ('accept', 'publish'), ('real_eval',)]
        self.stage.blockSignals(True); self.stage.clear()
        for key in groups[self.section.currentIndex()]: self.stage.addItem(LABELS[key], key)
        self.stage.blockSignals(False); self.changed()

    def poll_log(self):
        if not self.service.session: return
        path = self.service.session.parent / 'process.log'
        for line in self.log_tail.read(path):
            if self.raw_log.isChecked() or self.log_tail.relevant(line):
                self.result.appendPlainText(line)

    def open_log(self):
        if self.service.session:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.service.session.parent / 'process.log')))

    def show_result(self):
        dialog = QDialog(self); dialog.setWindowTitle('실행 결과 상세'); dialog.resize(900, 650)
        layout = QVBoxLayout(dialog); text = QPlainTextEdit(); text.setReadOnly(True)
        text.setPlainText(json.dumps(self.last_result, indent=2, ensure_ascii=False)); layout.addWidget(text)
        dialog.exec()

    def configure_stage(self):
        from .stages import feeds, controls, SIM_STAGES, INSPECTION_STAGES
        if not hasattr(self, 'feeds'): return
        stage = self.stage.currentData()
        visible = feeds(stage)
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
        self.task_state.setText('접근 / Grasp / 컵 안 정지: 실행 대기')
        self.task_selector.setVisible(stage not in ('accept', 'publish', 'hf_dataset', 'hf_policy'))
        for widget, show in ((self.mode, stage in SIM_STAGES), (self.kind, len(STAGES[stage][0]) > 1)):
            widget.setVisible(show); self.main_form.labelForField(widget).setVisible(show)
        self.browse_buttons[0].setText('Workspace 선택' if stage == 'source' else 'Policy 폴더 선택' if stage in ('sim_eval', 'real_eval', 'hf_policy') else 'Dataset 폴더 선택' if stage in ('train', 'hf_dataset') else '입력 파일 선택')
        descriptions = {'source': 'Leader로 조작 · 저장/다음 또는 리셋 · Episode/Reset Time은 초 단위',
                        'replay': '기록된 episode 재현 · 카메라/관절 오차 확인',
                        'annotate': 'Overview로 동작 확인 · grasp 및 place 판정 · 자동/수동 annotation',
                        'datagen': '생성 동작 preview · grasp/place 신호 및 실행 로그',
                        'convert': 'HDF5 → LeRobot · episode 변환 진행과 결과 경로',
                        'train': '학습 step / loss / checkpoint는 아래 실시간 로그에서 확인',
                        'sim_eval': 'Task별 평가 · Overview / Side / Wrist · episode 성공/실패',
                        'real_eval': '실제 카메라 및 실행 로그 · 실제 성공은 자동 판정하지 않음'}
        self.stage_summary.setText(descriptions.get(stage, LABELS.get(stage, '')))
        self.update_ready()
