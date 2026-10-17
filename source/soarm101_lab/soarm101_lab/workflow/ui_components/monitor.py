"""Run-local logs, status, supported commands, and immutable result details."""
import json
from pathlib import Path
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QShortcut, QKeySequence
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
                             QPlainTextEdit, QDialog, QSizePolicy)
from ..view import LogTail
from ..stages import controls, feeds, INSPECTION_STAGES
from ..presentation import datagen_status_text, send

COMMAND_LABELS = dict(save='저장 / 다음 (→)', discard='리셋 / 폐기 (←)', quit='기록 종료',
                      resume='재생 / Resume', pause='일시정지', step='한 스텝', mark='Subtask 표시', skip='Episode 건너뛰기')


def inspection_text(status, active):
    signals = status.get('task_signals', {})
    word = lambda value: '충족' if value is True else '미충족' if value is False else '판정 없음'
    text = ('Grasp (휴리스틱): ' + word(signals.get('grasp')) + ' · 작업 성공 조건: ' + word(signals.get('place'))
            + '\nEpisode 결과: ' + status.get('episode_outcome', '진행 중' if active else '대기')
            + ' · ' + status.get('outcome_reason', ''))
    if signals.get('tcp_distance_mm') is not None:
        text += f" · TCP {signals['tcp_distance_mm']:.1f} mm (성공 임계값 없음)"
    if 'attempts' in status:
        text += f"\n시도 {status['attempts']} · 성공 {status.get('successes',0)} · 실패 {status.get('failures',0)}"
    if status.get('failure_breakdown'):
        labels = dict(ik_failure='IK 계산 탈락', planning_failure='계획 실패', simulation_failure='시뮬레이션 실패',
                      grasp_failure='집기 실패', drop_failure='이동 중 놓침', place_failure='컵 배치 실패', timeout='시간 초과')
        text += '\n' + ' · '.join(f'{labels.get(k,k)}: {v}' for k,v in status['failure_breakdown'].items())
    return text


class RunMonitor(QWidget):
    def __init__(self, stage, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.session = None
        self.last_result = None
        self.log_tail = LogTail()
        self.running = False
        self.final_state = None
        layout = QVBoxLayout(self)
        self.action_bar = QWidget()
        row = QHBoxLayout(self.action_bar)
        self.status = QLabel('NOT_READY')
        self.status.setWordWrap(True)
        self.run = QPushButton('실행')
        self.run.setObjectName('primaryAction')
        self.stop = QPushButton('현재 작업 중지')
        self.stop.setObjectName('stopAction')
        self.stop.setEnabled(False)
        row.addWidget(self.status, 1)
        row.addWidget(self.run)
        row.addWidget(self.stop)
        layout.addWidget(self.action_bar)
        self.command_bar = QHBoxLayout()
        layout.addLayout(self.command_bar)
        self.controls, self.shortcuts = {}, []
        self.task_state = None
        if stage in INSPECTION_STAGES:
            self.task_state = QLabel()
            self.task_state.setWordWrap(True)
            self.task_state.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            self.task_state.setObjectName('taskBadge')
            layout.addWidget(self.task_state)
        self.preview_status = None
        if feeds(stage):
            self.preview_status = QLabel()
            self.preview_status.setWordWrap(True)
            layout.addWidget(self.preview_status)
        self.media_layout = QVBoxLayout()
        layout.addLayout(self.media_layout, 2)
        tools = QHBoxLayout()
        tools.addWidget(QLabel('진행 / 오류 로그'))
        self.raw_log = QCheckBox('전체 로그')
        self.raw_button = QPushButton('원본 로그 열기')
        self.result_button = QPushButton('결과 상세')
        self.raw_button.setEnabled(False)
        self.result_button.setEnabled(False)
        self.raw_button.clicked.connect(self.open_log)
        self.result_button.clicked.connect(self.show_result)
        for widget in (self.raw_log, self.raw_button, self.result_button):
            tools.addWidget(widget)
        self.tools = tools
        layout.addLayout(tools)
        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setMaximumBlockCount(4000)
        self.result.setObjectName('console')
        layout.addWidget(self.result, 1)
        self.configure_commands()

    def configure_commands(self, manual=False):
        allowed = controls(self.stage, manual)
        for command in list(self.controls):
            if command not in allowed:
                button = self.controls.pop(command)
                self.command_bar.removeWidget(button)
                button.deleteLater()
        for command in COMMAND_LABELS:
            if command in allowed and command not in self.controls:
                button = QPushButton(COMMAND_LABELS[command])
                button.setEnabled(False)
                button.clicked.connect(lambda _, c=command: self.control(c))
                self.command_bar.addWidget(button)
                self.controls[command] = button
        if self.stage == 'source' and not self.shortcuts:
            for key, command in [('Right', 'save'), ('Left', 'discard')]:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.activated.connect(lambda c=command: self.controls[c].click() if self.controls[c].isEnabled() else None)
                self.shortcuts.append(shortcut)

    def set_readiness(self, state, reason):
        self.run.setEnabled(state == 'READY' and not self.running)
        if not self.running and self.final_state is None:
            self.status.setText(state + (' — '+reason if reason else ''))
        elif not self.running:
            self.run.setToolTip(state + (' — '+reason if reason else ''))

    def begin(self):
        self.session, self.last_result, self.final_state = None, None, None
        self.running = True
        self.log_tail = LogTail()
        self.result.clear()
        if self.task_state is not None:
            self.task_state.clear()
        self.run.setEnabled(False)
        self.stop.setEnabled(True)
        self.raw_button.setEnabled(False)
        self.result_button.setEnabled(False)
        self.status.setText('RUNNING · 환경/정책 준비 중')

    def control(self, command):
        if self.running and self.session and command in self.controls:
            send(self.session, command)

    def poll(self, session=None):
        if session:
            self.session = Path(session)
        status = {}
        if self.session:
            try:
                status = json.loads((self.session/'status.json').read_text())
            except (OSError, ValueError):
                pass
            self.poll_log()
        active = self.running and status.get('state') in ('RUNNING', 'PAUSED')
        for button in self.controls.values():
            button.setEnabled(active)
        if self.running:
            phase = dict(preparing='환경 준비 중', generation='데이터 생성 중', source_validation='Source 사전 검증 중').get(status.get('workflow_phase'), '')
            state = 'PAUSED' if status.get('state') == 'PAUSED' else 'RUNNING'
            if status.get('state') in ('FINISHED', 'STOPPED'):
                phase = '실행 결과 검증 및 등록 중'
            self.status.setText(state + (' · '+phase if phase else ''))
        if self.stage in INSPECTION_STAGES:
            text = datagen_status_text(status) if self.stage == 'datagen' and 'target_successes' in status else inspection_text(status, active)
            self.task_state.setText(text)
            self.task_state.setMinimumHeight(max(self.task_state.heightForWidth(self.task_state.width()),
                self.task_state.fontMetrics().lineSpacing()*(text.count('\n')+1)+20))
        if self.preview_status is not None:
            self.preview_status.setText(f"{status.get('task_id','')} · episode {status.get('episode_index','-')}/{status.get('episode_total','-')} · step {status.get('steps',0)}"
                + '\n' + ' · '.join(str(v) for k,v in status.items() if k.endswith('_camera_error') or k=='leader_limit_status') if self.session else '')
        return status

    def poll_log(self):
        if not self.session:
            return
        path = self.session.parent/'process.log'
        self.raw_button.setEnabled(path.is_file())
        for line in self.log_tail.read(path):
            if self.raw_log.isChecked() or self.log_tail.relevant(line):
                self.result.appendPlainText(line if self.raw_log.isChecked() else self.log_tail.summary(line))

    def finish(self, result, stopped=False):
        self.poll_log()
        self.running = False
        self.last_result = result
        self.final_state = 'STOPPED' if stopped else 'SUCCEEDED' if result['ok'] else 'FAILED'
        self.status.setText(self.final_state)
        self.stop.setEnabled(False)
        for button in self.controls.values():
            button.setEnabled(False)
        self.result_button.setEnabled(True)
        artifact = result.get('artifact', {})
        self.result.appendPlainText(('완료: '+artifact.get('path','중지됨 · 완료된 성공 데이터 0개')) if result['ok'] else '오류: '+result['error'])

    def open_log(self):
        if self.session and (self.session.parent/'process.log').is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.session.parent/'process.log')))

    def show_result(self):
        if self.last_result is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('실행 결과 상세')
        dialog.resize(900, 650)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(json.dumps(self.last_result, indent=2, ensure_ascii=False))
        layout.addWidget(text)
        dialog.exec()
