"""Real evaluation approval UI; all worker safety checks remain in Operations/runtime."""
from PyQt6.QtWidgets import QLabel, QMessageBox
from .base import StageScreen


class RealEvaluationScreen(StageScreen):
    action = 'Real 평가 시작 · 모터 동작'
    basic_keys = {'port', 'follower_id', 'camera_devices', 'side_device', 'wrist_device',
                  'episodes', 'max_steps', 'enable_motion'}

    def build_inputs(self):
        note = QLabel('실제 모터 동작 · Enable Motion은 기본 OFF입니다.\nFollower calibration ID, 카메라, 연결 상태를 확인하세요.\n실제 성공은 사람이 결과 영상을 검토해야 합니다.')
        note.setWordWrap(True)
        self.form.addRow(note)

    def execute(self):
        snapshot = self.snapshot()
        if not snapshot['options'].get('enable_motion'):
            raise ValueError('Real evaluation requires explicit enable_motion=true')
        live = getattr(self.window(), 'bus', None)
        if live and live.owner_alive():
            raise ValueError('Real/Sim을 Disconnect한 뒤 Workflow를 실행하세요.')
        opts = snapshot['options']
        answer = QMessageBox.question(self, '실제 모터 동작 승인',
            f"Policy: {snapshot['artifact']['id']}\nFollower: {opts.get('port','')} / {opts.get('follower_id','')}\n실제 로봇을 움직입니다. 실행을 승인합니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.controller.execute(self.key, snapshot)
