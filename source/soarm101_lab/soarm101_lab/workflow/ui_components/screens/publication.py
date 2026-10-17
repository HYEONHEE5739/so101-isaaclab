"""Dataset/policy publication share one form and explicit scope confirmation."""
from PyQt6.QtWidgets import QMessageBox
from .base import StageScreen


class PublicationScreen(StageScreen):
    action = '업로드 시작'
    basic_keys = {'repo_id', 'private', 'create_repo'}

    def execute(self):
        snapshot = self.snapshot()
        opts = snapshot['options']
        if type(opts.get('private')) is not bool:
            raise ValueError('Private/Public 공개 범위를 선택하세요.')
        answer = QMessageBox.question(self, '업로드 대상 확인',
            f"대상: {snapshot['artifact']['path']}\nRepo: {opts.get('repo_id','')}\n공개 범위: {'Private' if opts['private'] else 'Public'}\n{'신규 Repo 생성' if opts.get('create_repo') else '기존 Repo 사용'}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.controller.execute(self.key, snapshot)
