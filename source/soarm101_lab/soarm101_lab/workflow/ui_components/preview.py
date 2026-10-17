"""Scene camera presentation; never changes recorded camera identity or resolution."""
import time
from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel
from ..view import CameraView
from ..stages import feeds, INSPECTION_STAGES


class CameraPreview(QWidget):
    def __init__(self, stage, cameras=('side', 'wrist'), parent=None):
        super().__init__(parent)
        self.stage = stage
        self.grid = QGridLayout(self)
        self.feeds, self.titles = {}, {}
        self.configure(cameras)

    def configure(self, cameras):
        prefix = 'real_' if self.stage == 'real_eval' else 'sim_'
        names = tuple(prefix + name for name in cameras)
        if self.stage in INSPECTION_STAGES:
            names = ('sim_overview', *names)
        if not feeds(self.stage):
            names = ()
        for name in names:
            if name not in self.feeds:
                self.feeds[name] = CameraView(name.upper() + ' — 실행 대기')
                self.titles[name] = QLabel(name.upper().replace('_', ' '))
        for name, label in list(self.feeds.items()):
            self.grid.removeWidget(label)
            self.grid.removeWidget(self.titles[name])
            if name not in names:
                label.hide()
                label.deleteLater()
                self.titles.pop(name).deleteLater()
                self.feeds.pop(name)
            else:
                label.show()
                self.titles[name].show()
        for index, name in enumerate(names):
            if len(names) == 3:
                row, col = (0, 0) if index == 0 else (2 * (index-1), 1)
            else:
                row, col = 0, index
            self.grid.addWidget(self.titles[name], row, col)
            self.grid.addWidget(self.feeds[name], row+1, col, 3 if len(names)==3 and index==0 else 1, 1)
        self.grid.setColumnStretch(0, 2 if len(names)==3 else 1)
        self.grid.setColumnStretch(1, 1)
        self.grid.setRowStretch(1, 1)
        self.grid.setRowStretch(3, 1 if len(names)==3 else 0)

    def update_frames(self, session, status, running):
        for name, label in self.feeds.items():
            stamp = status.get('frames', {}).get(name, 0)
            fresh = running and time.time()-stamp < 2
            frozen = stamp > 0 and name.startswith('sim_') and session is not None
            if session and (fresh or frozen) and (session/(name+'.jpg')).is_file():
                label.show_image(session/(name+'.jpg'))
            else:
                label.setText(name.upper() + (' — 새 프레임 대기' if running else ' — 연결된 영상 없음'))
