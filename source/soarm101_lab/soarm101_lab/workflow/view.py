"""Presentation-only widgets and incremental process-log reader."""
import codecs
import re
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class CameraView(QLabel):
    def __init__(self, text='', minimum=(120, 90)):
        super().__init__(text)
        self.setMinimumSize(*minimum)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def show_image(self, path):
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            self.setPixmap(pixmap)

    def paintEvent(self, event):
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        size = pix.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x, y = (self.width() - size.width()) // 2, (self.height() - size.height()) // 2
        painter.drawPixmap(x, y, size.width(), size.height(), pix)


class LogTail:
    def __init__(self):
        self.path = None
        self.offset = 0
        self.pending = ''
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.traceback = False

    def read(self, path):
        if path != self.path:
            self.__init__(); self.path = path
        if not path.exists(): return []
        with path.open('rb') as stream:
            stream.seek(self.offset); chunk = stream.read(65536); self.offset = stream.tell()
        self.pending += self.decoder.decode(chunk)
        lines = self.pending.split('\n'); self.pending = lines.pop()
        return [line.rstrip('\r') for line in lines]

    def relevant(self, line):
        if 'Traceback (most recent call last)' in line: self.traceback = True
        if self.traceback:
            if line and not line.startswith((' ', 'Traceback')): self.traceback = False
            return True
        if line.lstrip().startswith('[DIAGNOSTIC]'): return False
        # Encoder configuration is noise in the summary; retain warnings/errors.
        if line.lstrip().lower().startswith('svt[info]'): return False
        return bool(re.search(r'\[WORKFLOW\]|error|exception|out of memory|failed|failure|✅|❌|🛑|'
                              r'episode|annotat|exported|processed|subtask|final task|reset|saving|saved|replay|'
                              r'success|generation|trial|step:|loss|checkpoint|learning.rate', line, re.I)) and not line.startswith(('|', '+'))
