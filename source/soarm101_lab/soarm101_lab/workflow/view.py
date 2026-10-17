"""Presentation-only widgets and incremental process-log reader."""
import codecs
import json
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

    @staticmethod
    def summary(line):
        if line.startswith('{'):
            try:
                row=json.loads(line)
                if 'step' in row and ('loss' in row or 'training_loss' in row):
                    from .presentation import training_status_text
                    return training_status_text(row)
            except (ValueError,TypeError,KeyError):pass
        if not line.startswith('[MIMIC] {'):return line
        try:row=json.loads(line[len('[MIMIC] '):])
        except ValueError:return line
        labels={'success':'성공','ik_failure':'IK 계산 탈락','planning_failure':'계획 실패',
            'simulation_failure':'시뮬레이션 실패','grasp_failure':'집기 실패','place_failure':'배치 실패',
            'timeout':'시간 초과','interrupted':'중단','target_already_reached':'목표 수 도달'}
        reason=row.get('reason','')
        if row.get('category')=='ik_failure' and row.get('ik'):
            ik=row['ik'];reason=f"위치 {ik['position_error_m']*1000:.2f}/{ik['position_tolerance_m']*1000:g} mm · 방향 {ik['rotation_error_rad']:.3f}/{ik['rotation_tolerance_rad']:g} rad"
        return f"Env {row['env_id']} · 시도 {row['attempt']} · {labels.get(row['category'],row['category'])} · {row.get('steps',0)} step"+(' · '+reason[:300] if reason else '')
