"""MiniVLA metric resolution and plots extracted from the original WorkflowPanel."""
import json
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QDialog, QLabel, QComboBox, QPushButton, QFileDialog
from ..artifacts import artifact_description


class TrainingMetrics:
    def __init__(self, screen):
        self.screen = screen
        self.registry = screen.registry
        self.inputs = screen.inputs
        self._training_metrics_cache = None
        self._loss_dialog = None

    @property
    def busy(self):
        return self.screen.controller.busy and (self.screen.controller.active or {}).get('owner') == self.screen.key

    @property
    def active_stage(self):
        active = self.screen.controller.active or {}
        return active.get('stage') if active.get('owner') == self.screen.key else None

    def training_metrics_path(self):
        """Resolve the current run's plan, including custom outputs and pre-metric startup."""
        session=self.screen.monitor.session
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
        dialog=QDialog(self.screen);self._loss_dialog=dialog
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

class TrainingLossPlot(QWidget):
    """Incremental real measurements; attach before the trainer creates its first record."""
    def __init__(self,path=None,parent=None,*,resolver=None):
        super().__init__(parent)
        from ..view import LogTail
        self.path=Path(path) if path else None;self.resolver=resolver;self.tail=LogTail()
        self.rows=[];self.target=None;self.setMinimumSize(600,350)
    def reload(self):
        import math
        if self.resolver:
            path=self.resolver();self.path=Path(path) if path else None
        if self.path!=self.tail.path:
            from ..view import LogTail
            self.tail=LogTail();self.rows=[];self.target=None
        if self.path and self.path.is_file():
            try:
                if self.path.stat().st_size<self.tail.offset:
                    from ..view import LogTail
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
