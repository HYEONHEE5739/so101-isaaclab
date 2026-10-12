"""Dependency-free Qt plots for dataset inspection; no simulator/hardware launch."""
import json
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QThread, QTimer, QElapsedTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QPolygonF
from PyQt6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QPushButton,QLabel,QListWidget,
    QListWidgetItem,QSpinBox,QFileDialog,QMessageBox,QSplitter,QCheckBox,QWidget,QSlider,QComboBox)
from .curation import analyze, select_distribution, export_subset


class Job(QThread):
    result=pyqtSignal(object)
    failed=pyqtSignal(str)
    def __init__(self,fn,parent):
        super().__init__(parent);self.fn=fn
    def run(self):
        try:self.result.emit(self.fn())
        except Exception as exc:self.failed.emit(str(exc))


class Plot(QWidget):
    picked=pyqtSignal(int)
    def __init__(self, mode):
        super().__init__();self.mode=mode;self.rows=[];self.row=None;self.excluded=set();self.selected=-1
        self.yaw=.7;self.pitch=.55;self.drag=None;self.screen=[];self.zoom=1.0
        self.robot=None;self.frame=0;self.mesh=None
        self.setMinimumSize(240,200)
    def set_view(self, name):
        self.zoom=1.0
        self.yaw,self.pitch={'3D':(.7,.65),'XY':(0.,0.),'XZ':(0.,np.pi/2),'YZ':(np.pi/2,np.pi/2)}[name]
        self.update()

    def bounds(self):
        xyz=np.asarray(self.row['ee'],dtype=float)
        lo,hi=xyz.min(0),xyz.max(0)
        if self.robot is not None:
            lo=np.minimum(lo,self.robot_bounds[0]);hi=np.maximum(hi,self.robot_bounds[1])
        pad=np.maximum((hi-lo)*.12,.01)
        return lo-pad,hi+pad

    def project(self, points):
        p=np.asarray(points,dtype=float)
        if self.mode=='trajectory':
            lo,hi=self.bounds();p=p-(lo+hi)/2
            a,b=self.yaw,self.pitch
            p=p@np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
            p=p@np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
            # Equal metric scale on all axes, stable while rotating.
            scale=max(1,min(self.width()-150,self.height()-125))/max(np.linalg.norm(hi-lo),1e-6)
            p=p[:,:2]*scale*self.zoom
        else:
            p=p[:,:2];lo=p.min(0);span=np.maximum(p.max(0)-lo,1e-6)
            scale=min((self.width()-70)/span[0],(self.height()-75)/span[1])
            p=(p-(lo+span/2))*scale
        return [QPointF(x+self.width()/2,self.height()/2-y) for x,y in p]

    def draw_axes(self,painter):
        lo,hi=self.bounds()
        painter.setPen(QPen(QColor('#344454'),1))
        for t in np.linspace(0,1,5):
            for axis in (0,1):
                a=lo.copy();b=lo.copy();a[axis]=b[axis]=lo[axis]+t*(hi[axis]-lo[axis])
                b[1-axis]=hi[1-axis]
                painter.drawLine(*self.project([a,b]))
        from itertools import product
        for bits in product((0,1),repeat=3):
            a=np.where(bits,hi,lo)
            for axis in range(3):
                if bits[axis]:continue
                b=a.copy();b[axis]=hi[axis]
                painter.drawLine(*self.project([a,b]))
        for axis,(name,color) in enumerate(zip('XYZ',('#ff8b8b','#73df9d','#75bfff'))):
            end=lo.copy();end[axis]=hi[axis]
            painter.setPen(QPen(QColor(color),2));painter.drawLine(*self.project([lo,end]))
            for t in (0,.5,1):
                v=lo.copy();v[axis]+=t*(hi[axis]-lo[axis]);point=self.project([v])[0]
                painter.drawEllipse(point,2,2)
                offset=QPointF(5,14+axis*12 if t==0 else -6)
                painter.drawText(point+offset,f'{name} {v[axis]*1000:.0f}')
    def set_frame(self,frame):
        self.frame=frame
        self.mesh=self.robot.triangles(self.row['robot_q'][frame]) if self.robot is not None else None
        self.update()

    def draw_robot(self,painter):
        if self.mesh is None:return
        tri,dark=self.mesh
        # Painter's algorithm, using the same view rotation as trajectory projection.
        a,b=self.yaw,self.pitch
        rot=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])@np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
        view=tri@rot
        normals=np.cross(view[:,1]-view[:,0],view[:,2]-view[:,0])
        light=.45+.55*np.abs(normals[:,2])/np.maximum(np.linalg.norm(normals,axis=1),1e-12)
        points=self.project(tri.reshape(-1,3))
        painter.setPen(Qt.PenStyle.NoPen)
        for i in np.argsort(view[:,:,2].mean(1)):
            rgb=np.array([75,85,100] if dark[i] else [163,103,219])*light[i]
            painter.setBrush(QColor(*map(int,rgb)))
            painter.drawPolygon(QPolygonF(points[3*i:3*i+3]))

    def paintEvent(self,event):
        painter=QPainter(self);painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(),QColor('#141b23'));painter.setPen(QColor('#b5c5d3'))
        title={'scatter':'초기 목표 물체 · Workspace XY [mm]','trajectory':'EE 경로 · XYZ [mm] · 드래그 회전 / 휠 확대·축소','actions':'관절 목표값 [rad] · 가로축: 시간 [s]'}[self.mode]
        painter.drawText(12,22,title)
        if self.mode=='scatter' and self.rows:
            self.screen=self.project([r['xy'] for r in self.rows])
            for i,(row,point) in enumerate(zip(self.rows,self.screen)):
                color='#66717d' if row['name'] in self.excluded else '#efb467' if row['flags'] else '#55d6b0'
                painter.setPen(QPen(QColor('#ffffff' if i==self.selected else color),2))
                painter.setBrush(QColor(color));painter.drawEllipse(point,6 if i==self.selected else 4,6 if i==self.selected else 4)
            xy=np.asarray([r['xy'] for r in self.rows])*1000
            painter.setPen(QColor('#b5c5d3'));painter.drawText(12,self.height()-10,f"X {xy[:,0].min():.1f} … {xy[:,0].max():.1f} / Y {xy[:,1].min():.1f} … {xy[:,1].max():.1f} mm")
        elif self.row and self.mode=='trajectory':
            self.draw_axes(painter)
            self.draw_robot(painter)
            points=self.project(self.row['ee']);painter.setPen(QPen(QColor('#55d6b0'),2))
            for i,(a,b) in enumerate(zip(points,points[1:])):
                painter.setPen(QPen(QColor('#55d6b0' if i<self.frame else '#4b6870'),2));painter.drawLine(a,b)
            painter.setPen(QColor('#ffffff'));painter.setBrush(QColor('#ffffff'));painter.drawEllipse(points[self.frame],6,6)
            for point,label,color in [(points[0],'START','#55d6b0'),(points[-1],'END','#efb467')]:
                painter.setPen(QColor(color));painter.drawEllipse(point,5,5);painter.drawText(point,label)
            xyz=np.asarray(self.row['ee']);painter.drawText(12,self.height()-10,'XYZ span [mm]: '+str(np.round(np.ptp(xyz,axis=0)*1000,1)))
        elif self.row and self.mode=='actions':
            q=np.asarray(self.row['actions']);low=q.min();span=max(q.max()-low,1e-6)
            colors=['#55d6b0','#69b9ff','#efb467','#d196fa','#fb9292','#e5e7eb']
            for j,color in enumerate(colors):
                painter.setPen(QPen(QColor(color),1))
                points=[QPointF(35+i/max(len(q)-1,1)*(self.width()-55),self.height()-35-(v-low)/span*(self.height()-75)) for i,v in enumerate(q[:,j])]
                for a,b in zip(points,points[1:]):painter.drawLine(a,b)
                painter.drawText(15+j*65,40,str(j+1))
            painter.setPen(QPen(QColor('#ffffff'),1))
            x=35+self.frame/max(len(q)-1,1)*(self.width()-55);painter.drawLine(QPointF(x,45),QPointF(x,self.height()-35))
            painter.drawText(12,self.height()-10,f"0 … {self.row['duration']:.2f}s / {low:.2f} … {q.max():.2f} rad (1–5 arm, 6 gripper)")
        painter.end()
    def wheelEvent(self,event):
        if self.mode!='trajectory':
            event.ignore();return
        delta=event.angleDelta().y()/120
        if not delta:delta=event.pixelDelta().y()/120
        self.zoom=float(np.clip(self.zoom*1.15**np.clip(delta,-20,20),.2,8.0))
        self.update();event.accept()

    def mousePressEvent(self,event):
        self.drag=event.position()
        if self.mode=='scatter' and self.screen:
            d=[(p.x()-self.drag.x())**2+(p.y()-self.drag.y())**2 for p in self.screen]
            if min(d)<225:self.picked.emit(int(np.argmin(d)))
    def mouseMoveEvent(self,event):
        if self.drag is not None and self.mode=='trajectory':
            delta=event.position()-self.drag;self.yaw+=delta.x()*.01;self.pitch+=delta.y()*.01
            self.drag=event.position();self.update()
    def mouseReleaseEvent(self,event):self.drag=None


class CurationDialog(QDialog):
    def __init__(self,registry,parent=None):
        super().__init__(parent);self.registry=registry;self.rows=[];self.excluded=set();self.path=None;self.kind='datagen';self.job=None
        self.setWindowTitle('Data Curation · 분포와 동작 검토');self.resize(1400,900)
        layout=QVBoxLayout(self);bar=QHBoxLayout();layout.addLayout(bar)
        load=QPushButton('데이터 열기');load.clicked.connect(self.open_file);bar.addWidget(load)
        folder=QPushButton('데이터 폴더 열기');folder.clicked.connect(self.open_folder);bar.addWidget(folder)
        self.count=QSpinBox();self.count.setRange(1,100000);bar.addWidget(QLabel('최종 개수'));bar.addWidget(self.count)
        choose=QPushButton('분포 유지 자동 선별');choose.clicked.connect(self.auto_select);bar.addWidget(choose)
        restore=QPushButton('전체 복원');restore.clicked.connect(self.restore_all);bar.addWidget(restore)
        export=QPushButton('선택 데이터셋 저장');export.clicked.connect(self.save);bar.addWidget(export)
        self.info=QLabel('원본은 보존됩니다. 회색=제외 / 주황=확인 필요 / 초록=포함. 점 또는 목록을 선택하세요.');self.info.setWordWrap(True);layout.addWidget(self.info)
        split=QSplitter();layout.addWidget(split,1)
        left=QWidget();ll=QVBoxLayout(left);split.addWidget(left)
        self.scatter=Plot('scatter');ll.addWidget(self.scatter,1)
        self.only_flags=QCheckBox('확인 필요 에피소드만 표시');self.only_flags.toggled.connect(self.refresh_list);ll.addWidget(self.only_flags)
        self.list=QListWidget();ll.addWidget(self.list,1)
        self.exclude=QPushButton('선택 episode 제외 / 복원');self.exclude.clicked.connect(self.toggle);ll.addWidget(self.exclude)
        right=QWidget();rl=QVBoxLayout(right);split.addWidget(right)
        self.metrics=QLabel('에피소드를 선택하세요.');self.metrics.setWordWrap(True);rl.addWidget(self.metrics)
        self.trajectory=Plot('trajectory')
        views=QHBoxLayout();views.addWidget(QLabel('경로 시점'));views.addStretch()
        for name in ('3D','XY','XZ','YZ'):
            button=QPushButton(name);button.clicked.connect(lambda checked=False,n=name:self.trajectory.set_view(n));views.addWidget(button)
        rl.addLayout(views);rl.addWidget(self.trajectory,2)
        self.trajectory.set_view('3D')
        playbar=QHBoxLayout()
        self.play=QPushButton('재생');self.play.clicked.connect(self.toggle_play);playbar.addWidget(self.play)
        self.timeline=QSlider(Qt.Orientation.Horizontal);self.timeline.valueChanged.connect(self.seek);playbar.addWidget(self.timeline,1)
        self.clock_label=QLabel('0 / 0');playbar.addWidget(self.clock_label)
        self.rate=QComboBox();self.rate.addItems(['0.25×','0.5×','1×','2×']);self.rate.setCurrentIndex(2)
        self.rate.currentIndexChanged.connect(self.pause);playbar.addWidget(self.rate);rl.addLayout(playbar)
        self.play_timer=QTimer(self);self.play_timer.setInterval(33);self.play_timer.timeout.connect(self.advance)
        self.elapsed=QElapsedTimer();self.play_start=0;self.robot_cache={}
        self.actions=Plot('actions');rl.addWidget(self.actions,1)
        split.setSizes([450,850])
        self.scatter.picked.connect(self.select);self.list.currentItemChanged.connect(self.from_list)
    def pause(self,*args):
        self.play_timer.stop();self.play.setText('재생')

    def toggle_play(self):
        if not self.trajectory.row:return
        if self.play_timer.isActive():self.pause();return
        if self.timeline.value()==self.timeline.maximum():self.timeline.setValue(0)
        self.play_start=self.timeline.value();self.elapsed.start();self.play_timer.start();self.play.setText('일시정지')

    def advance(self):
        speed=(.25,.5,1.,2.)[self.rate.currentIndex()]
        frame=self.play_start+int(self.elapsed.elapsed()/1000*speed/self.trajectory.row.get('dt',1/30))
        self.timeline.blockSignals(True);self.timeline.setValue(min(frame,self.timeline.maximum()));self.timeline.blockSignals(False)
        self.show_frame(self.timeline.value())
        if frame>=self.timeline.maximum():self.pause()

    def seek(self,frame):
        self.pause();self.show_frame(frame)

    def show_frame(self,frame):
        self.trajectory.set_frame(frame)
        dt=self.trajectory.row.get('dt',1/30)
        self.actions.frame=frame;self.actions.update()
        self.clock_label.setText(f'{frame*dt:.2f}s / {self.timeline.maximum()*dt:.2f}s · {frame+1} frame')

    def error(self,exc):QMessageBox.warning(self,'Curation',str(exc))
    def run_job(self,fn,done):
        if self.job is not None and self.job.isRunning():
            self.error('현재 분석/저장이 끝난 뒤 실행하세요.');return
        self.job=Job(fn,self)
        self.job.result.connect(done);self.job.failed.connect(self.error)
        self.job.start()
    def reject(self):
        self.pause()
        if self.job is not None and self.job.isRunning():
            self.error('저장/분석 완료 후 닫아주세요.');return
        super().reject()
    def closeEvent(self,event):
        if self.job is not None and self.job.isRunning():event.ignore()
        else:super().closeEvent(event)
    def open_file(self):
        path,_=QFileDialog.getOpenFileName(self,'Source / Datagen HDF5', '', 'HDF5 (*.hdf5 *.h5)')
        if not path:return
        self.info.setText('분석 중… 이미지 전체를 읽지 않고 trajectory를 분석합니다.')
        self.run_job(lambda: analyze(path),lambda rows:self.loaded(path,rows))
    def open_folder(self):
        path=QFileDialog.getExistingDirectory(self,'Source / Datagen 데이터 폴더')
        if path:self.run_job(lambda:analyze(path),lambda rows:self.loaded(path,rows))
    def loaded(self,path,rows):
        try:
            self.rows=rows;self.path=path;self.excluded=set()
            known=next((a for a in self.registry.all() if a['path']==path),None)
            if known:self.kind=known['type']
            elif Path(path).is_dir():
                manifest=json.loads((Path(path)/'manifest.json').read_text())
                self.kind='annotated' if all((Path(path)/e['path']/'annotation.json').is_file() for e in manifest['episodes']) else 'source'
            else:
                import h5py
                with h5py.File(path) as f:
                    g=f['data'][self.rows[0]['name']]
                    self.kind='annotated' if 'obs/datagen_info/subtask_term_signals/grasp' in g else 'source'
            if self.kind not in ('source','annotated','datagen'):raise ValueError('Source / Annotation / Datagen 데이터만 지원합니다.')
            self.count.setMaximum(len(self.rows));self.count.setValue(len(self.rows))
            self.scatter.rows=self.rows;self.refresh_list();self.select(0)
        except Exception as exc:self.error(exc)
    def refresh_list(self):
        self.list.blockSignals(True);self.list.clear()
        for i,r in enumerate(self.rows):
            if self.only_flags.isChecked() and not r['flags']:continue
            text=('제외 · ' if r['name'] in self.excluded else '')+r['name']+' · '+r['task_id']+' · '+(', '.join(r['flags']) or '특이 지표 없음')
            item=QListWidgetItem(text);item.setData(Qt.ItemDataRole.UserRole,i);self.list.addItem(item)
        self.list.blockSignals(False);self.scatter.excluded=self.excluded;self.scatter.update()
        self.info.setText(f'포함 {len(self.rows)-len(self.excluded)} / 전체 {len(self.rows)} · 확인 필요 {sum(bool(r["flags"]) for r in self.rows)} · 원본 보존 / 회색=제외 · 지표는 불량 확정이 아닙니다.')
    def from_list(self,current,previous):
        if current:self.select(current.data(Qt.ItemDataRole.UserRole))
    def select(self,i):
        self.pause()
        r=self.rows[i];self.scatter.selected=i;self.scatter.update()
        self.trajectory.robot=None;self.trajectory.mesh=None;self.trajectory.frame=0
        path=r.get('robot_urdf')
        if path and r.get('robot_q') is not None:
            try:
                from .robot_preview import RobotPreview
                if path not in self.robot_cache:self.robot_cache[path]=RobotPreview(path)
                model=self.robot_cache[path]
                q=np.asarray(r['robot_q'],dtype=float)
                if q.shape!=(len(r['ee']),6) or not np.isfinite(q).all():raise ValueError('관절 상태 길이/값 불일치')
                # Link extents across the whole episode keep framing steady during playback.
                lows=[];highs=[]
                for state in q:
                    poses=model.transforms(state)
                    pts=np.array([t[:3,3] for t in poses.values()])
                    lows.append(pts.min(0)-.07);highs.append(pts.max(0)+.07)
                self.trajectory.robot_bounds=(np.min(lows,axis=0),np.max(highs,axis=0))
                self.trajectory.robot=model
            except Exception as exc:self.error(f'URDF 표시 실패: {exc}')
        for p in (self.trajectory,self.actions):p.row=r;p.update()
        self.timeline.blockSignals(True);self.timeline.setRange(0,len(r['ee'])-1);self.timeline.setValue(0);self.timeline.blockSignals(False)
        self.show_frame(0)
        self.metrics.setText(f"{r['name']} · {r['task_id']}\n시간 {r['duration']:.2f}s · EE 정지 비율 {100*r['stop_fraction']:.1f}% (<2mm/s)\n정지 명령 비율 {100*r.get('command_stop_fraction',0):.1f}% (<1deg/s)\nArm 명령 가속도 RMS {r['jitter']:.2f} rad/s² · gripper {r['gripper_jitter']:.2f} rad/s²\n"+(' / '.join(r['flags']) or '통계적 이상 없음'))
    def toggle(self):
        if not self.rows:return
        name=self.rows[self.scatter.selected]['name']
        if name in self.excluded:self.excluded.remove(name)
        else:self.excluded.add(name)
        self.refresh_list()
    def restore_all(self):
        self.excluded.clear();self.refresh_list()

    def auto_select(self):
        try:
            eligible=[r for r in self.rows if r['name'] not in self.excluded]
            chosen=set(select_distribution(eligible,self.count.value()))
            self.excluded={r['name'] for r in self.rows if r['name'] not in chosen};self.refresh_list()
        except Exception as exc:self.error(exc)
    def save(self):
        if not self.path:return
        canonical=Path(self.path).is_dir()
        dest,_=QFileDialog.getSaveFileName(self,'새 데이터셋 저장','curated' if canonical else 'curated.hdf5','폴더 이름 (*)' if canonical else 'HDF5 (*.hdf5)')
        if not dest:return
        names=[r['name'] for r in self.rows if r['name'] not in self.excluded]
        source,kind=self.path,self.kind
        self.info.setText('새 데이터 저장·검증·등록 중… 원본은 유지됩니다.')
        def work():
            output=export_subset(source,dest,names)
            parent=next((a for a in self.registry.all() if a['path']==source),None)
            self.registry.register(kind,output,parents=[parent['id']] if parent else [])
            return output
        self.run_job(work,lambda output:self.info.setText(f'저장·등록 완료: {output} · {len(names)} episodes'))
