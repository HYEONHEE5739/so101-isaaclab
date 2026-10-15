"""Dependency-free Qt plots for dataset inspection; no simulator/hardware launch."""
import json
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QThread, QTimer, QElapsedTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QPolygonF, QImage, QPixmap
from PyQt6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QPushButton,QLabel,QListWidget,
    QListWidgetItem,QSpinBox,QDoubleSpinBox,QFileDialog,QMessageBox,QSplitter,QCheckBox,QWidget,QSlider,QComboBox,QFormLayout)
from .curation import analyze, load_episode, EpisodeReader, select_distribution, export_subset


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
        self.robot=None;self.frame=0;self.mesh=None;self.payload={};self.sample_frame=0
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
        for poses in self.row.get('object_paths',{}).values():
            if poses is not None:
                points=poses[:,:3,3];lo=np.minimum(lo,points.min(0)-.03);hi=np.maximum(hi,points.max(0)+.03)
        for pose in self.payload.get('initial_object_poses',{}).values():
            if pose is not None:lo=np.minimum(lo,pose[:3,3]-.03);hi=np.maximum(hi,pose[:3,3]+.03)
        pad=np.maximum((hi-lo)*.12,.01)
        return lo-pad,hi+pad

    def project(self, points):
        p=np.asarray(points,dtype=float)
        if self.mode=='trajectory':
            lo,hi=self.bounds();p=p-(lo+hi)/2
            a,b=self.yaw,self.pitch
            p=p@np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
            p=p@np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
            # Equal metric scale; fit the projected box rather than its 3D diagonal.
            a,b=self.yaw,self.pitch
            rot=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])@np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
            span=(hi-lo)@np.abs(rot)
            scale=min(max(1,self.width()-90)/max(span[0],1e-6),max(1,self.height()-80)/max(span[1],1e-6))
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
    def set_frame(self,frame,payload=None):
        self.frame=frame
        self.payload=payload or {}
        samples=self.row.get('sample_indices',np.arange(len(self.row['ee'])))
        self.sample_frame=int(np.clip(np.searchsorted(samples,frame),0,len(samples)-1))
        q=self.payload.get('robot_q')
        if q is None and self.row.get('robot_q') is not None:q=self.row['robot_q'][self.sample_frame]
        self.mesh=self.robot.triangles(q) if self.robot is not None and q is not None else None
        self.update()

    def draw_objects(self,painter):
        from .robot_preview import object_triangles
        for name,poses in self.row.get('object_paths',{}).items():
            spec=self.row.get('object_specs',{}).get(name,{})
            rgb=spec.get('appearance',{}).get('color',[.6,.65,.7]);color=QColor(*[int(np.clip(v,0,1)*255) for v in rgb])
            target=name==self.row.get('pick_object_id');painter.setPen(QPen(color.lighter(160),2 if target else 1,Qt.PenStyle.SolidLine if target else Qt.PenStyle.DashLine))
            if poses is not None:
                points=self.project(poses[:,:3,3])
                for a,b in zip(points,points[1:]):painter.drawLine(a,b)
            pose=self.payload.get('object_poses',{}).get(name)
            if pose is None:
                initial=self.payload.get('initial_object_poses',{}).get(name)
                if initial is not None:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    point=self.project([initial[:3,3]])[0];painter.drawEllipse(point,5,5);painter.drawText(point,name+' initial only')
                continue
            if spec.get('dimensions_m') and spec.get('shape'):
                triangles=object_triangles(spec,pose);fill=QColor(color);fill.setAlpha(100);painter.setBrush(fill)
                vertices=self.project(triangles.reshape(-1,3))
                for i in range(len(triangles)):painter.drawPolygon(QPolygonF(vertices[3*i:3*i+3]))
            point=self.project([pose[:3,3]])[0];painter.setBrush(color);painter.drawEllipse(point,4,4)
            painter.setPen(QColor('#ffffff' if target else '#b5c5d3'))
            painter.drawText(point,name+(' [Target]' if target else ' [Distractor]'))

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
        title={'scatter':'초기 목표 물체 · Workspace XY [mm]','trajectory':'Robot / EE / Objects · '+str((self.row or {}).get('display_frame',''))+' [mm]','actions':'관절 목표값 [rad] · 가로축: 시간 [s]'}[self.mode]
        painter.drawText(12,22,title)
        if self.mode=='scatter' and self.rows:
            self.screen=self.project([r['xy'] for r in self.rows])
            for i,(row,point) in enumerate(zip(self.rows,self.screen)):
                selected=getattr(self,'original_indices',list(range(len(self.rows))))[i]==self.selected
                color='#66717d' if row['name'] in self.excluded else '#efb467' if row['flags'] else '#55d6b0'
                painter.setPen(QPen(QColor('#ffffff' if selected else color),2))
                painter.setBrush(QColor(color));painter.drawEllipse(point,6 if selected else 4,6 if selected else 4)
            xy=np.asarray([r['xy'] for r in self.rows])*1000
            painter.setPen(QColor('#b5c5d3'));painter.drawText(12,self.height()-10,f"X {xy[:,0].min():.1f} … {xy[:,0].max():.1f} / Y {xy[:,1].min():.1f} … {xy[:,1].max():.1f} mm")
        elif self.row and self.mode=='trajectory':
            self.draw_axes(painter)
            self.draw_robot(painter)
            self.draw_objects(painter)
            points=self.project(self.row['ee']);painter.setPen(QPen(QColor('#55d6b0'),2))
            for i,(a,b) in enumerate(zip(points,points[1:])):
                painter.setPen(QPen(QColor('#55d6b0' if i<self.sample_frame else '#4b6870'),2));painter.drawLine(a,b)
            current=self.project([self.payload['ee']])[0] if 'ee' in self.payload else points[self.sample_frame]
            painter.setPen(QColor('#ffffff'));painter.setBrush(QColor('#ffffff'));painter.drawEllipse(current,6,6)
            for point,label,color in [(points[0],'START','#55d6b0'),(points[-1],'END','#efb467')]:
                painter.setPen(QColor(color));painter.drawEllipse(point,5,5);painter.drawText(point,label)
            xyz=np.asarray(self.row['ee']);painter.drawText(12,self.height()-10,'XYZ span [mm]: '+str(np.round(np.ptp(xyz,axis=0)*1000,1)))
        elif self.row and self.mode=='actions':
            q=np.asarray(self.row['actions']);low=q.min();span=max(q.max()-low,1e-6)
            indices=self.row.get('sample_indices',np.arange(len(q)));total=max(self.row.get('frames',len(q))-1,1)
            times=self.row.get('timestamps')
            def phase(i):return (times[i]-times[0])/max(times[-1]-times[0],1e-9) if times is not None else i/total
            colors=['#55d6b0','#69b9ff','#efb467','#d196fa','#fb9292','#e5e7eb']
            for j,color in enumerate(colors):
                painter.setPen(QPen(QColor(color),1))
                points=[QPointF(35+phase(indices[i])*(self.width()-55),self.height()-35-(v-low)/span*(self.height()-75)) for i,v in enumerate(q[:,j])]
                for a,b in zip(points,points[1:]):painter.drawLine(a,b)
                painter.drawText(15+j*65,40,str(j+1))
            painter.setPen(QPen(QColor('#ffffff'),1))
            x=35+phase(self.frame)*(self.width()-55);painter.drawLine(QPointF(x,45),QPointF(x,self.height()-35))
            for label,index in self.row.get('events',{}).items():
                x=35+phase(index)*(self.width()-55);painter.setPen(QPen(QColor('#efb467'),1,Qt.PenStyle.DashLine));painter.drawLine(QPointF(x,45),QPointF(x,self.height()-35));painter.drawText(QPointF(x,55),label)
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
            if min(d)<225:
                index=int(np.argmin(d));self.picked.emit(getattr(self,'original_indices',list(range(len(d))))[index])
    def mouseMoveEvent(self,event):
        if self.drag is not None and self.mode=='trajectory':
            delta=event.position()-self.drag;self.yaw+=delta.x()*.01;self.pitch+=delta.y()*.01
            self.drag=event.position();self.update()
    def mouseReleaseEvent(self,event):self.drag=None


class CurationDialog(QDialog):
    def __init__(self,registry,parent=None):
        super().__init__(parent);self.registry=registry;self.rows=[];self.excluded=set();self.path=None;self.kind='datagen';self.job=None;self.split_plan=None
        self.reader=None;self.frame_job=None;self.pending_frame=None;self.generation=0;self.seek_generation=0;self.pending_selection=None
        self.setWindowTitle('Data Curation · 분포와 동작 검토');self.resize(1400,900)
        layout=QVBoxLayout(self);bar=QHBoxLayout();layout.addLayout(bar)
        load=QPushButton('데이터 열기');load.clicked.connect(self.open_file);bar.addWidget(load)
        folder=QPushButton('데이터 폴더 열기');folder.clicked.connect(self.open_folder);bar.addWidget(folder)
        self.count=QSpinBox();self.count.setRange(1,100000);bar.addWidget(QLabel('최종 개수'));bar.addWidget(self.count)
        choose=QPushButton('분포 유지 자동 선별');choose.clicked.connect(self.auto_select);bar.addWidget(choose)
        restore=QPushButton('전체 복원');restore.clicked.connect(self.restore_all);bar.addWidget(restore)
        export=QPushButton('선택 데이터셋 저장');export.clicked.connect(self.save);bar.addWidget(export)
        sourcebar=QHBoxLayout();layout.addLayout(sourcebar)
        self.train_ratio=QDoubleSpinBox();self.train_ratio.setRange(1,99);self.train_ratio.setValue(80);self.train_ratio.setSuffix('%')
        self.split_seed=QSpinBox();self.split_seed.setRange(0,2147483647)
        sourcebar.addWidget(QLabel('Source Train'));sourcebar.addWidget(self.train_ratio);sourcebar.addWidget(QLabel('Seed'));sourcebar.addWidget(self.split_seed)
        self.plan_split=QPushButton('Source 분할 미리보기');self.plan_split.clicked.connect(self.preview_source_split);sourcebar.addWidget(self.plan_split)
        self.source_assignment=QComboBox();self.source_assignment.addItem('Train','train');self.source_assignment.addItem('Validation','val');sourcebar.addWidget(self.source_assignment)
        self.assign_split=QPushButton('선택 Source 소속 변경');self.assign_split.clicked.connect(self.assign_source_split);sourcebar.addWidget(self.assign_split)
        self.export_split=QPushButton('Train / Validation 저장');self.export_split.clicked.connect(self.save_source_split);sourcebar.addWidget(self.export_split)
        self.info=QLabel('원본은 보존됩니다. 회색=제외 / 주황=확인 필요 / 초록=포함. 점 또는 목록을 선택하세요.');self.info.setWordWrap(True);layout.addWidget(self.info)
        split=QSplitter();layout.addWidget(split,1)
        left=QWidget();ll=QVBoxLayout(left);split.addWidget(left)
        self.scatter=Plot('scatter');ll.addWidget(self.scatter,1)
        self.only_flags=QCheckBox('확인 필요 에피소드만 표시');self.only_flags.toggled.connect(self.refresh_list);ll.addWidget(self.only_flags)
        self.filters={};form=QFormLayout();ll.addLayout(form)
        for key,label in [('task_id','Task'),('active_object_count','Active count'),('active_object_ids','Active IDs'),('pick_object_id','Target'),
                          ('scene_configuration','Scene hash'),('source_group','Source group'),('split','Train/Validation'),('recorded_success','Recorded success'),('annotation_status','Annotation')]:
            combo=QComboBox();combo.addItem('전체',None);combo.currentIndexChanged.connect(self.refresh_list)
            self.filters[key]=combo;form.addRow(label,combo)
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
        self.actions=Plot('actions');self.actions.setMinimumHeight(145);rl.addWidget(self.actions,1)
        rgbbar=QHBoxLayout();rgbbar.addWidget(QLabel('RGB'));self.camera=QComboBox();self.camera.currentIndexChanged.connect(lambda:self.seek(self.timeline.value()))
        rgbbar.addWidget(self.camera);self.event_label=QLabel('Annotation unavailable');self.event_label.setWordWrap(True);rgbbar.addWidget(self.event_label,1);rl.addLayout(rgbbar)
        self.rgb=QLabel('RGB unavailable');self.rgb.setAlignment(Qt.AlignmentFlag.AlignCenter);self.rgb.setMinimumSize(240,160);self.rgb.setMaximumHeight(210);rl.addWidget(self.rgb)
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
        times=self.trajectory.row.get('timestamps')
        frame=int(np.searchsorted(times,times[self.play_start]+self.elapsed.elapsed()/1000*speed,side='right')-1) if times is not None else self.play_start+int(self.elapsed.elapsed()/1000*speed/self.trajectory.row.get('dt',1/30))
        self.timeline.blockSignals(True);self.timeline.setValue(min(frame,self.timeline.maximum()));self.timeline.blockSignals(False)
        self.show_frame(self.timeline.value())
        if frame>=self.timeline.maximum():self.pause()

    def seek(self,frame):
        self.pause();self.seek_generation+=1;self.show_frame(frame)

    def show_frame(self,frame):
        if self.trajectory.row is None:return
        if self.reader is not None:
            self.pending_frame=(self.generation,frame,self.camera.currentData())
            self.start_frame_job();return
        self.apply_frame({'index':frame})

    def start_frame_job(self):
        if self.frame_job is not None or self.pending_frame is None:return
        generation,frame,camera=self.pending_frame;self.pending_frame=None;reader=self.reader;seek_generation=self.seek_generation
        if reader is None:return
        self.frame_job=Job(lambda:reader.frame(frame,camera),self)
        self.frame_job.result.connect(lambda payload:self.apply_frame(payload) if generation==self.generation and seek_generation==self.seek_generation else None)
        self.frame_job.failed.connect(lambda error:self.info.setText(error) if generation==self.generation else None)
        def finished():
            self.frame_job=None;self.start_frame_job()
        self.frame_job.finished.connect(finished);self.frame_job.finished.connect(self.frame_job.deleteLater);self.frame_job.start()

    def apply_frame(self,payload):
        frame=payload['index'];self.trajectory.set_frame(frame,payload)
        self.timeline.blockSignals(True);self.timeline.setValue(frame);self.timeline.blockSignals(False)
        dt=self.trajectory.row.get('dt',1/30)
        self.actions.frame=frame;self.actions.update()
        times=self.trajectory.row.get('timestamps');start=times[0] if times is not None else 0
        stamp=payload.get('timestamp',frame*dt)
        duration=times[-1]-start if times is not None else self.timeline.maximum()*dt
        self.clock_label.setText(f'{stamp-start:.2f}s / {duration:.2f}s · {frame+1} frame')
        self.event_label.setText(' · '.join(f'{n}: frame {i+1}'+(' ✓' if frame>=i else '') for n,i in self.trajectory.row.get('events',{}).items()) or 'Grasp/Place annotation unavailable')
        if payload.get('rgb') is not None:
            rgb=payload['rgb'];h,w,_=rgb.shape;picture=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format.Format_RGB888).copy()
            self.rgb.setPixmap(QPixmap.fromImage(picture).scaled(self.rgb.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        elif self.reader is not None:self.rgb.clear();self.rgb.setText('RGB unavailable')

    def error(self,exc):QMessageBox.warning(self,'Curation',str(exc))
    def run_job(self,fn,done):
        if self.job is not None and self.job.isRunning():
            self.error('현재 분석/저장이 끝난 뒤 실행하세요.');return
        self.job=Job(fn,self);result=[];errors=[]
        self.job.result.connect(result.append);self.job.failed.connect(errors.append)
        def finish():
            self.job=None
            if errors:self.error(errors[0])
            elif result:done(result[0])
            if self.pending_selection is not None:
                index=self.pending_selection;self.pending_selection=None;self.select(index)
        self.job.finished.connect(finish);self.job.finished.connect(self.job.deleteLater)
        self.job.start()
    def reject(self):
        self.pause();self.pending_frame=None
        if any(j is not None and j.isRunning() for j in (self.job,self.frame_job)):
            self.error('저장/분석 완료 후 닫아주세요.');return
        super().reject()
    def closeEvent(self,event):
        self.pause();self.pending_frame=None
        if any(j is not None and j.isRunning() for j in (self.job,self.frame_job)):event.ignore()
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
            self.rows=rows;self.path=path;self.excluded=set();self.split_plan=None
            locked=any(r.get('source_partition') for r in rows)
            for control in (self.plan_split,self.assign_split,self.export_split):control.setEnabled(not locked)
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
            for key,combo in self.filters.items():
                combo.blockSignals(True);combo.clear();combo.addItem('전체',None)
                for value in sorted({self.filter_value(r,key) for r in rows}):combo.addItem(value,value)
                combo.blockSignals(False)
            self.scatter.rows=self.rows;self.refresh_list();self.select(0)
        except Exception as exc:self.error(exc)
    def refresh_list(self):
        self.list.blockSignals(True);self.list.clear()
        visible=[]
        for i,r in enumerate(self.rows):
            if self.only_flags.isChecked() and not r['flags']:continue
            if any(c.currentData() is not None and self.filter_value(r,k)!=c.currentData() for k,c in self.filters.items()):continue
            visible.append(i)
            text=('제외 · ' if r['name'] in self.excluded else '')+r['name']+' · '+(r.get('split') or '미분할')+' · '+r['task_id']+' · '+(', '.join(r['flags']) or '특이 지표 없음')
            item=QListWidgetItem(text);item.setData(Qt.ItemDataRole.UserRole,i);self.list.addItem(item)
        self.list.blockSignals(False);points=[i for i in visible if self.rows[i].get('xy') is not None]
        self.scatter.rows=[self.rows[i] for i in points];self.scatter.original_indices=points
        self.scatter.excluded=self.excluded;self.scatter.update()
        self.info.setText(f'포함 {len(self.rows)-len(self.excluded)} / 전체 {len(self.rows)} · 확인 필요 {sum(bool(r["flags"]) for r in self.rows)} · 원본 보존 / 회색=제외 · 지표는 불량 확정이 아닙니다.')
        if self.split_plan or any(r.get('source_partition') for r in self.rows):
            counts={s:sum(r.get('split')==s and r['name'] not in self.excluded for r in self.rows) for s in ('train','val')}
            pools={s:{key for r in self.rows if r.get('source_partition') and r['source_partition']['split']==s for key in r['source_partition']['source_ids']} for s in ('train','val')}
            sources={s:len(pools[s]) for s in pools} if not self.split_plan else counts
            self.info.setText(self.info.text()+f'\nTrain Source {sources["train"]} / episodes {counts["train"]} · Validation Source {sources["val"]} / episodes {counts["val"]} · '+('분할 계획: 교집합 없음' if self.split_plan else '기록 Partition 유지 · 계보 교집합은 학습 전에 검사'))

    @staticmethod
    def filter_value(row,key):
        value=row.get(key)
        if key=='scene_configuration':value=(value or {}).get('sha256')
        if isinstance(value,(list,tuple)):return ', '.join(value)
        return 'UNKNOWN' if value is None else str(value)
    def from_list(self,current,previous):
        if current:self.select(current.data(Qt.ItemDataRole.UserRole))
    def select(self,i):
        if not 0<=i<len(self.rows):return
        self.pause();self.generation+=1;generation=self.generation
        self.scatter.selected=i;self.scatter.update();self.pending_frame=None;self.reader=None
        self.trajectory.row=None;self.rgb.clear();self.rgb.setText('Episode loading…')
        row=self.rows[i]
        if 'data_ref' not in row:
            self.install_episode(row,None,None);return
        if self.job is not None and self.job.isRunning():
            self.pending_selection=i;return
        self.metrics.setText(row['name']+' · telemetry/URDF loading…')
        def prepare():
            loaded=load_episode(row);model=None;bounds=None;path=loaded.get('robot_urdf')
            if path and loaded.get('robot_q') is not None and loaded['display_frame']=='robot_base':
                from .robot_preview import RobotPreview
                if path not in self.robot_cache:self.robot_cache[path]=RobotPreview(path)
                model=self.robot_cache[path];q=np.asarray(loaded['robot_q'],dtype=float)
                if q.shape!=(len(loaded['ee']),6) or not np.isfinite(q).all():raise ValueError('Recorded joint state alignment mismatch')
                points=np.array([t[:3,3] for state in q[::max(1,len(q)//120)] for t in model.transforms(state).values()])
                bounds=(points.min(0)-.07,points.max(0)+.07)
            return loaded,model,bounds
        self.run_job(prepare,lambda result:self.install_episode(*result) if generation==self.generation else None)

    def install_episode(self,row,model,bounds):
        self.trajectory.robot=model;self.trajectory.mesh=None;self.trajectory.frame=0
        if bounds is not None:self.trajectory.robot_bounds=bounds
        self.reader=EpisodeReader(row) if 'data_ref' in row else None
        for plot in (self.trajectory,self.actions):plot.row=row;plot.update()
        self.camera.blockSignals(True);self.camera.clear()
        for name in row.get('cameras',{}):self.camera.addItem(name,name)
        self.camera.blockSignals(False)
        self.timeline.blockSignals(True);self.timeline.setRange(0,row.get('frames',len(row['ee']))-1);self.timeline.setValue(0);self.timeline.blockSignals(False)
        self.show_frame(0)
        def mm(value):return 'UNKNOWN' if value is None else f'{value*1000:.1f}mm'
        missing=[n for n,v in row.get('object_paths',{}).items() if v is None]
        self.metrics.setText(f"{row['name']} · {row['task_id']} · Target {row.get('pick_object_id','UNKNOWN')}\n"
            f"Active {row.get('active_object_count','UNKNOWN')}: {', '.join(row.get('active_object_ids') or [])} · Group {row.get('source_group','UNKNOWN')}\n"
            f"Scene {((row.get('scene_configuration') or {}).get('sha256') or 'UNKNOWN')[:16]} · Recorded success {self.filter_value(row,'recorded_success')} / Physical {self.filter_value(row,'physical_success')} ({row.get('success_source','unknown')})\n"
            f"시간 {row['duration']:.2f}s · EE 정지 {100*row['stop_fraction']:.1f}% · 정지 명령 {100*row.get('command_stop_fraction',0):.1f}% · Jitter {row['jitter']:.2f} / {row['gripper_jitter']:.2f}rad/s²\n"
            f"Target 이동 {mm(row.get('target_movement_m'))} · Lift {mm(row.get('lift_height_m'))} · EE 최소거리 {mm(row.get('target_ee_distance_m',{}).get('min'))}\n"
            f"Grasp 전/후 이동 {mm(row.get('movement_before_grasp_m'))} / {mm(row.get('movement_after_grasp_m'))} · Annotation {row.get('annotation_status','UNAVAILABLE')}\n"
            +('Trajectory unavailable: '+', '.join(missing) if missing else '')
            +(' · 기하/모델 근거 없음: pose marker만 표시' if not row.get('object_specs') else '')
            +'\n객체별 최소 접근: '+', '.join(n+': '+mm(v['min_distance_m']) for n,v in row.get('object_approach',{}).items())
            +' · 접근 거리는 grasp 성공 판정이 아닙니다.')

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

    def preview_source_split(self):
        if not self.path:return
        from .episodes import source_split
        names=[r['name'] for r in self.rows if r['name'] not in self.excluded]
        path,ratio,seed=self.path,self.train_ratio.value()/100,self.split_seed.value()
        self.run_job(lambda:source_split(path,ratio,seed,names=names),self.install_source_split)

    def install_source_split(self,plan):
        self.split_plan=plan
        for row in self.rows:row['split']=plan['assignments'].get(row['name'])
        combo=self.filters['split'];combo.blockSignals(True);combo.clear();combo.addItem('전체',None)
        for value in ('train','val'):combo.addItem(value,value)
        combo.blockSignals(False)
        self.refresh_list()

    def assign_source_split(self):
        if not self.split_plan:return self.error('먼저 Source 분할 미리보기를 실행하세요.')
        from .episodes import source_split
        assignments=dict(self.split_plan['assignments']);name=self.rows[self.scatter.selected]['name']
        if name not in assignments:return self.error('포함된 Source를 선택하세요.')
        for member in self.split_plan['group_members'][name]:assignments[member]=self.source_assignment.currentData()
        path,ratio,seed=self.path,self.train_ratio.value()/100,self.split_seed.value()
        self.run_job(lambda:source_split(path,ratio,seed,assignments=assignments,names=list(assignments)),self.install_source_split)

    def save_source_split(self):
        if not self.split_plan:return self.error('Source 분할을 먼저 확인하세요.')
        dest,_=QFileDialog.getSaveFileName(self,'새 Train/Validation 출력 폴더','source_split','폴더 이름 (*)')
        if not dest:return
        from .episodes import export_source_splits
        source,kind=self.path,self.kind;names=[r['name'] for r in self.rows if r['name'] not in self.excluded]
        assignments={n:self.split_plan['assignments'][n] for n in names}
        ratio,seed=self.train_ratio.value()/100,self.split_seed.value()
        def work():
            result=export_source_splits(source,dest,ratio,seed,assignments,names)
            parent=next((a for a in self.registry.all() if a['path']==source),None)
            for output in result['outputs'].values():self.registry.register(kind,output,parents=[parent['id']] if parent else [])
            return result
        self.run_job(work,lambda result:self.info.setText(f'Train {result["counts"]["train"]} / Validation {result["counts"]["val"]} 저장·등록 완료 · 원본 불변 · {dest}'))

    def save(self):
        if not self.path:return
        if self.split_plan:return self.error('분할 계획은 Train / Validation 저장으로 저장하세요.')
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
