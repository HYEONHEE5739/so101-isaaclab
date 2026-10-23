"""Dependency-free Qt plots for dataset inspection; no simulator/hardware launch."""
import json
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, QEvent, pyqtSignal, QThread, QTimer, QElapsedTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QPolygonF, QImage, QPixmap
from PyQt6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QPushButton,QLabel,QListWidget,
    QListWidgetItem,QSpinBox,QDoubleSpinBox,QFileDialog,QMessageBox,QSplitter,QCheckBox,QWidget,QSlider,QComboBox,QFormLayout,QTabWidget,QScrollArea,QStyle,QStyleOptionSlider,QToolTip,QLineEdit)
from .curation import analyze, load_episode, select_distribution, export_subset
from ..data.episodes import EpisodeReader
from .view import CameraView

SPLIT_COLORS={'train':'#55d6b0','val':'#69b9ff'}


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
        self.robot=None;self.frame=0;self.mesh=None;self.payload={};self.sample_frame=0;self.context_rows=[]
        self.action_lines=[];self.action_row=None;self.setMouseTracking(mode=='actions')
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
        if self.mesh is not None:
            vertices=self.mesh[0].reshape(-1,3);lo=np.minimum(lo,vertices.min(0));hi=np.maximum(hi,vertices.max(0))
        pad=np.maximum((hi-lo)*.04,.005)
        return lo-pad,hi+pad

    def project(self, points):
        p=np.asarray(points,dtype=float)
        if not p.size:return []
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
            basis,center,span=self.scatter_geometry()
            p=p[:,:2]@basis
            scale=min((self.width()-80)/span[0],(self.height()-110)/span[1])
            p=(p-center)*scale
        return [QPointF(x+self.width()/2,self.height()/2-y) for x,y in p]

    def scatter_geometry(self):
        rows=self.context_rows or self.rows
        regions=[r['placement_region'] for r in rows if r.get('placement_region')]
        points=[r['xy'] for r in rows if r.get('xy') is not None]
        for region in regions:points.extend(region['support']);points.append(region['robot_xy'])
        basis=np.eye(2)
        if regions:
            region=regions[0];delta=np.mean(region['support'],axis=0)-region['robot_xy']
            if np.linalg.norm(delta)>1e-8:
                # Keep support edges straight: face the nearest workspace cardinal direction.
                forward=np.zeros(2);axis=int(np.argmax(np.abs(delta)));forward[axis]=np.sign(delta[axis])
                basis=np.column_stack(([forward[1],-forward[0]],forward))
        projected=np.asarray(points or [[0,0]])@basis;lo=projected.min(0);hi=projected.max(0)
        return basis,(lo+hi)/2,np.maximum(hi-lo,.02)*1.08

    def draw_distribution(self,painter):
        regions={json.dumps(r['placement_region'],sort_keys=True):r['placement_region'] for r in (self.context_rows or self.rows) if r.get('placement_region')}
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for region in regions.values():
            for key,color,style in [('support','#586574',Qt.PenStyle.DashLine),('allowed','#93b8d8',Qt.PenStyle.SolidLine)]:
                painter.setPen(QPen(QColor(color),1.5,style));painter.drawPolygon(QPolygonF(self.project(region[key])))
        robots={tuple(region['robot_xy']) for region in regions.values()}
        for xy in robots:
            point=self.project([xy])[0]
            painter.setPen(QPen(QColor('#d7b3f4'),1.5));painter.setBrush(QColor('#a367db'))
            painter.drawRect(QRectF(point.x()-6,point.y()-6,12,12))
            painter.drawText(point+QPointF(10,4),'로봇')
        painter.setPen(QColor('#b5c5d3'))
        if len(regions)==1:
            region=next(iter(regions.values()));size=np.ptp(region['allowed'],axis=0)*1000;support=np.ptp(region['support'],axis=0)*1000
            mode='균일 XY' if region['mode']=='uniform' else '네 anchor 주변' if region['mode']=='four_anchors' else region['mode']
            painter.drawText(12,43,f'지지면 {support[0]:.0f}×{support[1]:.0f} · 중심 배치 {size[0]:.0f}×{size[1]:.0f} mm · {mode}')
        painter.drawText(12,self.height()-32,'실선: 중심 배치 · 점선: 지지면 · 보라 네모: 로봇' if regions else '배치 경계 정보 없음 · 기록된 위치만 표시')
        legend='초록 성공 · 빨강 실패 · 회색 제외 · 흰 테두리 선택' if any(r.get('evaluation_outcome') for r in (self.context_rows or self.rows)) else '초록 Train · 파랑 Val · 회색 제외 · 주황 테두리 확인' if self.has_splits() else '초록 포함 · 주황 확인 · 회색 제외 · 흰 테두리 선택'
        painter.drawText(12,self.height()-12,legend)

    def has_splits(self):
        return any(r.get('split') in SPLIT_COLORS for r in (self.context_rows or self.rows))

    def point_color(self,row,split_mode):
        if row['name'] in self.excluded:return '#66717d'
        if row.get('evaluation_outcome'):return '#55d6b0' if row['recorded_success'] else '#ff7777'
        if split_mode:return SPLIT_COLORS.get(row.get('split'),'#b5c5d3')
        return '#efb467' if row['flags'] else '#55d6b0'

    def draw_axes(self,painter):
        lo,hi=self.bounds()
        painter.setPen(QPen(QColor('#344454'),1))
        for t in np.linspace(0,1,5):
            for axis in (0,1):
                a=lo.copy();b=lo.copy();a[axis]=b[axis]=lo[axis]+t*(hi[axis]-lo[axis])
                b[1-axis]=hi[1-axis]
                painter.drawLine(*self.project([a,b]))
        for axis,(name,color) in enumerate(zip('XYZ',('#ff8b8b','#73df9d','#75bfff'))):
            end=lo.copy();end[axis]=hi[axis]
            painter.setPen(QPen(QColor(color),2));painter.drawLine(*self.project([lo,end]))
            point=self.project([end])[0];painter.drawText(point+QPointF(5,-6),name)
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
            painter.drawText(point+QPointF(8,-12 if target else 16),name+(' · 목표' if target else ''))

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
        self.action_lines=[];self.action_row=None
        painter=QPainter(self);painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(),QColor('#141b23'));painter.setPen(QColor('#b5c5d3'))
        robot_view=any(r.get('placement_region') for r in (self.context_rows or self.rows))
        title={'scatter':'초기 목표 물체 분포 · '+('위에서 본 배치' if robot_view else 'Workspace XY'),'trajectory':'로봇 동작과 목표 물체 · '+str((self.row or {}).get('display_frame',''))+' [mm]','actions':'관절 목표값 [rad] · 가로축: 시간 [s]'}[self.mode]
        painter.drawText(12,22,title)
        if self.mode=='scatter' and (self.rows or self.context_rows):
            self.draw_distribution(painter)
            self.screen=self.project([r['xy'] for r in self.rows])
            split_mode=self.has_splits()
            for i,(row,point) in enumerate(zip(self.rows,self.screen)):
                selected=getattr(self,'original_indices',list(range(len(self.rows))))[i]==self.selected
                color=self.point_color(row,split_mode)
                edge='#efb467' if row['flags'] and row['name'] not in self.excluded else color
                painter.setPen(QPen(QColor(edge),2));painter.setBrush(QColor(color));painter.drawEllipse(point,5,5)
                if selected:
                    painter.setPen(QPen(QColor('#ffffff'),2));painter.setBrush(Qt.BrushStyle.NoBrush);painter.drawEllipse(point,8,8)
        elif self.row and self.mode=='trajectory':
            self.draw_axes(painter)
            self.draw_robot(painter)
            self.draw_objects(painter)
            points=self.project(self.row['ee']);painter.setPen(QPen(QColor('#55d6b0'),2))
            for i,(a,b) in enumerate(zip(points,points[1:])):
                painter.setPen(QPen(QColor('#55d6b0' if i<self.sample_frame else '#4b6870'),2));painter.drawLine(a,b)
            current=self.project([self.payload['ee']])[0] if 'ee' in self.payload else points[self.sample_frame]
            painter.setPen(QColor('#ffffff'));painter.setBrush(QColor('#ffffff'));painter.drawEllipse(current,6,6)
            for point,label,color in [(points[0],'시작','#55d6b0'),(points[-1],'끝','#efb467')]:
                painter.setPen(QColor(color));painter.drawEllipse(point,5,5);painter.drawText(point+QPointF(8,18),label)
        elif self.row and self.mode=='actions':
            q=np.asarray(self.row['actions']);low=q.min();span=max(q.max()-low,1e-6)
            indices=self.row.get('sample_indices',np.arange(len(q)));total=max(self.row.get('frames',len(q))-1,1)
            times=self.row.get('timestamps')
            def phase(i):return (times[i]-times[0])/max(times[-1]-times[0],1e-9) if times is not None else i/total
            colors=['#55d6b0','#69b9ff','#efb467','#d196fa','#fb9292','#e5e7eb']
            for j,color in enumerate(colors):
                painter.setPen(QPen(QColor(color),1))
                points=[QPointF(35+phase(indices[i])*(self.width()-55),self.height()-35-(v-low)/span*(self.height()-75)) for i,v in enumerate(q[:,j])]
                self.action_lines.append(np.array([[p.x(),p.y()] for p in points]))
                for a,b in zip(points,points[1:]):painter.drawLine(a,b)
                painter.drawText(15+j*65,40,str(j+1))
            painter.setPen(QPen(QColor('#ffffff'),1))
            x=35+phase(self.frame)*(self.width()-55);painter.drawLine(QPointF(x,45),QPointF(x,self.height()-35))
            for label,index in self.row.get('events',{}).items():
                x=35+phase(index)*(self.width()-55);painter.setPen(QPen(QColor('#efb467'),1,Qt.PenStyle.DashLine));painter.drawLine(QPointF(x,45),QPointF(x,self.height()-35));painter.drawText(QPointF(x,55),label)
            painter.drawText(12,self.height()-10,f"0 … {self.row['duration']:.2f}s / {low:.2f} … {q.max():.2f} rad (1–5 arm, 6 gripper)")
            self.action_row=self.row
        painter.end()

    def action_tooltip(self,position):
        if self.row is None or self.action_row is not self.row:return ''
        from ..so101_joint_mapping import JOINTS
        cursor=np.array([position.x(),position.y()]);hits=[]
        for joint,points in enumerate(self.action_lines):
            if len(points)==1:
                sample=0;distance=float(np.linalg.norm(cursor-points[0]))
            else:
                start=points[:-1];delta=np.diff(points,axis=0)
                fraction=np.clip(np.sum((cursor-start)*delta,axis=1)/np.maximum(np.sum(delta*delta,axis=1),1e-12),0,1)
                distances=np.linalg.norm(start+fraction[:,None]*delta-cursor,axis=1)
                segment=int(np.argmin(distances));distance=float(distances[segment])
                sample=segment+int(fraction[segment]>.5)
            if distance<=8:hits.append((distance,joint,sample))
        if not hits:return ''
        hits.sort();lines=['관절 목표값 · 가까운 기록']
        q=np.asarray(self.row['actions']);indices=self.row.get('sample_indices',np.arange(len(q)))
        times=self.row.get('timestamps')
        for distance,joint,sample in hits:
            if distance>hits[0][0]+1:continue
            frame=int(indices[sample]);value=float(q[sample,joint])
            elapsed=float(times[frame]-times[0]) if times is not None else frame*self.row.get('dt',self.row['duration']/max(self.row.get('frames',len(q))-1,1))
            lines.append(f'{joint+1}축 · {JOINTS[joint]}: {np.degrees(value):.2f}° ({value:.4f} rad)\n'
                         f'  {elapsed:.3f}s · {frame+1} frame')
        return '\n'.join(lines)

    def leaveEvent(self,event):
        if self.mode=='actions':QToolTip.hideText()
        super().leaveEvent(event)
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
        if self.mode=='actions':
            text=self.action_tooltip(event.position())
            if text:QToolTip.showText(event.globalPosition().toPoint(),text,self)
            else:QToolTip.hideText()
            return
        if self.drag is not None and self.mode=='trajectory':
            delta=event.position()-self.drag;self.yaw+=delta.x()*.01;self.pitch+=delta.y()*.01
            self.drag=event.position();self.update()
    def mouseReleaseEvent(self,event):self.drag=None


class DatasetPicker(QFileDialog):
    """One picker accepts either a canonical dataset directory or an HDF5 file."""
    def __init__(self,parent,directory,*,allow_evaluation=True,allow_collection=False):
        super().__init__(parent,'데이터 열기 · HDF5 파일 또는 데이터 폴더',directory)
        self.allow_evaluation = allow_evaluation
        self.allow_collection = allow_collection
        self.setOption(QFileDialog.Option.DontUseNativeDialog)
        self.setFileMode(QFileDialog.FileMode.AnyFile)
        self.setNameFilter('데이터 (*.hdf5 *.h5)')
        self.setLabelText(QFileDialog.DialogLabel.Accept,'열기')

    def accept(self):
        selected=self.selectedFiles()
        if not selected:return
        path=Path(selected[0])
        if self.allow_collection and path.is_dir():
            QDialog.accept(self);return
        from .paths import input_format
        fmt=input_format(path)
        if fmt in ('hdf5','canonical') or (self.allow_evaluation and fmt in ('evaluation','hdf5-collection')):
            QDialog.accept(self)
        elif path.is_dir():self.setDirectory(str(path))


class SquareViewport(QWidget):
    """Keep the trajectory canvas square as the preview area changes size."""
    def __init__(self,plot):
        super().__init__();self.plot=plot;plot.setParent(self);self.setMinimumSize(320,320)
    def resizeEvent(self,event):
        side=min(self.width(),self.height())
        self.plot.setGeometry((self.width()-side)//2,(self.height()-side)//2,side,side)


class EventTimeline(QSlider):
    """Recorded event markers share the slider's native frame-to-pixel mapping."""
    eventPicked=pyqtSignal(int)
    LABELS={'approach':'접근','grasp':'집기','transport':'이동','place':'놓기'}
    COLORS=('#efb467','#55d6b0','#69b9ff','#d196fa')

    def __init__(self):
        super().__init__(Qt.Orientation.Horizontal);self.markers=[];self.setMinimumHeight(38)

    def set_events(self,events,timestamps=None,dt=1/30):
        self.markers=[]
        for order,(name,frame) in enumerate(sorted(events.items(),key=lambda item:(item[1],item[0]))):
            stamp=float(timestamps[frame]-timestamps[0]) if timestamps is not None else frame*dt
            label=self.LABELS.get(name,name)
            self.markers.append(dict(name=name,frame=frame,time=stamp,label=label,color=self.COLORS[order%len(self.COLORS)]))
        self.update()

    def marker_position(self,frame):
        option=QStyleOptionSlider();self.initStyleOption(option)
        groove=self.style().subControlRect(QStyle.ComplexControl.CC_Slider,option,QStyle.SubControl.SC_SliderGroove,self)
        handle=self.style().subControlRect(QStyle.ComplexControl.CC_Slider,option,QStyle.SubControl.SC_SliderHandle,self)
        span=max(0,groove.width()-handle.width())
        offset=QStyle.sliderPositionFromValue(self.minimum(),self.maximum(),frame,span,option.upsideDown)
        return QPointF(groove.x()+handle.width()/2+offset,groove.center().y())

    def markers_at(self,point):
        return sorted((m for m in self.markers if abs(self.marker_position(m['frame']).x()-point.x())<=7
                       and abs(self.marker_position(m['frame']).y()-point.y())<=16),
                      key=lambda m:abs(self.marker_position(m['frame']).x()-point.x()))

    def paintEvent(self,event):
        super().paintEvent(event);painter=QPainter(self);painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for marker in self.markers:
            point=self.marker_position(marker['frame']);x,y=point.x(),point.y()
            painter.setPen(QPen(QColor(marker['color']),2));painter.drawLine(QPointF(x,y-6),QPointF(x,y+7))
            painter.setBrush(QColor(marker['color']))
            painter.drawPolygon(QPolygonF([QPointF(x,y-8),QPointF(x-4,y-13),QPointF(x+4,y-13)]))
        painter.end()

    def mousePressEvent(self,event):
        hits=self.markers_at(event.position())
        if event.button()==Qt.MouseButton.LeftButton and hits:
            self.eventPicked.emit(hits[0]['frame']);event.accept();return
        super().mousePressEvent(event)

    def event(self,event):
        if event.type()==QEvent.Type.ToolTip:
            hits=self.markers_at(event.pos())
            if hits:
                QToolTip.showText(event.globalPos(),'\n'.join(f"{m['label']} ({m['name']}) · {m['time']:.2f}s · {m['frame']+1} frame" for m in hits),self)
                return True
        return super().event(event)


class CurationDialog(QDialog):
    def __init__(self,root,parent=None,workspace=None):
        super().__init__(parent);self.root=Path(root).expanduser().resolve();self.workspace=workspace;self.split_generation=0
        self.rows=[];self.excluded=set();self.path=None;self.kind='datagen';self.job=None;self.split_plan=None;self.saved_collection=None;self.output_parent=None
        self.reader=None;self.frame_job=None;self.pending_frame=None;self.generation=0;self.seek_generation=0;self.pending_selection=None
        self.finished.connect(lambda _result:self.release_reader())
        self.setWindowTitle('Data Curation · 에피소드 검토');self.resize(1400,920)
        layout=QVBoxLayout(self);layout.setSpacing(10)
        header=QHBoxLayout();layout.addLayout(header)
        title=QLabel('Data Curation');title.setStyleSheet('font-size: 22px; font-weight: 600;');header.addWidget(title)
        header.addStretch();self.load_button=QPushButton('데이터 열기');self.load_button.clicked.connect(self.open_data);header.addWidget(self.load_button)
        self.dataset_label=QLabel('Episode 검토 · 영상과 동작 확인 · 포함 / 제외');layout.addWidget(self.dataset_label)
        self.info=QLabel('데이터를 열면 에피소드 목록이 표시됩니다.');self.info.setWordWrap(True);layout.addWidget(self.info)
        scroll=QScrollArea();self.page_scroll=scroll;scroll.setWidgetResizable(True);layout.addWidget(scroll,1)
        content=QWidget();page=QVBoxLayout(content);page.setContentsMargins(0,0,0,0);page.setSpacing(10);scroll.setWidget(content)
        split=QSplitter();split.setMinimumHeight(490);page.addWidget(split,1)
        left=QWidget();ll=QVBoxLayout(left);split.addWidget(left);ll.addWidget(QLabel('Episode 목록'))
        self.filter_toggle,self.filter_panel,filter_layout=self.fold(ll,'에피소드 필터')
        help_label=QLabel('현재 연 데이터 안에서 목록을 좁힙니다. 포함 / 제외 결과에는 영향을 주지 않습니다.');help_label.setWordWrap(True);filter_layout.addWidget(help_label)
        self.only_flags=QCheckBox('확인 필요 에피소드만 표시');self.only_flags.toggled.connect(self.refresh_list);filter_layout.addWidget(self.only_flags)
        self.filters={};form=QFormLayout();self.filter_form=form;filter_layout.addLayout(form)
        for key,label in [('task_id','작업'),('active_object_count','물체 개수'),('active_object_ids','물체 구성'),('pick_object_id','목표 물체'),
                          ('scene_configuration','Scene 구성'),('source_group','원본 그룹'),('split','Train / Validation'),('recorded_success','기록된 성공 라벨'),('annotation_status','Annotation 상태'),('failure_reason','평가 실패 유형'),('env_id','평가 환경')]:
            combo=QComboBox();combo.addItem('전체',None);combo.currentIndexChanged.connect(self.refresh_list)
            self.filters[key]=combo;form.addRow(label,combo)
            if key in ('failure_reason','env_id'):
                combo.hide();form.labelForField(combo).hide()
        self.list=QListWidget();self.list.setMinimumWidth(260);self.list.setWordWrap(True);ll.addWidget(self.list,1)
        self.exclude=QPushButton('선택 에피소드 제외');self.exclude.clicked.connect(self.toggle);ll.addWidget(self.exclude)
        right=QWidget();rl=QVBoxLayout(right);split.addWidget(right)
        self.episode_title=QLabel('에피소드를 선택하세요.');rl.addWidget(self.episode_title)
        self.review_reason=QLabel();self.review_reason.setWordWrap(True);self.review_reason.setTextFormat(Qt.TextFormat.PlainText)
        self.review_reason.setStyleSheet('color: #efb467; padding: 4px 0;');self.review_reason.hide();rl.addWidget(self.review_reason)
        self.preview_tabs=QTabWidget();rl.addWidget(self.preview_tabs,1)
        videos=QWidget();rgb_layout=QHBoxLayout(videos);self.rgb_views={}
        for camera in ('side','wrist'):
            column=QVBoxLayout();rgb_layout.addLayout(column,1)
            column.addWidget(QLabel(camera.title()))
            label=CameraView('영상 없음',(180,180));column.addWidget(label,1);self.rgb_views[camera]=label
        self.rgb=self.rgb_views['side']
        self.preview_tabs.addTab(videos,'영상 · Side / Wrist')
        trajectory_page=QWidget();trajectory_layout=QVBoxLayout(trajectory_page)
        views=QHBoxLayout();views.addWidget(QLabel('드래그 회전 · 휠 확대'));views.addStretch()
        self.trajectory=Plot('trajectory')
        for name in ('3D','XY','XZ','YZ'):
            button=QPushButton(name);button.clicked.connect(lambda checked=False,n=name:self.trajectory.set_view(n));views.addWidget(button)
        trajectory_layout.addLayout(views);self.trajectory_viewport=SquareViewport(self.trajectory);trajectory_layout.addWidget(self.trajectory_viewport,1)
        self.preview_tabs.addTab(trajectory_page,'로봇 경로');self.trajectory.set_view('3D')
        playbar=QHBoxLayout();rl.addLayout(playbar)
        self.play=QPushButton('재생');self.play.clicked.connect(self.toggle_play);playbar.addWidget(self.play)
        self.timeline=EventTimeline();self.timeline.valueChanged.connect(self.seek);self.timeline.eventPicked.connect(self.jump_to_event);playbar.addWidget(self.timeline,1)
        self.clock_label=QLabel('0 / 0');playbar.addWidget(self.clock_label)
        self.rate=QComboBox();self.rate.addItems(['0.25×','0.5×','1×','2×']);self.rate.setCurrentIndex(2)
        self.rate.currentIndexChanged.connect(self.pause);playbar.addWidget(self.rate)
        self.event_bar=QHBoxLayout();self.event_buttons={};rl.addLayout(self.event_bar)
        self.event_label=QLabel('');self.event_label.setWordWrap(True);self.event_bar.addWidget(self.event_label);self.event_bar.addStretch()
        self.play_timer=QTimer(self);self.play_timer.setInterval(33);self.play_timer.timeout.connect(self.advance)
        self.elapsed=QElapsedTimer();self.play_start=0;self.robot_cache={}
        split.setSizes([300,1050]);split.setStretchFactor(1,1)
        self.analysis_toggle,self.analysis_panel,analysis_layout=self.fold(page,'상세 분석 · 초기 분포 / 관절 목표 / 품질 지표')
        analysis_row=QHBoxLayout();analysis_layout.addLayout(analysis_row)
        self.scatter=Plot('scatter');self.scatter.setMinimumSize(410,360)
        self.scatter.setToolTip('점선: 지지면 전체 / 실선: 목표 물체 중심의 배치 범위\n보라색 네모: 기록된 로봇 기준점 위치(크기는 표시용)\n분할 표시: 초록 Train / 파랑 Val · 주황 테두리 확인 필요 / 흰 테두리 선택\n배치 범위에는 물체 크기와 기록 당시 여백이 반영되며 다른 물체와의 간격 조건도 적용됩니다.')
        analysis_row.addWidget(self.scatter,1)
        analysis_details=QVBoxLayout();analysis_row.addLayout(analysis_details,2)
        self.metrics=QLabel('선택한 에피소드의 기록과 품질 지표');self.metrics.setWordWrap(True);analysis_details.addWidget(self.metrics)
        self.actions=Plot('actions');self.actions.setMinimumHeight(175);analysis_details.addWidget(self.actions)
        self.advanced_toggle,self.advanced_panel,advanced_layout=self.fold(page,'데이터셋 구성 · 자동 선별 / Train·Validation 분할')
        bar=QHBoxLayout();advanced_layout.addLayout(bar)
        self.count=QSpinBox();self.count.setRange(1,100000);bar.addWidget(QLabel('사용할 데이터 수'));bar.addWidget(self.count)
        self.choose_button=QPushButton('적용');self.choose_button.clicked.connect(self.auto_select);bar.addWidget(self.choose_button)
        self.restore_button=QPushButton('제외 취소 · 모두 포함');self.restore_button.clicked.connect(self.restore_all);bar.addWidget(self.restore_button);self.exclude_all=QPushButton('전체 제외');self.exclude_all.clicked.connect(lambda:self.set_excluded({r['name'] for r in self.rows}));bar.addWidget(self.exclude_all);bar.addStretch()
        self.help_button=QPushButton('설명');self.help_button.clicked.connect(self.show_curation_help);bar.addWidget(self.help_button)
        sourcebar=QHBoxLayout();advanced_layout.addLayout(sourcebar)
        self.train_ratio=QDoubleSpinBox();self.train_ratio.setRange(1,99);self.train_ratio.setValue(80);self.train_ratio.setSuffix('%')
        self.split_seed=QSpinBox();self.split_seed.setRange(0,2147483647)
        self.train_ratio.setToolTip('포함된 Episode 중 Train으로 배정할 비율입니다. 분할은 선택 사항입니다.')
        self.split_seed.setToolTip('분할 조합을 재현하는 랜덤값입니다. 같은 데이터와 같은 값은 같은 분할을 만들며, 값을 바꾸면 다른 조합을 시도합니다. 에피소드 개수가 아닙니다.')
        sourcebar.addWidget(QLabel('학습 비율 (Train)'));sourcebar.addWidget(self.train_ratio)
        self.val_ratio=QLabel('검증 20%');sourcebar.addWidget(self.val_ratio)
        sourcebar.addWidget(QLabel('분할 랜덤값 (Seed)'));sourcebar.addWidget(self.split_seed)
        self.plan_split=QPushButton('Train/Val 배치 미리보기');self.plan_split.clicked.connect(self.preview_source_split);sourcebar.addWidget(self.plan_split)
        assignbar=QHBoxLayout();advanced_layout.addLayout(assignbar);assignbar.addWidget(QLabel('미리보기 수동 조정'))
        self.source_assignment=QComboBox();self.source_assignment.addItem('학습 (Train)','train');self.source_assignment.addItem('검증 (Validation)','val');assignbar.addWidget(self.source_assignment)
        self.assign_split=QPushButton('선택 Episode를 이쪽으로 이동');self.assign_split.clicked.connect(self.assign_source_split);assignbar.addWidget(self.assign_split)
        self.assign_split.setToolTip('선택한 Episode 하나만 이동합니다. 알려진 원본 계보 중복은 경고로 표시합니다.')
        self.cancel_split=QPushButton('미리보기 지우기');self.cancel_split.clicked.connect(self.cancel_source_split);assignbar.addWidget(self.cancel_split);assignbar.addStretch()
        self.split_status=QLabel('비율과 랜덤값을 정한 뒤 미리보기를 누르면 분포 그림에 색으로 표시됩니다.');self.split_status.setWordWrap(True);advanced_layout.addWidget(self.split_status)
        self.names_form=QFormLayout();page.addLayout(self.names_form)
        self.storage_name=QLineEdit();self.storage_name.setPlaceholderText('필수 · 예: 선별_1 또는 분할_1');self.names_form.addRow('저장 이름',self.storage_name)
        self.output_preview=QLabel();self.output_preview.setWordWrap(True);self.output_preview.setTextFormat(Qt.TextFormat.PlainText);self.names_form.addRow('예상 저장 위치',self.output_preview)
        self.output_browse=QPushButton('출력 폴더 선택');self.output_browse.clicked.connect(self.choose_output_parent);self.names_form.addRow('출력 위치',self.output_browse)
        self.storage_name.textChanged.connect(self.update_output_preview)
        footer=QHBoxLayout();layout.addLayout(footer);self.storage_note=QLabel('원본 보존 · 제외한 에피소드는 새 데이터셋에 저장하지 않습니다.');footer.addWidget(self.storage_note);footer.addStretch()
        self.export_button=QPushButton('데이터 저장');self.export_button.setObjectName('primaryAction');self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.save);footer.addWidget(self.export_button)
        self.cancel_job=QPushButton('저장 취소');self.cancel_job.setEnabled(False);self.cancel_job.clicked.connect(lambda:self.job.requestInterruption() if self.job is not None else None);footer.addWidget(self.cancel_job)
        self.scatter.picked.connect(self.select);self.list.currentItemChanged.connect(self.from_list)
        self.exclude.setEnabled(False)
        self.train_ratio.valueChanged.connect(self.split_settings_changed);self.split_seed.valueChanged.connect(self.split_settings_changed)
        self.update_split_controls()

    @staticmethod
    def fold(layout,title):
        button=QPushButton('▸ '+title);button.setCheckable(True);layout.addWidget(button)
        panel=QWidget();body=QVBoxLayout(panel);layout.addWidget(panel);panel.hide()
        def toggle(expanded):
            panel.setVisible(expanded);button.setText(('▾ ' if expanded else '▸ ')+title)
        button.toggled.connect(toggle)
        return button,panel,body

    def show_curation_help(self):
        QMessageBox.information(self,'데이터셋 구성 설명',
            '사용할 데이터 수 / 적용\n'
            '현재 포함된 후보에서 위치·작업·물체 구성의 비율을 고려해\n'
            '입력한 개수만 사용하고 나머지는 제외합니다. 20개 중 20개면 변화가 없습니다.\n\n'
            '제외 취소 · 모두 포함\n'
            '수동/자동 제외를 취소합니다. 원본 파일이나 로봇 상태를 바꾸지 않습니다.\n\n'
            '학습 비율 / 랜덤값 (Seed)\n'
            'Train은 학습용, Validation은 성능 확인용입니다.\n'
            '20개에서 학습 70%이면 약 14개 / 검증 6개입니다.\n'
            'Episode 단위로 배정하며 같은 원본 계보는 경고만 표시합니다.\n'
            '같은 데이터·같은 Seed는 같은 분할을 재현합니다. Seed는 개수가 아닙니다.\n\n'
            '미리보기 / 수동 조정 / 저장\n'
            '미리보기는 저장 전 계획이며 Train 초록 / Val 파랑으로 표시합니다.\n'
            '선택한 Episode 하나만 학습 또는 검증 쪽으로 옮깁니다.\n'
            '포함/제외·비율·Seed를 바꾸면 미리보기를 다시 만드세요.\n'
            '선별본이나 과거 분할본도 다시 선택·분할할 수 있습니다.\n'
            '하단 저장 버튼 하나를 사용합니다. 미리보기가 있으면 Train / Val을 각각 저장하고,\n'
            '없으면 현재 포함된 데이터를 저장합니다. 과거 Partition은 역사 정보로 보존하며 새 배정과 구분합니다.')

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

    def jump_to_event(self,frame):
        self.timeline.blockSignals(True);self.timeline.setValue(frame);self.timeline.blockSignals(False)
        self.seek(frame)

    def clear_events(self):
        self.timeline.set_events({})
        for button in self.event_buttons.values():
            self.event_bar.removeWidget(button);button.hide();button.deleteLater()
        self.event_buttons={};self.event_label.setText('')

    def install_events(self,row):
        self.clear_events();self.timeline.set_events(row.get('events',{}),row.get('timestamps'),row.get('dt',1/30))
        for marker in self.timeline.markers:
            button=QPushButton(f"{marker['label']} · {marker['time']:.2f}s · {marker['frame']+1}F");button.setCheckable(True)
            button.setToolTip(f"{marker['name']} · 기록된 주석 프레임으로 이동")
            button.setStyleSheet(f"QPushButton {{ border-color: {marker['color']}; padding: 5px 9px; }} QPushButton:checked {{ color: {marker['color']}; }}")
            button.clicked.connect(lambda checked=False,index=marker['frame']:self.jump_to_event(index))
            self.event_buttons[marker['name']]=button;self.event_bar.insertWidget(self.event_bar.count()-1,button)
        self.update_events(0)

    def update_events(self,frame):
        reached=[m for m in self.timeline.markers if m['frame']<=frame]
        if not self.timeline.markers:self.event_label.setText('집기 / 놓기 주석 없음')
        elif reached:
            last=reached[-1];self.event_label.setText('주석 시점 · 현재: '+last['label']+(' 시점' if frame==last['frame'] else ' 이후'))
        else:self.event_label.setText('주석 시점 · 현재: 첫 주석 이전')
        for marker in self.timeline.markers:self.event_buttons[marker['name']].setChecked(frame>=marker['frame'])

    def show_frame(self,frame):
        if self.trajectory.row is None:return
        if self.reader is not None:
            self.pending_frame=(self.generation,frame,tuple(n for n in self.rgb_views if n in self.reader.row.get('cameras',{})))
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
        self.update_events(frame)
        for camera,label in self.rgb_views.items():
            rgb=payload.get('rgb_frames',{}).get(camera)
            if rgb is not None:
                h,w,_=rgb.shape;picture=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format.Format_RGB888).copy()
                label.setPixmap(QPixmap.fromImage(picture))
            else:label.clear();label.setText(camera.title()+' · 기록된 영상 없음')

    def error(self,exc):QMessageBox.warning(self,'Curation',str(exc))
    def run_job(self,fn,done,*,cancellable=False):
        if self.job is not None and self.job.isRunning():
            self.error('현재 분석/저장이 끝난 뒤 실행하세요.');return
        self.job=Job(fn,self);result=[];errors=[]
        self.cancel_job.setEnabled(cancellable);self.export_button.setEnabled(False)
        for control in (self.load_button,self.list,self.plan_split,self.assign_split,self.exclude,self.restore_button,self.exclude_all,self.choose_button,self.storage_name,self.output_browse,self.train_ratio,self.split_seed,self.count,self.cancel_split,self.source_assignment):control.setEnabled(False)
        self.job.result.connect(result.append);self.job.failed.connect(errors.append)
        def finish():
            self.job=None;self.cancel_job.setEnabled(False)
            for control in (self.load_button,self.list,self.exclude,self.exclude_all,self.storage_name,self.output_browse):control.setEnabled(True)
            self.update_split_controls()
            if errors:
                if 'Curation 취소됨' in str(errors[0]):self.info.setText(str(errors[0]))
                else:self.error(errors[0])
            elif result:done(result[0])
            if self.pending_selection is not None:
                index=self.pending_selection;self.pending_selection=None;self.select(index)
        self.job.finished.connect(finish);self.job.finished.connect(self.job.deleteLater)
        self.job.start()
    def reject(self):
        self.pause();self.pending_frame=None
        if any(j is not None and j.isRunning() for j in (self.job,self.frame_job)):
            self.error('저장/분석 완료 후 닫아주세요.');return
        self.release_reader()
        super().reject()
    def closeEvent(self,event):
        self.pause();self.pending_frame=None
        if any(j is not None and j.isRunning() for j in (self.job,self.frame_job)):event.ignore()
        else:
            self.release_reader();super().closeEvent(event)

    def release_reader(self):
        """The dialog owns playback; an in-flight frame retains it until finished."""
        reader=self.reader;self.reader=None
        if reader is None:return
        if self.frame_job is not None and self.frame_job.isRunning():
            self.frame_job.finished.connect(reader.close)
        else:reader.close()
    def open_data(self):
        from .paths import browse_directory
        directory=browse_directory('source',self.path or '',root=self.root)
        picker=DatasetPicker(self,directory)
        if picker.exec()==QDialog.DialogCode.Accepted:self.load_data(picker.selectedFiles()[0])

    def open_latest(self):
        from .paths import discover_inputs
        def newest():
            items=discover_inputs(self.root,category='dataset')
            items=[v for v in items if v['format'] in ('hdf5','canonical','evaluation')]
            if self.workspace:
                from .dataset import EpisodeDataset
                compatible=[]
                for item in items:
                    try:
                        if item['format']=='evaluation':
                            identity=json.loads((Path(item['path'])/'evaluation_datasets.json').read_text()).get('workspace')
                        else:identity=EpisodeDataset(item['path']).args.get('workspace')
                        if identity==self.workspace:compatible.append(item)
                    except (ValueError,OSError,KeyError):continue
                items=compatible
            return max(items,key=lambda v:Path(v['path']).stat().st_mtime)['path'] if items else None
        def done(path):
            if path:self.load_data(path)
            else:self.info.setText('완료 데이터를 찾지 못했습니다. 데이터 열기에서 Path를 선택하세요.')
        self.run_job(newest,done)

    def load_data(self,path):
        if self.job is not None:return self.error('현재 분석/저장이 끝난 뒤 실행하세요.')
        path=str(Path(path).resolve());self.pause();self.generation+=1;self.pending_frame=None;self.pending_selection=None
        self.clear_split_preview()
        self.clear_events()
        self.release_reader();self.trajectory.row=None;self.list.setEnabled(False);self.load_button.setEnabled(False);self.export_button.setEnabled(False)
        self.info.setText('데이터 분석 중…')
        def read():
            selected=Path(path)
            if selected.is_dir() and not (selected/'manifest.json').is_file() and not (selected/'evaluation_datasets.json').is_file():
                candidates=sorted([*selected.glob('*.hdf5'),*selected.glob('*.h5')])
                if not candidates and (selected/'origin').is_dir():
                    candidates=sorted([*(selected/'origin').glob('*.hdf5'),*(selected/'origin').glob('*.h5')])
                if len(candidates)==1:selected=candidates[0]
            resolved=str(selected)
            rows=analyze(resolved)
            from .dataset import episode_input_kind
            kind='evaluation_dataset' if any(r.get('evaluation_outcome') for r in rows) else episode_input_kind(resolved)
            return resolved,rows,kind
        self.run_job(read,lambda result:self.loaded(*result))
    def loaded(self,path,rows,kind=None):
        try:
            self.rows=rows;self.path=path;self.excluded=set();self.split_plan=None;self.split_generation+=1
            self.kind=kind or ('evaluation_dataset' if any(r.get('evaluation_outcome') for r in rows) else 'source')
            if self.kind not in ('source','annotated','datagen','evaluation_dataset'):raise ValueError('Source / Annotation / Datagen / 평가 데이터를 선택하세요.')
            self.count.setMaximum(len(self.rows));self.count.setValue(len(self.rows))
            for key,combo in self.filters.items():
                if key in ('failure_reason','env_id'):
                    combo.setVisible(self.kind=='evaluation_dataset');self.filter_form.labelForField(combo).setVisible(self.kind=='evaluation_dataset')
                combo.blockSignals(True);combo.clear();combo.addItem('전체',None)
                for value in sorted({self.filter_value(r,key) for r in rows}):combo.addItem(value,value)
                combo.blockSignals(False)
            self.dataset_label.setText({'source':'Source · 수집 데이터','annotated':'Annotation · 주석 데이터','datagen':'DataGen · 생성 데이터','evaluation_dataset':'Sim 평가 · 성공 / 실패 기록'}[self.kind]+'  /  '+Path(path).name)
            self.dataset_label.setToolTip(str(path));self.export_button.setEnabled(True);self.exclude.setEnabled(True)
            self.scatter.context_rows=self.rows;self.scatter.rows=self.rows;self.refresh_list();self.select(0)
        except Exception as exc:self.error(exc)
    def refresh_list(self):
        self.list.blockSignals(True);self.list.clear()
        self.scatter.context_rows=self.rows;self.scatter.excluded=self.excluded;split_mode=self.scatter.has_splits()
        visible=[]
        for i,r in enumerate(self.rows):
            if self.only_flags.isChecked() and not r['flags']:continue
            if any(c.currentData() is not None and self.filter_value(r,k)!=c.currentData() for k,c in self.filters.items()):continue
            visible.append(i)
            state='제외' if r['name'] in self.excluded else '확인 필요' if r['flags'] else '포함'
            text=r['name']+'  ·  '+state+'\n'+str(r.get('pick_object_id') or r['task_id'])+(' · '+r['split'] if r.get('split') else '')
            if r.get('evaluation_outcome'):
                from .presentation import failure_label
                text+='\n'+('성공' if r['recorded_success'] else failure_label(r.get('failure_reason')))+f" · Env {r.get('env_id','—')}"
            if r['flags']:text+='\n확인: '+', '.join(r['flags'])
            item=QListWidgetItem(text);item.setToolTip(r['task_id']+' · '+(r.get('split') or '미분할')+'\n'+(self.review_text(r) or '특이 지표 없음'))
            item.setForeground(QColor(self.scatter.point_color(r,split_mode)))
            item.setData(Qt.ItemDataRole.UserRole,i);self.list.addItem(item)
            if i==self.scatter.selected:self.list.setCurrentItem(item)
        self.list.blockSignals(False);points=[i for i in visible if self.rows[i].get('xy') is not None]
        self.scatter.rows=[self.rows[i] for i in points];self.scatter.original_indices=points
        self.scatter.excluded=self.excluded;self.scatter.update()
        self.info.setText(f'전체 {len(self.rows)}   ·   포함 {len(self.rows)-len(self.excluded)}   ·   제외 {len(self.excluded)}   ·   확인 필요 {sum(bool(r["flags"]) for r in self.rows)}   ·   표시 {len(visible)}')
        if 0<=self.scatter.selected<len(self.rows):self.exclude.setText('선택 에피소드 복원' if self.rows[self.scatter.selected]['name'] in self.excluded else '선택 에피소드 제외')
        self.update_split_controls()

    @staticmethod
    def filter_value(row,key):
        value=row.get(key)
        if key=='failure_reason':
            from .presentation import failure_label
            return failure_label(value)
        if key=='scene_configuration':value=(value or {}).get('sha256')
        if isinstance(value,(list,tuple)):return ', '.join(value)
        return 'UNKNOWN' if value is None else str(value)
    def from_list(self,current,previous):
        if current:self.select(current.data(Qt.ItemDataRole.UserRole))

    @staticmethod
    def review_text(row):
        if row.get('evaluation_outcome'):
            from .presentation import failure_label
            evaluation=row.get('evaluation',{})
            result='성공' if row['recorded_success'] else failure_label(row.get('failure_reason'))
            checks=' · '.join(k+(': 충족' if v else ': 미충족') for k,v in evaluation.get('checks',{}).items())
            return f"Sim 평가: {result} · Env {row.get('env_id','—')}\n"+checks
        if not row.get('flags'):return ''
        lines=[];explained=set()
        for detail in row.get('flag_details',[]):
            def value(number):
                if detail['metric'] in ('stop_fraction','command_stop_fraction'):return f'{number*100:.1f}%'
                return f'{number:.2f}'+('s' if detail['metric']=='duration' else ' rad/s²')
            comparison='초과' if detail['direction']=='high' else '미만'
            lines.append(f"{detail['label']}: {value(detail['value'])} · 비교 기준 {value(detail['threshold'])} {comparison} "
                         f"(같은 조건 {detail['group_size']}개 중앙값 {value(detail['median'])})")
            explained.add(detail['label'])
        lines.extend(flag for flag in row['flags'] if flag not in explained)
        return '확인 필요 · '+'\n'.join(lines)+'\n같은 작업·물체 구성 내 비교입니다. 불량 확정은 아닙니다.'

    def update_split_controls(self):
        evaluation=self.kind=='evaluation_dataset'
        locked=evaluation and Path(self.path or '').is_dir();planned=self.split_plan is not None
        self.advanced_toggle.setVisible(not locked)
        if locked:self.advanced_toggle.setChecked(False);self.advanced_panel.hide()
        self.storage_name.setVisible(not locked);self.names_form.labelForField(self.storage_name).setVisible(not locked)
        self.output_preview.setVisible(not locked);self.names_form.labelForField(self.output_preview).setVisible(not locked)
        self.export_button.setVisible(not locked)
        self.output_browse.setVisible(not locked);self.names_form.labelForField(self.output_browse).setVisible(not locked)
        self.storage_note.setText('평가 완료 시 성공/실패 데이터가 자동 저장됩니다. 현재 화면은 평가 기록을 검토합니다.' if locked else '원본 보존 · 제외한 에피소드는 새 데이터셋에 저장하지 않습니다.')
        eligible=sum(r['name'] not in self.excluded for r in self.rows)
        self.plan_split.setEnabled(bool(self.path) and eligible>=2 and not locked)
        for control in (self.assign_split,self.source_assignment):control.setEnabled(planned and not locked)
        for control in (self.train_ratio,self.split_seed):control.setEnabled(bool(self.rows) and not locked)
        self.cancel_split.setVisible(planned);self.choose_button.setEnabled(bool(self.rows));self.count.setEnabled(bool(self.rows))
        self.restore_button.setEnabled(bool(self.excluded));self.export_button.setEnabled(bool(self.path) and eligible>0 and not locked)
        self.export_button.setText('Train / Val 저장' if planned else '데이터 저장')
        self.export_button.setToolTip('같은 Source 묶음의 입력한 이름 아래 Train / Validation을 저장합니다.' if planned else '현재 포함된 에피소드를 입력한 이름으로 저장합니다. 과거 분할은 역사 정보로만 보존합니다.')
        if planned:
            counts={s:sum(r.get('split')==s and r['name'] not in self.excluded for r in self.rows) for s in ('train','val')}
            self.split_status.setText(f"저장 전 · Train {counts['train']}개 / Val {counts['val']}개"+'\n'+'\n'.join(self.split_plan.get('warnings',[])))
            self.split_status.setToolTip('Episode별 현재 배정입니다. 경고가 없어도 계보가 완전하거나 독립적임을 증명하지 않습니다.')
        elif locked:
            self.split_status.setText('평가 기록 · 성공/실패 분포와 궤적 비교 · Train/Val 분할 대상이 아닙니다.')
            self.split_status.setToolTip('에피소드 필터에서 성공 라벨·실패 유형·환경을 선택하세요.')
        else:
            self.split_status.clear()
            self.split_status.setToolTip('')
        self.split_status.setVisible(planned or locked)
        self.update_output_preview()

    def curation_destination(self):
        from .paths import validate_storage_name, source_collection
        if not self.storage_name.text():raise ValueError('저장 이름을 입력하세요.')
        name=validate_storage_name(self.storage_name.text())
        split=self.split_plan is not None
        if self.output_parent:return Path(self.output_parent)/(name if split or Path(self.path or '').is_dir() else name+'.hdf5')
        if self.path and self.kind=='source':
            if name=='origin':raise ValueError('origin은 원본 폴더 이름입니다. 다른 저장 이름을 입력하세요.')
            folder=source_collection(self.path,self.root)/name
            return folder if split or Path(self.path).is_dir() else folder/'dataset.hdf5'
        bucket={'annotated':'annotations','datagen':'datagen','evaluation_dataset':'evaluations'}.get(self.kind,'datasets')
        return self.root/bucket/(name if split or self.path and Path(self.path).is_dir() else name+'.hdf5')

    def choose_output_parent(self):
        folder=QFileDialog.getExistingDirectory(self,'새 결과를 저장할 폴더',str(self.output_parent or self.root))
        if folder:self.output_parent=folder;self.update_output_preview()

    def update_output_preview(self):
        try:self.output_preview.setText(str(self.curation_destination()))
        except ValueError as exc:self.output_preview.setText(str(exc))

    def clear_split_preview(self):
        self.split_generation+=1;self.split_plan=None
        for row in self.rows:row['split']=None
        combo=self.filters['split'];combo.blockSignals(True);combo.clear();combo.addItem('전체',None)
        for value in sorted({r['split'] for r in self.rows if r.get('split')}):combo.addItem(value,value)
        combo.blockSignals(False)

    def cancel_source_split(self):
        self.clear_split_preview();self.refresh_list()

    def split_settings_changed(self,*args):
        self.val_ratio.setText(f'검증 {100-self.train_ratio.value():g}%')
        self.clear_split_preview();self.refresh_list()

    def set_excluded(self,names):
        names=set(names)
        if names!=self.excluded:self.excluded=names;self.clear_split_preview()
        self.refresh_list()

    def select(self,i):
        if not 0<=i<len(self.rows):return
        self.pause();self.generation+=1;generation=self.generation
        self.scatter.selected=i;self.scatter.update();self.pending_frame=None;self.release_reader()
        self.list.blockSignals(True)
        for index in range(self.list.count()):
            if self.list.item(index).data(Qt.ItemDataRole.UserRole)==i:self.list.setCurrentRow(index);break
        self.list.blockSignals(False)
        self.exclude.setText('선택 에피소드 복원' if self.rows[i]['name'] in self.excluded else '선택 에피소드 제외')
        self.trajectory.row=None
        self.clear_events()
        for label in self.rgb_views.values():label.clear();label.setText('에피소드 불러오는 중…')
        row=self.rows[i]
        reason=self.review_text(row);self.review_reason.setText(reason);self.review_reason.setVisible(bool(reason))
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
                points=np.concatenate([model.triangles(state)[0].reshape(-1,3) for state in q[::max(1,len(q)//40)]])
                bounds=(points.min(0)-.005,points.max(0)+.005)
            return loaded,model,bounds
        self.run_job(prepare,lambda result:self.install_episode(*result) if generation==self.generation else None)

    def install_episode(self,row,model,bounds):
        self.release_reader()
        self.trajectory.robot=model;self.trajectory.mesh=None;self.trajectory.frame=0;self.trajectory.payload={}
        if bounds is not None:self.trajectory.robot_bounds=bounds
        self.reader=EpisodeReader(row) if 'data_ref' in row else None
        for plot in (self.trajectory,self.actions):plot.row=row;plot.update()
        self.episode_title.setText(f"{row['name']} · {row.get('pick_object_id') or row['task_id']} · {row['duration']:.2f}s")
        reason=self.review_text(row);self.review_reason.setText(reason);self.review_reason.setVisible(bool(reason))
        self.timeline.blockSignals(True);self.timeline.setRange(0,row.get('frames',len(row['ee']))-1);self.timeline.setValue(0);self.timeline.blockSignals(False)
        self.install_events(row)
        self.show_frame(0)
        def mm(value):return '기록 없음' if value is None else f'{value*1000:.1f} mm'
        def success(value):return '성공' if value is True else '실패' if value is False else '판정 근거 없음'
        missing=[n for n,v in row.get('object_paths',{}).items() if v is None]
        self.metrics.setText(
            (reason+'\n' if reason else '품질 참고값 · 확인 필요 표시는 불량 확정이 아닙니다.\n')+
            f"기록 라벨: {success(row.get('recorded_success'))} · 물리 판정: {success(row.get('physical_success'))}\n"
            f"EE 정지 {100*row['stop_fraction']:.1f}% · 정지 명령 {100*row.get('command_stop_fraction',0):.1f}%\n"
            f"명령 가속도 RMS · 팔 {row['jitter']:.2f} / 그리퍼 {row['gripper_jitter']:.2f} rad/s²\n"
            f"목표 이동 {mm(row.get('target_movement_m'))} · 들어올린 높이 {mm(row.get('lift_height_m'))}\n"
            f"EE 최소거리 {mm(row.get('target_ee_distance_m',{}).get('min'))} · 근접 자체는 집기 성공을 뜻하지 않습니다."
            +('\n물체 경로 기록 없음: '+', '.join(missing) if missing else ''))
        self.metrics.setToolTip(
            'EE 정지: 끝단 속도 2 mm/s 미만인 구간. 정지 명령: 관절 목표 변화 1°/s 미만인 구간.\n'
            '명령 가속도 RMS는 명령의 급격한 변화 정도이며 측정 관절 가속도와 다릅니다.\n'
            f"Task: {row['task_id']}\nActive: {', '.join(row.get('active_object_ids') or [])}\n"
            f"과거 Partition: {row.get('historical_split') or '없음'} (현재 배정 아님)\n"
            f"Source group: {row.get('source_group','UNKNOWN')}\nScene: {((row.get('scene_configuration') or {}).get('sha256') or 'UNKNOWN')}\n"
            f"Success source: {row.get('success_source','unknown')}\n"
            f"Grasp 전/후 이동: {mm(row.get('movement_before_grasp_m'))} / {mm(row.get('movement_after_grasp_m'))}\n"
            +'객체별 최소 접근: '+', '.join(n+': '+mm(v['min_distance_m']) for n,v in row.get('object_approach',{}).items()))

    def toggle(self):
        if not 0<=self.scatter.selected<len(self.rows):return
        name=self.rows[self.scatter.selected]['name']
        self.set_excluded(self.excluded^{name})
    def restore_all(self):
        self.count.setValue(len(self.rows));self.set_excluded(set())

    def auto_select(self):
        try:
            eligible=[r for r in self.rows if r['name'] not in self.excluded]
            chosen=set(select_distribution(eligible,self.count.value()))
            self.set_excluded({r['name'] for r in self.rows if r['name'] not in chosen})
        except Exception as exc:self.error(exc)

    def preview_source_split(self):
        if not self.path:return
        from .curation import split_dataset
        names=[r['name'] for r in self.rows if r['name'] not in self.excluded]
        path,ratio,seed=self.path,self.train_ratio.value()/100,self.split_seed.value();generation=self.split_generation
        self.run_job(lambda:split_dataset(path,ratio,seed,names=names),lambda plan:self.install_source_split(plan) if generation==self.split_generation else None)

    def install_source_split(self,plan):
        self.split_plan=plan
        for row in self.rows:row['split']=plan['assignments'].get(row['name'])
        combo=self.filters['split'];combo.blockSignals(True);combo.clear();combo.addItem('전체',None)
        for value in ('train','val'):combo.addItem(value,value)
        combo.blockSignals(False)
        self.refresh_list()
        self.analysis_toggle.setChecked(True)
        if self.isVisible():QTimer.singleShot(0,lambda:self.page_scroll.ensureWidgetVisible(self.scatter,0,20))

    def assign_source_split(self):
        if not self.split_plan:return self.error('먼저 Source 분할 미리보기를 실행하세요.')
        if not 0<=self.scatter.selected<len(self.rows):return self.error('분포의 점이나 목록에서 옮길 에피소드를 먼저 선택하세요.')
        from .curation import split_dataset
        assignments=dict(self.split_plan['assignments']);name=self.rows[self.scatter.selected]['name']
        if name not in assignments:return self.error('포함된 Source를 선택하세요.')
        assignments[name]=self.source_assignment.currentData()
        if len(set(assignments.values()))<2:return self.error('학습과 검증에 각각 최소 한 개의 Episode을 남겨야 합니다.')
        path,ratio,seed=self.path,self.train_ratio.value()/100,self.split_seed.value();generation=self.split_generation
        self.run_job(lambda:split_dataset(path,ratio,seed,assignments=assignments,names=list(assignments)),lambda plan:self.install_source_split(plan) if generation==self.split_generation else None)

    def save_source_split(self):
        if not self.split_plan:return self.error('Source 분할을 먼저 확인하세요.')
        try:
            dest=self.curation_destination()
            if dest.exists():raise ValueError('이미 존재하는 결과물입니다. 새 저장 이름을 입력하세요.')
        except ValueError as exc:return self.error(exc)
        from .curation import export_splits
        source,kind=self.path,self.kind;names=[r['name'] for r in self.rows if r['name'] not in self.excluded]
        assignments={n:self.split_plan['assignments'][n] for n in names}
        ratio,seed=self.train_ratio.value()/100,self.split_seed.value()
        def work():
            result=export_splits(source,dest,ratio,seed,assignments,names,cancel=self.job.isInterruptionRequested)
            result['folder']=str(dest)
            return result
        def done(result):
            self.saved_collection=result['folder']
            self.info.setText(f'Train {result["counts"]["train"]} / Validation {result["counts"]["val"]} 저장 완료 · {self.saved_collection}')
            self.update_output_preview()
        self.run_job(work,done,cancellable=True)

    def save(self):
        if not self.path:return
        if self.split_plan is not None:return self.save_source_split()
        canonical=Path(self.path).is_dir()
        try:proposed=self.curation_destination()
        except ValueError as exc:return self.error(exc)
        dest=str(proposed)
        if Path(dest).exists():return self.error('이미 존재하는 결과물입니다. 새 저장 이름/위치를 선택하세요.')
        names=[r['name'] for r in self.rows if r['name'] not in self.excluded]
        source,kind=self.path,self.kind
        self.info.setText('새 데이터 저장·검증 중… 원본은 유지됩니다.')
        def work():
            output=export_subset(source,dest,names,cancel=self.job.isInterruptionRequested)
            return output
        def done(output):
            self.saved_collection=str(output)
            self.info.setText(f'저장 완료: {output} · {len(names)} episodes')
        self.run_job(work,done,cancellable=True)
