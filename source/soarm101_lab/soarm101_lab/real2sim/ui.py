"""Desktop hardware control, measured capture datasets and workflow monitoring."""

import argparse
import json
from pathlib import Path
import sys
import time
from threading import Event
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QFileDialog,
    QMessageBox,
    QTabWidget,
    QComboBox,
    QListWidget,
    QGroupBox,
    QScrollArea,
    QCheckBox,
    QSplitter,
    QSizePolicy,
)
from ..workflow.view import CameraView
from .profile import load
from .storage import Bus, atomic

ROOT = Path(__file__).resolve().parents[4]


class Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(self.progress.emit))
        except Exception as exc:
            self.failed.emit(str(exc))


class Window(QMainWindow):
    def __init__(self, workspace):
        super().__init__()
        self.workspace=Path(workspace).resolve();self.workspace.mkdir(parents=True,exist_ok=True)
        self.bus=Bus(self.workspace/'ipc')
        from .launcher import RuntimeLauncher
        self.runtime_launcher=RuntimeLauncher(self.bus,self.workspace,ROOT)
        self.runtime_enabled=False
        self.runtime_session=None
        self.requested_backend=None
        self.teleop_start_buttons=[]
        self.worker=None;self.cancel=Event();self.pending={};self.last_reply=None
        self.setWindowTitle('SO-101 Real2Sim · Snapshot Capture');self.resize(1280,900)
        root=QWidget();self.setCentralWidget(root);layout=QVBoxLayout(root)
        self.status=QLabel('장치 연결 대기');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.content=QSplitter(Qt.Orientation.Horizontal);layout.addWidget(self.content,1)
        self.tabs=QTabWidget();self.tabs.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Expanding);self.content.addWidget(self.tabs)
        capture=QWidget();cap=QVBoxLayout(capture)
        self.capture_scroll=QScrollArea();self.capture_scroll.setWidgetResizable(True);self.capture_scroll.setWidget(capture)
        self.tabs.addTab(self.capture_scroll,'1 · Real2Sim Capture')
        cameras=QHBoxLayout();cap.addLayout(cameras,1);self.capture_views={}
        for name,title in [('real_side','REAL SIDE CAMERA'),('real_wrist','REAL WRIST CAMERA')]:
            tile=QWidget();column=QVBoxLayout(tile);column.addWidget(QLabel(title))
            image=CameraView(minimum=(240, 160));image.setText('No live frame');column.addWidget(image,1)
            cameras.addWidget(tile,1);self.capture_views[name]=image
        self.capture_readiness=QLabel();self.capture_readiness.setWordWrap(True);cap.addWidget(self.capture_readiness)
        self.capture_session_label=QLabel('세션: Start 후 첫 캡처부터 같은 폴더에 저장합니다.')
        self.capture_session_label.setWordWrap(True);cap.addWidget(self.capture_session_label)
        capture_controls=QHBoxLayout();cap.addLayout(capture_controls)
        self.capture_start=self.button(capture_controls,'Start · 카메라/로봇 연결',self.start_capture)
        self.teleop_start_buttons.append(self.button(capture_controls,'Start Teleop',lambda:self.send('start')))
        self.button(capture_controls,'Stop Teleop',self.stop)
        self.button(capture_controls,'Disconnect',self.disconnect_sim)
        hint=QLabel('Start는 실제 장치만 연결합니다. Sim은 실행하지 않습니다. 로봇 추종은 Start Teleop으로 시작하세요.');hint.setWordWrap(True);cap.addWidget(hint)
        self.capture_button=self.button(cap,'현재 상태 캡처',self.capture_snapshot)
        self.capture_result=QLabel('아직 캡처하지 않았습니다.');self.capture_result.setWordWrap(True);cap.addWidget(self.capture_result)
        self.snapshot_list=QListWidget();self.snapshot_list.setMaximumHeight(130);cap.addWidget(self.snapshot_list)
        hint=QLabel('물체 배치와 robot base는 유지하고 로봇 자세만 바꾸세요. 배치가 달라지면 Scene state를 변경하세요.');hint.setWordWrap(True);cap.addWidget(hint)
        self.capture_options=QGroupBox('Snapshot 저장 설정');opts=QFormLayout(self.capture_options)
        opts.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.capture_name=QLineEdit('Scene');opts.addRow('Scene 이름',self.capture_name)
        self.capture_output=QLineEdit(str(self.workspace/'snapshots'));opts.addRow('저장 위치',self.capture_output)
        folder_buttons=QHBoxLayout();self.capture_output_browse=self.button(folder_buttons,'폴더 선택',self.choose_capture_output);opts.addRow(folder_buttons)
        self.capture_scene_state=QLineEdit('static_001');opts.addRow('Scene state (물체 배치)',self.capture_scene_state)
        live=QWidget();live_l=QVBoxLayout(live)
        self.live_scroll=QScrollArea();self.live_scroll.setWidgetResizable(True);self.live_scroll.setWidget(live)
        self.tabs.addTab(self.live_scroll,'2 · Real / Sim')
        backend_row=QHBoxLayout();live_l.addLayout(backend_row)
        backend_row.addWidget(QLabel('Simulation backend'))
        self.live_backend=QComboBox();self.live_backend.addItem('Isaac Sim / Isaac Lab','isaac');self.live_backend.addItem('MuJoCo','mujoco');backend_row.addWidget(self.live_backend,1)
        grid=QGridLayout();live_l.addLayout(grid,1);self.live_grid=grid;self.live_tiles={};self.views={}
        for i,(name,title) in enumerate([('real_side','REAL SIDE'),('sim_side','SIM SIDE'),('real_wrist','REAL WRIST'),('sim_wrist','SIM WRIST')]):
            tile=QWidget();column=QVBoxLayout(tile);column.addWidget(QLabel(title));image=CameraView(minimum=(240, 160))
            image.setText('No live frame');column.addWidget(image,1);grid.addWidget(tile,i//2,i%2);self.views[name]=image;self.live_tiles[name]=tile
        controls=QHBoxLayout();live_l.addLayout(controls)
        self.button(controls,'Connect',self.connect_hw)
        self.teleop_start_buttons.append(self.button(controls,'Start Teleop',lambda:self.send('start')))
        self.button(controls,'Stop Teleop',self.stop)
        self.button(controls,'Disconnect',self.disconnect_sim)
        self.pair_button=self.button(controls,'현재 상태 Pair 캡처',self.capture_pair)
        self.pair_result=QLabel('Pair 저장 위치와 Scene state는 오른쪽 Snapshot 저장 설정에서 변경하세요.');self.pair_result.setWordWrap(True);live_l.addWidget(self.pair_result)
        hint=QLabel('Connect는 새 follower 목표를 보내지 않습니다. Start가 Teleop을 활성화하고 Stop은 새 목표 전송을 멈춥니다.');hint.setWordWrap(True);live_l.addWidget(hint)
        settings=QWidget();settings_layout=QVBoxLayout(settings)
        self.scene_options=QGroupBox('Scene / 관절 보정 · 다음 Connect');scene_form=QVBoxLayout(self.scene_options)
        settings_layout.addWidget(self.scene_options)
        self.scene_controls=QWidget();scene_controls=QVBoxLayout(self.scene_controls);scene_controls.setContentsMargins(0,0,0,0)
        scene_form.addWidget(self.scene_controls)
        scene_controls.addWidget(QLabel('Scene (Workspace Pipeline과 공유)'))
        self.live_scene=QComboBox();self.live_scene.setMinimumContentsLength(16)
        self.live_scene.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        scene_controls.addWidget(self.live_scene)
        scene_buttons=QHBoxLayout();scene_controls.addLayout(scene_buttons)
        self.scene_browse=self.button(scene_buttons,'Scene 선택 / Import',lambda:self.workflow_panel.browse_workspace())
        self.scene_refresh=self.button(scene_buttons,'목록 새로고침',self.refresh_live_scenes)
        self.live_workspace=QCheckBox('선택한 Scene 사용');self.live_workspace.setChecked(True);scene_controls.addWidget(self.live_workspace)
        self.scene_selection_hint=QLabel();self.scene_selection_hint.setWordWrap(True)
        self.scene_selection_hint.setSizePolicy(QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Minimum)
        scene_controls.addWidget(self.scene_selection_hint)
        self.connection_group=QGroupBox('연결 설정');form=QFormLayout(self.connection_group);self.connection_form=form
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        settings_layout.addWidget(self.connection_group)
        self.sim_options=QGroupBox('Simulation');sim_form=QFormLayout(self.sim_options)
        sim_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows);form.addRow(self.sim_options)
        self.live_real_preview=QCheckBox('미리보기에서 REAL 카메라 연결');sim_form.addRow(self.live_real_preview)
        self.live_renderer=QComboBox();self.live_renderer.addItems(['software-egl','egl','osmesa']);sim_form.addRow('MuJoCo renderer',self.live_renderer)
        self.live_renderer.setCurrentText('egl')
        from .launcher import resolve_live_renderer
        if resolve_live_renderer('osmesa') != 'osmesa':
            item=self.live_renderer.model().item(self.live_renderer.findText('osmesa'))
            item.setEnabled(False);item.setToolTip('OSMesa 라이브러리 없음 · software-egl로 CPU 렌더링을 사용하세요.')
        self.fields={}
        for key,title,default in [('profile','Profile / camera calibration',''),('leader_port','Leader port',''),('follower_port','Follower port',''),('follower_id','Follower calibration ID',''),('leader_calibration','Leader calibration file','so101_leader.json'),('follower_calibration_dir','Follower calibration directory',''),('side_camera','Side camera index/path',''),('wrist_camera','Wrist camera index/path','')]:
            self.fields[key]=QLineEdit(default);form.addRow(title,self.fields[key])
        buttons=QVBoxLayout();form.addRow(buttons)
        self.button(buttons,'Profile 선택',self.open_profile)
        self.button(buttons,'선택 Profile 적용',lambda:self.send('apply_profile',{'profile':self.profile_path()}))
        self.sim_preview_button=self.button(buttons,'Sim 미리보기',lambda:self.connect_hw(True))
        self.use_precision=QCheckBox('저장된 관절 보정 사용');scene_form.addWidget(self.use_precision)
        self.identity_sim=QCheckBox('URDF 원형 검사 · Scene 관절 보정 제외');scene_form.addWidget(self.identity_sim)
        self.use_precision.setToolTip('현재 활성 Leader/Follower Precision 보정을 사용합니다. Capture와 Real/Sim이 같은 설정을 사용합니다.')
        self.identity_sim.setToolTip('체크하면 Scene의 Sim 관절 보정을 제외합니다. 접촉 보정 Scene을 검토할 때는 해제하세요. 실제 Precision 보정 파일은 바뀌지 않습니다.')
        has_precision=any((self.workspace/'joint_calibration'/role/'active.json').is_file() for role in ('leader','follower'))
        self.use_precision.setChecked(has_precision);self.identity_sim.setChecked(has_precision)
        settings_layout.addWidget(self.capture_options);settings_layout.addStretch()
        # Capture and Real/Sim share one right-hand configuration panel.
        self.settings_scroll=QScrollArea();self.settings_scroll.setWidgetResizable(True);self.settings_scroll.setWidget(settings)
        self.settings_scroll.setMinimumWidth(320);self.content.addWidget(self.settings_scroll)
        self.content.setChildrenCollapsible(False)
        self.content.setStretchFactor(0,1);self.content.setStretchFactor(1,0)
        self.content.setSizes([880,360])
        self.live_details=QLabel();self.live_details.setWordWrap(True);live_l.addWidget(self.live_details)
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(100);layout.addWidget(self.log)
        settings_path=self.workspace/'ui_settings.json'
        if settings_path.exists():
            settings=json.loads(settings_path.read_text())
            for k,v in settings.items():
                if k in self.fields:self.fields[k].setText(v)
            choices=settings.get('real_sim',{})
            for key,widget in [('use_precision',self.use_precision),('identity_sim_for_validation',self.identity_sim),('use_workspace',self.live_workspace)]:
                if isinstance(choices.get(key),bool):widget.setChecked(choices[key])
        from ..workflow.ui import WorkflowPanel
        self.workflow_panel=WorkflowPanel(self.work,self)
        self.live_backend.setCurrentIndex(self.live_backend.findData(self.workflow_panel.backend.currentData()))
        self.live_backend.currentIndexChanged.connect(lambda i:self.workflow_panel.backend.setCurrentIndex(i))
        self.workflow_panel.backend.currentIndexChanged.connect(lambda i:self.live_backend.setCurrentIndex(i))
        self.tabs.addTab(self.workflow_panel,'3 · Workspace Pipeline')
        self.live_scene.currentIndexChanged.connect(lambda:self.guard(self.select_live_scene))
        self.workflow_panel.workspaceChanged.connect(self.refresh_live_scenes)
        self.refresh_live_scenes()
        for widget in (self.live_workspace,self.use_precision,self.identity_sim):
            widget.toggled.connect(self.save_settings)
            widget.toggled.connect(self.update_scene_selection_hint)
        from .precision_ui import PrecisionPanel
        self.precision_panel=PrecisionPanel(self);self.tabs.addTab(self.precision_panel,'4 · Precision Calibration')
        self.tabs.currentChanged.connect(self.change_tab);self.change_tab(0)
        self.timer=QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(200);self.refresh()

    def guard(self,fn):
        try:return fn()
        except Exception as exc:
            self.log.appendPlainText(str(exc));QMessageBox.warning(self,'확인 필요',str(exc))

    def button(self,row,title,fn):
        button=QPushButton(title);button.clicked.connect(lambda checked=False:self.guard(fn));row.addWidget(button);return button

    def refresh_live_scenes(self):
        """List the existing registry; selecting a Scene never starts a worker."""
        path=self.workflow_panel.workspace_path.strip()
        self.live_scene.blockSignals(True);self.live_scene.clear()
        self.live_scene.addItem('Scene을 선택하세요', '')
        seen=set()
        for artifact in self.workflow_panel.registry.all():
            scene=artifact['path']
            if artifact['type']!='workspace' or artifact.get('status','SUCCEEDED')!='SUCCEEDED' or not Path(scene).exists() or scene in seen:continue
            seen.add(scene)
            folder=Path(scene)
            self.live_scene.addItem(folder.parent.name+'/'+folder.name,scene)
            self.live_scene.setItemData(self.live_scene.count()-1,scene,Qt.ItemDataRole.ToolTipRole)
        if path and path not in seen:
            self.live_scene.addItem(Path(path).parent.name+'/'+Path(path).name,path)
        self.live_scene.setCurrentIndex(max(0,self.live_scene.findData(path)))
        self.live_scene.blockSignals(False)
        self.update_scene_selection_hint()
        if self.tabs.currentIndex()==1 and not self.runtime_enabled:self.sync_scene_cameras()

    def select_live_scene(self):
        path=self.live_scene.currentData()
        if path:
            try:self.workflow_panel.select_workspace(path)
            except Exception:
                self.refresh_live_scenes()
                raise
        else:
            self.refresh_live_scenes()
        self.update_scene_selection_hint()

    def update_scene_selection_hint(self):
        path=self.workflow_panel.workspace_path.strip()
        self.live_scene.setToolTip(path)
        selected=(Path(path).parent.name+'/'+Path(path).name if path else '없음') if self.live_workspace.isChecked() else '연결 설정의 Profile 사용'
        self.scene_selection_hint.setText('선택: '+selected+'\n연결 중 변경했다면 Disconnect → Connect하세요.')

    def save_settings(self):
        atomic(self.workspace/'ui_settings.json',{**{k:v.text() for k,v in self.fields.items()},
            'real_sim':{'use_precision':self.use_precision.isChecked(),
                        'identity_sim_for_validation':self.identity_sim.isChecked(),
                        'use_workspace':self.live_workspace.isChecked()}})

    def sync_scene_cameras(self):
        path = self.workflow_panel.workspace_path.strip()
        if not path or path == getattr(self,'_live_scene_path',None): return
        from .workspaces.package import load as load_workspace
        try: workspace = load_workspace(path)
        except (ValueError,OSError,KeyError): return
        if 'profile' not in workspace: return
        names = list(workspace['profile']['cameras'])
        self._live_scene_path = path
        for i,name in enumerate(names):
            for prefix in ('real_','sim_'):
                key = prefix+name
                if key not in self.views:
                    tile=QWidget();column=QVBoxLayout(tile);column.addWidget(QLabel(key.upper().replace('_',' ')))
                    image=CameraView(minimum=(240,160));image.setText('No live frame');column.addWidget(image,1)
                    self.views[key]=image;self.live_tiles[key]=tile
        for key,tile in self.live_tiles.items():
            tile.setVisible(key.removeprefix('real_').removeprefix('sim_') in names)
            self.live_grid.removeWidget(tile)
        for i,name in enumerate(names):
            for col,prefix in enumerate(('real_','sim_')):
                tile=self.live_tiles[prefix+name]
                self.live_grid.addWidget(tile,i,col)
                tile.setVisible(True)
        bindings = workspace['task'].get('real_camera_bindings', {})
        for name in names:
            key=bindings.get(name,name+'_camera')
            if key not in self.fields:
                settings_path=self.workspace/'ui_settings.json'
                saved=json.loads(settings_path.read_text()) if settings_path.exists() else {}
                edit=QLineEdit(saved.get(key,''));self.fields[key]=edit
                self.connection_form.addRow(name+' camera index/path',edit)
            self.fields[key].setToolTip('Metadata camera ID: '+name+'; select the actual connected device')

    def profile_path(self):
        path=self.fields['profile'].text().strip()
        if not path:raise ValueError('연결 설정에서 Profile을 선택하세요')
        return str(Path(path).resolve())

    def selected(self,path):
        load(path);self.fields['profile'].setText(str(Path(path).resolve()))

    def open_profile(self):
        from ..workflow.artifacts import browse_directory
        path,_=QFileDialog.getOpenFileName(self,'Profile / camera calibration',browse_directory('profile'),'JSON (*.json)')
        if path:self.selected(path)

    def choose_capture_output(self):
        path=QFileDialog.getExistingDirectory(self,'Snapshot 저장 위치',self.capture_output.text())
        if path:self.capture_output.setText(path)

    def capture_snapshot(self):
        self.send('capture_snapshot',{'output':self.capture_output.text(),'name':self.capture_name.text(),'scene_state':self.capture_scene_state.text().strip()})

    def capture_pair(self):
        self.send('capture_pair',{'output':self.capture_output.text(),'name':self.capture_name.text(),
            'scene_state':self.capture_scene_state.text().strip()})
        self.capture_result.setText('현재 observation 확보 및 저장 중…')

    def send(self,name,payload=None):
        if not self.runtime_enabled or self.runtime_session != self.bus.status().get('session'):
            raise RuntimeError('Capture의 Start 또는 Real/Sim의 Connect로 먼저 연결하세요.')
        identifier=self.bus.send(name,payload);self.pending[identifier]=name;return identifier

    def change_tab(self,index):
        if index == 1 and hasattr(self, 'workflow_panel'): self.refresh_live_scenes()
        self.log.setVisible(index!=2)
        self.settings_scroll.setVisible(index!=2)
        self.scene_controls.setVisible(index==1)
        self.scene_options.setVisible(index in (0,1))
        self.capture_options.setVisible(index in (0,1))
        self.sim_options.setVisible(index==1);self.sim_preview_button.setVisible(index==1)

    def start_capture(self):
        self.connect_hw(backend='real')

    def connect_precision(self, payload):
        if self.workflow_panel.busy: raise ValueError('Workflow을 중지하고 장치를 해제하세요.')
        s=self.bus.status()
        if self.bus.owner_alive() and s.get('backend')!='calibration':
            raise ValueError('먼저 Capture/Real-Sim에서 Disconnect하세요. Calibration은 Teleop과 장치를 공유하지 않습니다.')
        if self.runtime_enabled and self.runtime_session==s.get('session') and s.get('backend')=='calibration':
            return self.send('precision.connect',payload)
        identifier=self.runtime_launcher.connect({'precision':payload},self.workflow_panel.worker_python.strip(),backend='calibration',headless=True)
        self.runtime_enabled=True;self.requested_backend='calibration';self.runtime_session=s.get('session') if identifier else None
        if identifier:self.pending[identifier]='connect'

    def stop(self):
        self.cancel.set()
        if self.runtime_launcher.pending is not None or self.runtime_launcher.replacement is not None:
            self.disconnect_sim();return
        self.send('stop')

    def connect_hw(self,preview=False,backend=None):
        if self.workflow_panel.busy:raise ValueError('Workflow 실행을 중지한 뒤 연결하세요. 중복 장치 연결을 방지합니다.')
        if self.bus.owner_alive() and self.bus.status().get('backend')=='calibration':
            raise ValueError('Precision 장치를 Disconnect한 뒤 Runtime을 Disconnect하세요. 중복 포트 연결을 하지 않습니다.')
        if backend != 'real': self.sync_scene_cameras()
        hardware={k:v.text().strip() for k,v in self.fields.items() if k!='profile'}
        payload={'hardware':hardware,'preview_only':preview,'real_camera_preview':self.live_real_preview.isChecked()}
        backend=backend or self.live_backend.currentData()
        profile=None
        if backend != 'real' and self.live_workspace.isChecked() and self.workflow_panel.workspace_path.strip():
            from .workspaces.package import load as load_workspace
            w=load_workspace(self.workflow_panel.workspace_path);profile=str(w['root']/'profile.json');payload['workspace']=str(w['root'])
            selected=self.workflow_panel.task_selector.selection()
            if selected:payload['task_definition']=selected[0]
        payload['profile']=profile or self.profile_path()
        payload['precision_options']={'use_precision':self.use_precision.isChecked(),
                                      'identity_sim_for_validation':self.identity_sim.isChecked()}
        renderer=self.live_renderer.currentText()
        if backend=='mujoco':
            from .launcher import resolve_live_renderer
            resolved=resolve_live_renderer(renderer)
            if resolved!=renderer:
                self.log.appendPlainText(f'{renderer} 사용 불가 · {resolved}로 렌더링합니다.')
                self.live_renderer.setCurrentText(resolved)
            renderer=resolved
        identifier=self.runtime_launcher.connect(payload,self.workflow_panel.worker_python.strip(),headless=self.workflow_panel.mode.currentData()=='operator',backend=backend,renderer=renderer)
        # A superseded connection may never reply after its worker is cancelled.
        self.pending={key:value for key,value in self.pending.items() if value!='connect'}
        self.runtime_enabled=True
        self.requested_backend=backend
        self.runtime_session=self.bus.status().get('session') if identifier else None
        if identifier:self.pending[identifier]='connect'
        self.save_settings()

    def disconnect_sim(self):
        identifier=self.runtime_launcher.disconnect()
        self.pending={key:value for key,value in self.pending.items() if value!='connect'}
        if identifier:self.pending[identifier]='disconnect'
        self.runtime_enabled=False
        self.runtime_session=None
        if self.tabs.currentIndex()==1:self.sync_scene_cameras()

    def refresh(self):
        if self.runtime_enabled and self.runtime_session == self.bus.status().get('session'):
            atomic(self.bus.root/'ui_heartbeat.json',{'time':time.time()})
        try:
            identifier=self.runtime_launcher.poll()
            if identifier:
                self.pending[identifier]='connect'
                self.runtime_session=self.bus.status().get('session')
        except Exception as exc:self.log.appendPlainText(str(exc))
        s=self.bus.status()
        starting=self.runtime_launcher.pending is not None or self.runtime_launcher.replacement is not None or self.runtime_launcher.stopping
        connecting=starting or 'connect' in self.pending.values()
        backend=self.requested_backend if starting else s.get('backend')
        # Choice is for the next explicit Connect; old status must not override or lock it.
        self.live_backend.setEnabled(not starting and not self.workflow_panel.busy)
        self.workflow_panel.backend.setEnabled(not starting and not self.workflow_panel.busy);self.live_renderer.setEnabled(not starting)
        for widget in (self.live_scene,self.scene_browse,self.scene_refresh,self.live_workspace,self.use_precision,self.identity_sim):
            widget.setEnabled(not starting and not self.workflow_panel.busy)
        fresh=bool(self.runtime_enabled and self.runtime_session == s.get('session') and self.bus.responsive(s))
        self.precision_panel.refresh(s.get('precision') if fresh else {})
        availability=s.get('availability',{}) if fresh else {}
        data=(s.get('snapshot_dataset') or {}) if fresh else {};count=data.get('count',0)
        session=(s.get('snapshot_session') or {}) if fresh else {}
        self.capture_session_label.setText('세션: '+str(data.get('root') or session.get('id') or 'Start 후 첫 캡처부터 같은 폴더에 저장합니다.'))
        for widget in (self.capture_output,self.capture_output_browse,self.capture_name):
            widget.setEnabled(not bool(data))
        connected=bool(fresh and s.get('real_connected'))
        side=connected and availability.get('real_side')=='frame received';wrist=connected and availability.get('real_wrist')=='frame received'
        feedback=connected and availability.get('robot_state')
        self.capture_start.setEnabled(not connecting and not self.workflow_panel.busy and not (connected and backend=='real'))
        for button in self.teleop_start_buttons:
            button.setEnabled(bool(connected and not connecting and s.get('precision_teleop_ready',True)))
        self.capture_readiness.setText(f"Robot: {'Connected' if connected else 'Disconnected'} · Side: {'Connected' if side else 'Disconnected'} · Wrist: {'Connected' if wrist else 'Disconnected'}\nJoint Feedback: {'Receiving' if feedback else 'Unavailable'} · Captured Snapshots: {count}")
        self.capture_button.setEnabled(bool(feedback and side and wrist and not connecting and 'capture_snapshot' not in self.pending.values()))
        self.pair_button.setEnabled(bool(feedback and side and wrist and s.get('sim_connected') and not connecting
            and 'capture_pair' not in self.pending.values()))
        self.status.setText(f"{backend or 'runtime'} · {s.get('state')} · {s.get('message','')}\n{s.get('error') or ''}" if fresh else '대기 · Capture Start 또는 Real/Sim Connect를 눌러 시작하세요.')
        options=s.get('precision_run_configuration',{}) if fresh else {}
        current_scene=s.get('workspace') if fresh else None
        current_options=(f"연결된 설정 · 저장 관절 보정: {'ON' if options.get('use_precision') else 'OFF'} · Scene 관절 보정 제외: {'ON' if options.get('identity_sim_for_validation') else 'OFF'}"
            if fresh else '연결된 설정: 없음 (아직 Connect하지 않음)')
        self.live_details.setText(f"연결된 Scene: {current_scene or ('Profile 사용' if fresh else '없음')}\n"
            f"{current_options}\n"
            f"적용 profile hash: {s.get('profile_hash') if fresh else '없음'} · Real: {connected} / Sim: {bool(fresh and s.get('sim_connected'))}")
        if self.runtime_launcher.pending is not None:self.status.setText('Runtime 시작 중 · 로그: '+str(self.runtime_launcher.log_path))
        elif self.runtime_launcher.stopping:self.status.setText('이전 Runtime 종료 중 · 장치 재연결은 종료 후에 진행합니다.')
        elif not self.runtime_enabled and self.bus.owner_alive():
            self.status.setText('대기 · 이전 '+str(s.get('backend'))+' Runtime이 남아 있습니다. Start/Connect 시 모드를 전환하거나 Disconnect로 종료하세요.')
        self.snapshot_list.clear()
        for item in s.get('snapshot_results',[]) if fresh else []:self.snapshot_list.addItem(item['id']+' — '+item['status']+' · '+item['path'])
        for name,label in [*self.views.items(),*self.capture_views.items()]:
            path=self.bus.root/(name+'.jpg');ready=(s.get('real_camera_connected') if name.startswith('real_') else s.get('sim_connected'))
            if fresh and ready and path.exists() and time.time()-path.stat().st_mtime<2:label.show_image(path)
            else:label.clear();label.setText('No live frame')
        for identifier,name in list(self.pending.items()):
            path=self.bus.root/('reply_'+identifier+'.json')
            if not path.exists():continue
            reply=json.loads(path.read_text());path.unlink();del self.pending[identifier]
            if name.startswith('precision.'):
                self.precision_panel.reply(name,reply)
                self.log.appendPlainText(name+': '+self.precision_panel.operation_status.text())
            else:self.log.appendPlainText(name+': '+json.dumps(reply,ensure_ascii=False))
            if name in ('capture_snapshot','capture_pair'):
                if reply.get('ok'):
                    result=reply['result'];self.capture_result.setText(f"{result['id']} — {result['status']} · {result['timestamp_ns']} ns\n{result['path']}\n"+'; '.join(result['warnings']))
                else:self.capture_result.setText('Capture 실패 · 저장 완료로 등록하지 않음 · '+str(reply.get('error',reply.get('result')))+' · 정지 자세에서 다시 캡처하세요.')
                if name=='capture_pair':self.pair_result.setText(self.capture_result.text())

    def work(self,fn,done=None):
        if self.worker and self.worker.isRunning():raise ValueError('다른 작업이 진행 중입니다')
        self.cancel.clear();self.worker=Worker(fn);self.worker.progress.connect(self.log.appendPlainText)
        self.worker.failed.connect(lambda s:self.log.appendPlainText('ERROR: '+s));self.worker.done.connect(done or (lambda r:self.log.appendPlainText(json.dumps(r,ensure_ascii=False,indent=2))));self.worker.start()

    def closeEvent(self,event):
        self.cancel.set();self.workflow_panel.service.cancel()
        try:
            if self.runtime_launcher.pending is not None:self.runtime_launcher.disconnect()
            elif self.runtime_enabled:
                if self.requested_backend in ('real','calibration'):self.runtime_launcher.disconnect()
                else:self.bus.send('stop')
        except RuntimeError:pass
        self.runtime_launcher.cancel_connect()
        if self.worker and self.worker.isRunning():event.ignore();return
        self.save_settings();event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(ROOT / "outputs/real2sim"))
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    from ..workflow.theme import STYLE
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    w = Window(args.workspace)
    w.show()
    sys.exit(app.exec())
