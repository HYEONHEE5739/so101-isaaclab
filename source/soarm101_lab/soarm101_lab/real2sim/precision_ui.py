"""Precision calibration controls. All device operations use the existing worker IPC."""
import json
import math
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QLineEdit, QPlainTextEdit, QMessageBox, QScrollArea, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView)
from soarm101_lab.so101_joint_mapping import JOINTS, LEGACY_MAPPER
from .precision import unwrap, joint_angle, joint_raw_coordinate
from ..workflow.view import CameraView


class PrecisionPanel(QWidget):
    def __init__(self, window):
        super().__init__(window); self.window = window; self.latest = {}; self.buttons = {}; self.review_key = None; self.pending_exclusion = False
        layout = QVBoxLayout(self); self.banner = QLabel('장치 연결 대기 · 이 탭은 Sim/모터를 자동 시작하지 않습니다.'); self.banner.setWordWrap(True); layout.addWidget(self.banner)
        content = QWidget(); body = QVBoxLayout(content); scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(content); layout.addWidget(scroll)
        form = QFormLayout(); body.addLayout(form)
        self.robot = QComboBox(); self.robot.addItems(['follower', 'leader']); form.addRow('보정할 로봇', self.robot)
        self.joint = QComboBox(); self.joint.addItems(JOINTS); form.addRow('보정할 관절', self.joint)
        row = QHBoxLayout(); body.addLayout(row)
        self.add_button(row, 'connect', '1. 연결 · 위치만 읽기', self.connect_device)
        self.add_button(row, 'disconnect', '선택 로봇 연결 종료', lambda:self.send('disconnect'))
        self.add_button(row, 'end_session', '보정 작업 종료', self.window.disconnect_sim)
        self.raw_value = QLabel(); self.raw_value.setWordWrap(True); body.addWidget(self.raw_value)
        self.applied_angle = QLabel(); self.applied_angle.setWordWrap(True)
        self.applied_angle.setStyleSheet('font-size: 22px; font-weight: bold; padding: 8px;')
        body.addWidget(self.applied_angle)
        self.difference = QLabel(); self.difference.setWordWrap(True)
        self.live_reference_error = QLabel(); self.live_reference_error.setWordWrap(True)
        self.current_angle = QLabel(); self.current_angle.setWordWrap(True)
        self.candidate_angle = QLabel(); self.candidate_angle.setWordWrap(True)
        for label in (self.current_angle, self.candidate_angle):
            label.setStyleSheet('font-size: 15px; padding: 4px;')
            body.addWidget(label)
        body.addWidget(self.difference); body.addWidget(self.live_reference_error)
        self.angle_help = QLabel('각도는 모터값을 변환한 값입니다. 입력한 수동 기준각으로 보정하며 외부 각도계 측정값은 아닙니다.')
        self.angle_help.setWordWrap(True); body.addWidget(self.angle_help)
        self.torque_label = QLabel(); body.addWidget(self.torque_label)
        self.support = QCheckBox('로봇을 손/지그로 지지했고 중력 낙하 위험을 확인했습니다.'); body.addWidget(self.support)
        row = QHBoxLayout(); body.addLayout(row)
        self.add_button(row, 'release', '2. 선택 관절 힘 풀기 · 손으로 움직이기', self.release)
        refs = QFormLayout(); body.addLayout(refs)
        self.angle = QDoubleSpinBox(); self.angle.setRange(-360, 360); self.angle.setDecimals(3); self.angle.setSuffix(' °')
        refs.addRow('손으로 맞춘 실제 기준각', self.angle)
        help_text = QLabel('손으로 기준 자세에 맞추고 각도를 입력해 지금 자세를 기록하세요. 모든 새 기록을 보정 계산에 사용합니다.\n'
                           '보정 계산 후 자세 검증을 누르고 손으로 관절을 움직이면 보정 전·후 각도를 확인할 수 있습니다.')
        help_text.setWordWrap(True); body.addWidget(help_text)
        row = QHBoxLayout(); body.addLayout(row); self.add_button(row, 'record', '3. 지금 자세 기록', self.record)
        self.reference_table = QTableWidget(0, 6)
        self.reference_table.setHorizontalHeaderLabels(['입력 기준각', '모터 숫자', '보정 전 각도', '보정 전 − 기준', '보정 후 각도', '보정 후 − 기준'])
        self.reference_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.reference_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.reference_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.reference_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.reference_table.setMaximumHeight(180); body.addWidget(self.reference_table)
        self.raw_gap = QLabel(); self.raw_gap.setWordWrap(True); body.addWidget(self.raw_gap)
        row = QHBoxLayout(); body.addLayout(row)
        self.add_button(row, 'exclude_reference', '선택 자세 삭제', self.exclude_reference)
        self.reference_table.itemSelectionChanged.connect(self.update_exclude_button)
        row = QHBoxLayout(); body.addLayout(row)
        self.add_button(row, 'fit', '4. 보정 계산', self.fit)
        self.add_button(row, 'review', '5. 자세 검증 · 손으로 움직여 확인', self.start_review)
        self.validation_summary = QLabel(); self.validation_summary.setWordWrap(True); body.addWidget(self.validation_summary)
        self.next_step = QLabel(); self.next_step.setWordWrap(True); body.addWidget(self.next_step)
        self.operation_status = QLabel(); self.operation_status.setWordWrap(True); body.addWidget(self.operation_status)
        row = QHBoxLayout(); body.addLayout(row)
        self.add_button(row, 'save', '6. 보정 파일 저장', lambda:self.send('save'))
        self.add_button(row, 'activate', '7. 확인한 보정 적용', self.activate)
        self.saved_path = QLineEdit(); self.saved_path.setReadOnly(True)
        saved_form = QFormLayout(); saved_form.addRow('저장된 보정 파일', self.saved_path); body.addLayout(saved_form)
        preview = QGroupBox('가상 로봇 그림 보기 · 선택 사항'); preview.setCheckable(True); preview.setChecked(False); body.addWidget(preview)
        preview_content = QWidget(); preview_layout = QVBoxLayout(preview); preview_layout.addWidget(preview_content)
        preview_body = QVBoxLayout(preview_content)
        self.renderer=QComboBox();self.renderer.addItems(['egl','software-egl']);preview_body.addWidget(self.renderer)
        row = QHBoxLayout(); preview_body.addLayout(row)
        self.add_button(row, 'preview', '현재 적용 자세 / 입력한 기준 자세 보기', lambda:self.send('preview', angle_rad=math.radians(self.angle.value()), renderer=self.renderer.currentText()))
        images = QHBoxLayout(); preview_body.addLayout(images); self.views = {}
        for name, title in (('current', '현재 적용 변환의 자세'), ('reference', '입력한 기준각의 자세')):
            column = QVBoxLayout(); column.addWidget(QLabel(title)); view = CameraView(minimum=(200,150)); column.addWidget(view); images.addLayout(column); self.views[name] = view
        preview.toggled.connect(preview_content.setVisible); preview_content.setVisible(False)
        self.advanced = QGroupBox('상세 정보 / 시험·복구 설정'); self.advanced.setCheckable(True); self.advanced.setChecked(False); body.addWidget(self.advanced)
        advanced_content = QWidget(); advanced_layout = QVBoxLayout(self.advanced); advanced_layout.addWidget(advanced_content)
        advanced_body = QVBoxLayout(advanced_content)
        advanced_form = QFormLayout(); advanced_body.addLayout(advanced_form)
        self.tolerance = QDoubleSpinBox(); self.tolerance.setRange(.1, 5); self.tolerance.setValue(.5); self.tolerance.setSuffix(' °')
        advanced_form.addRow('기록 재현 오차 · 기본 0.5° (3점 이상에서 유효)', self.tolerance)
        self.fake = QCheckBox('Fake Motor · 실제 장치 연결 없음'); advanced_form.addRow(self.fake)
        self.fake_raw = QSpinBox(); self.fake_raw.setRange(-8192, 8192); self.fake_raw.setValue(2048); advanced_form.addRow('시험용 모터 숫자', self.fake_raw)
        row = QHBoxLayout(); advanced_body.addLayout(row)
        self.add_button(row, 'fake_raw', '시험용 위치 변경', lambda:self.send('fake_raw', raw=self.fake_raw.value()))
        self.add_button(row, 'compare', '기존 / 새 보정 상세 비교', lambda:self.send('compare'))
        self.add_button(row, 'restore', '이전 보정으로 복구', self.restore)
        self.add_button(row, 'compare_robots', 'Leader / Follower 정적 비교', self.compare_robots)
        self.identity = QLabel(); self.identity.setWordWrap(True); advanced_body.addWidget(self.identity)
        self.axis = QLabel(); self.axis.setWordWrap(True); advanced_body.addWidget(self.axis)
        self.values = QPlainTextEdit(); self.values.setReadOnly(True); self.values.setMaximumHeight(210); advanced_body.addWidget(self.values)
        self.reference_list = QPlainTextEdit(); self.reference_list.setReadOnly(True); self.reference_list.setMaximumHeight(200); advanced_body.addWidget(self.reference_list)
        self.result = QPlainTextEdit(); self.result.setReadOnly(True); self.result.setMaximumHeight(180); advanced_body.addWidget(self.result)
        self.advanced.toggled.connect(advanced_content.setVisible); advanced_content.setVisible(False)
        self.robot.currentTextChanged.connect(self.select); self.joint.currentTextChanged.connect(self.select)
        self.fake.toggled.connect(self.fake_raw.setEnabled); self.fake_raw.setEnabled(False)
        self.refresh({})

    def add_button(self, row, name, label, callback):
        button = QPushButton(label); row.addWidget(button); self.buttons[name] = button
        button.clicked.connect(lambda checked=False:self.window.guard(callback))

    def send(self, name, **payload):
        return self.window.send('precision.' + name, {'role': self.robot.currentText(), 'joint': self.joint.currentText(), **payload})

    def select(self):
        self.saved_path.clear(); self.operation_status.clear(); self.review_key = None
        self.reference_table.clearSelection()
        if self.window.runtime_enabled and self.window.runtime_session==self.window.bus.status().get('session') and self.window.bus.status().get('backend') == 'calibration': self.send('select')
        self.refresh(self.latest)

    def connect_device(self):
        self.window.connect_precision({'role':self.robot.currentText(), 'fake':self.fake.isChecked(),
                                       'hardware':{k:v.text().strip() for k,v in self.window.fields.items() if k!='profile'}})
        self.send_selection_pending = True

    def approved(self, title, text):
        return QMessageBox.question(self, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def release(self):
        if not self.support.isChecked(): raise ValueError('먼저 로봇을 지지하고 안전 확인을 체크하세요.')
        if self.approved('Torque release', '선택한 관절의 토크를 해제합니다. 중력으로 떨어질 수 있습니다. 로봇을 지지했습니까?'):
            self.send('release', supported_and_approved=True)

    def record(self):
        if not self.support.isChecked(): raise ValueError('토크 해제와 수동 지지 상태를 확인하세요.')
        self.saved_path.clear(); self.review_key = None
        self.send('record', angle_rad=math.radians(self.angle.value()))

    def selected_reference_id(self):
        if not self.reference_table.selectedItems(): return None
        item = self.reference_table.item(self.reference_table.currentRow(), 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def update_exclude_button(self):
        connected = self.latest.get('robots',{}).get(self.robot.currentText(),{}).get('connected')
        self.buttons['exclude_reference'].setEnabled(bool(connected and self.selected_reference_id() and not self.pending_exclusion))

    def exclude_reference(self):
        identifier = self.selected_reference_id()
        if not identifier: raise ValueError('표에서 삭제할 자세를 먼저 선택하세요.')
        if self.approved('선택 자세 삭제', '선택한 자세를 보정 계산에서 제외할까요? 원본 측정은 이력에 보관합니다. 기존 적용 보정은 유지되며 다시 계산·저장·적용해야 합니다.'):
            self.send('exclude_reference', reference_id=identifier, approved=True, reason='사용자가 잘못된 기준 자세로 판단하여 제외')
            self.pending_exclusion = True
            self.saved_path.clear(); self.review_key = None
            self.update_exclude_button()

    def fit(self):
        self.saved_path.clear(); self.review_key = None
        self.send('fit')

    def start_review(self):
        self.saved_path.clear()
        self.review_key = (self.robot.currentText(), self.joint.currentText())
        self.send('review', tolerance_deg=self.tolerance.value())
        self.refresh(self.latest)

    def activate(self):
        if not self.saved_path.text(): raise ValueError('Save Candidate를 먼저 실행하세요.')
        if self.approved('Software profile activation', '기존 LeRobot 파일/레지스터는 유지하고 소프트웨어 Profile만 활성화합니다. 다음 Real/Sim Connect에 명시적으로 적용하시겠습니까?'):
            self.send('activate', path=self.saved_path.text(), approved=True)

    def restore(self):
        if self.approved('Restore previous profile', '이 로봇의 이전 소프트웨어 Profile로 복구합니까? 기존 모터 Calibration은 변경하지 않습니다.'):
            self.send('restore', approved=True)

    def compare_robots(self):
        if self.approved('Static comparison', '사용자가 두 로봇의 선택 관절을 같은 실제 물리적 자세에 정지시켰습니까? 동적 추종 검사와 다릅니다.'):
            self.send('compare_robots', same_pose_confirmed=True)

    def reply(self, name, reply):
        self.result.setPlainText(json.dumps(reply, ensure_ascii=False, indent=2))
        if name == 'precision.exclude_reference':
            self.pending_exclusion = False
            if reply.get('ok'): self.saved_path.clear(); self.review_key = None
            self.update_exclude_button()
        if not reply.get('ok'):
            if name == 'precision.review': self.review_key = None
            self.operation_status.setText('실행 실패: ' + self.explain_error(str(reply.get('result', reply))))
            return
        actions = {'record':'자세를 기록했습니다','fit':'새 보정을 계산했습니다','review':'손으로 관절을 움직이며 현재 모터값과 보정 전·후 각도를 확인하세요. 로봇 이동 명령은 보내지 않습니다.',
                   'exclude_reference':'선택 자세를 계산에서 삭제했습니다. 올바른 자세를 다시 기록한 뒤 보정 계산을 누르세요. 기존 적용 보정은 유지됩니다.',
                   'save':'보정 파일을 저장했습니다','activate':'보정을 활성화했습니다. Real/Sim은 다음 Connect에 적용됩니다.',
                   'restore':'이전 보정으로 복구했습니다','compare':'비교했습니다. 현재각은 위에서 확인하세요.'}
        self.operation_status.setText(actions.get(name.removeprefix('precision.'), '실행 완료'))
        result = reply.get('result')
        if name == 'precision.save': self.saved_path.setText(result['path'])
        if name == 'precision.activate' and not self.fake.isChecked():
            self.window.use_precision.setChecked(True);self.window.identity_sim.setChecked(True)
        if name == 'precision.preview':
            for n,v in self.views.items(): v.show_image(Path(result[n]))

    @staticmethod
    def explain_error(message):
        explanations = {
            'Record a reference first': '먼저 기준 자세를 기록하세요.',
            'References are not distinct enough': '보정 계산용 두 기준각이 너무 가깝습니다. 각도 차이 5° 이상인 기준을 사용하세요.',
            'Wrap branch spans half-turn; ambiguous': '기록 범위가 모터 반 바퀴 이상이라 회전 구간을 구별할 수 없습니다. 더 가까운 기준 자세로 새 기록을 시작하세요.',
            'Ambiguous half-turn encoder wrap; add closer references': '기준과 반 바퀴 떨어진 모터값은 회전 방향이 불명확합니다. 더 가까운 기준 자세가 필요합니다.',
            'Reference exceeds URDF limits': '입력한 기준각이 이 관절의 URDF 허용 범위를 벗어났습니다.',
        }
        for raw, explanation in explanations.items():
            if raw in message: return explanation
        return message

    @staticmethod
    def degrees(value):
        if value is None: return '측정 없음'
        degree = math.degrees(value)
        return f'{0. if abs(degree)<.005 else degree:+.2f}°'

    @staticmethod
    def candidate_value(entry, raw):
        if not entry: return None, '아직 보정을 계산하지 않았습니다'
        if raw is None: return None, '선택 관절 측정 대기'
        try:
            x = joint_raw_coordinate(entry, raw)
            outside = not entry['raw_domain'][0]-3 <= x <= entry['raw_domain'][1]+3
            return joint_angle(entry, raw), '기록 범위 밖 · 보정식으로 계산' if outside else ''
        except (ValueError, KeyError) as exc:
            reason = str(exc)
            if 'outside URDF' in reason: reason = 'URDF 허용 범위 밖'
            elif 'Ambiguous encoder branch' in reason: reason = 'URDF 범위 안에서 모터 회전 구간을 구별할 수 없음'
            return None, reason

    @staticmethod
    def before_value(sample, joint, active_entry=None):
        if active_entry:
            return PrecisionPanel.candidate_value(active_entry, sample.get('raw'))[0]
        if sample.get('legacy_urdf_angle_rad') is not None: return sample['legacy_urdf_angle_rad']
        if sample.get('legacy_position') is None: return None
        return LEGACY_MAPPER.calibrated_to_urdf({n: sample['legacy_position'] if n==joint else 0. for n in JOINTS})[joint]

    def show_summary(self, robot, sample, entry, joint):
        active_entry = (robot.get('active_profile') or {}).get('joints',{}).get(joint)
        current = sample.get('urdf_angle_rad') if 'urdf_angle_rad' in sample else (None if active_entry else self.before_value(sample,joint))
        before = current
        state = '저장 보정 적용' if active_entry else '이 관절은 기존 변환 사용 · 적용한 보정 없음'
        if not sample: state = '선택 관절 측정 대기'
        elif active_entry and current is None: state = '저장 보정 있음 · 현재값은 적용 범위 밖이거나 변환 불가'
        self.applied_angle.setText('현재 각도: ' + self.degrees(current) + '  (' + state + ')')
        self.raw_value.setText('현재 모터 숫자 (raw): ' + (str(sample['raw']) if sample.get('raw') is not None else '측정 대기'))
        self.current_angle.setText('보정 전 현재각 (현재 적용값): ' + self.degrees(before))
        display_entry = entry or active_entry
        if entry: q, reason = self.candidate_value(entry, sample.get('raw'))
        else: q, reason = current, ('현재값 변환 불가' if sample and current is None else '측정 대기' if not sample else '')
        applied = bool(display_entry and active_entry == display_entry and current is not None)
        label = '보정 후 현재각: '
        note = '(적용됨)' if applied else '(검토용 · 미적용)' if entry else '(새 계산 전 · 현재와 동일)'
        self.candidate_angle.setText(label + (self.degrees(q) if q is not None else reason)
                                    + '  ' + note
                                    + (' · ' + reason if q is not None and reason else ''))
        self.difference.setText('보정 전·후 차이: ' + (self.degrees(q-before) if q is not None and before is not None else '보정 계산 후 확인'))
        live = self.review_key == (self.robot.currentText(), joint)
        self.live_reference_error.setVisible(live)
        if live:
            self.live_reference_error.setText('입력 기준각 대비 차이: ' + (self.degrees(q-math.radians(self.angle.value())) if q is not None else '측정 대기')
                + f' / 허용 {self.tolerance.value():.2f}° · 손으로 입력한 기준 자세에 맞췄을 때 확인하세요.')
        self.torque_label.setText('관절 힘: ' + ('해제됨 · 손으로 지지하세요' if sample.get('torque')==0 else
                                             '켜짐' if sample.get('torque') is not None else '측정 대기'))
        references = robot.get('references',{}).get(joint,[])
        selected = self.selected_reference_id()
        self.reference_table.blockSignals(True)
        self.reference_table.setRowCount(len(references))
        for i, ref in enumerate(references):
            predicted, reason = self.candidate_value(display_entry, ref['raw'])
            original = self.before_value(ref, joint, active_entry)
            values = [self.degrees(ref['angle_rad']), str(ref['raw']), self.degrees(original),
                      self.degrees(original-ref['angle_rad']) if original is not None else '—',
                      self.degrees(predicted) if predicted is not None else ('—' if not display_entry else reason),
                      self.degrees(predicted-ref['angle_rad']) if predicted is not None else '—']
            for j, value in enumerate(values):
                item = QTableWidgetItem(value); item.setData(Qt.ItemDataRole.UserRole,ref.get('reference_id'))
                self.reference_table.setItem(i,j,item)
        row = next((i for i,ref in enumerate(references) if selected and ref.get('reference_id')==selected), None)
        if row is None:
            self.reference_table.clearSelection(); self.reference_table.setCurrentCell(-1,-1)
        else:self.reference_table.setCurrentCell(row,0); self.reference_table.selectRow(row)
        self.reference_table.blockSignals(False)
        train = [r for r in references if r['split']=='train']
        self.raw_gap.setVisible(len(train)>=2)
        if len(train)>=2:
            a,b = min(train,key=lambda r:r['angle_rad']), max(train,key=lambda r:r['angle_rad'])
            qa,qb = self.before_value(a,joint,active_entry),self.before_value(b,joint,active_entry)
            try:
                counts = unwrap(b['raw'],a['raw'],a['motor']['resolution'])-a['raw']
                if qa is None or qb is None or abs(qb-qa)<1e-8: raise ValueError('Baseline span unavailable')
                expected = (b['angle_rad']-a['angle_rad'])*counts/(qb-qa)
                self.raw_gap.setText(f'기준각 {self.degrees(a["angle_rad"])} → {self.degrees(b["angle_rad"])}: raw 차이 {counts:+.1f} / 현재 저장 비율의 예상 {expected:+.1f} counts · '
                    f'각도 간격 차이 {self.degrees((qb-qa)-(b["angle_rad"]-a["angle_rad"]))}. 영점만 바꿔서는 이 간격 차이를 고칠 수 없습니다.')
            except (KeyError,ValueError):self.raw_gap.setText('기준점 raw 간격: 기존 비율로 비교할 수 없는 범위 또는 자료입니다.')
        check = (entry or {}).get('review')
        if check:
            result = '허용 범위 안' if check['passed'] else '허용 오차 초과 · 적용할 수 없음'
            self.validation_summary.setText(f"기록한 기준각 재현: {result} · 최대 차이 {check['max_error_deg']:.2f}° / 허용 {check['tolerance_deg']:.2f}°. 독립 검증 결과가 아닙니다.")
            if len(train)==2:
                self.validation_summary.setText(self.validation_summary.text()+' 두 점은 그대로 맞추므로 raw 간격의 이상을 이 오차로 확인할 수 없습니다.')
            elif len(train)==1:
                self.validation_summary.setText(self.validation_summary.text()+' 한 점은 영점만 맞춘 값이며 방향·비율을 새로 확인한 결과가 아닙니다.')
        else:
            self.validation_summary.setText(f'기록 {len(references)}개 · 기준각 허용 오차 {self.tolerance.value():.2f}°. 자세 검증은 손으로 움직이며 각도를 확인하는 기능입니다.')
        saved_refs = (active_entry or {}).get('references',[])
        restored = bool(active_entry and len(references)==len(saved_refs) and
                        all((a['raw'],a['angle_rad'],a['split'])==(b['raw'],b['angle_rad'],b['split']) for a,b in zip(references,saved_refs)))
        if not robot.get('connected'): step = '1. 로봇과 관절을 선택하고 위치 읽기 연결을 누르세요.'
        elif not sample: step = '선택한 관절의 측정값을 기다리는 중입니다.'
        elif active_entry and (not references or restored) and not entry:
            step = ('저장된 보정을 불러와 사용 중입니다. 다시 기록·계산할 필요 없이 위의 현재 각도를 확인하세요.' if current is not None else
                    '저장 보정은 있지만 현재값을 적용할 수 없습니다. 표시된 범위/변환 상태를 확인하세요.')
        elif sample.get('torque')!=0: step = '2. 로봇을 지지한 뒤 선택 관절의 힘을 풀어 주세요.'
        elif not references: step = '3. 손으로 기준 자세에 맞추고 지금 자세 기록을 누르세요.'
        elif not entry: step = '4. 보정 계산을 누르세요. 기준각을 추가로 기록하면 방향·비율도 계산할 수 있습니다.'
        elif live: step = '손으로 관절을 움직이며 위의 모터 숫자와 보정 전·후 현재각을 확인하세요. 확인 후 저장·적용할 수 있습니다.'
        elif entry.get('status') in ('REVIEWED_MANUAL','VALIDATED_MANUAL'):
            step = '보정 저장 후 적용할 수 있습니다. 자세 검증을 누르면 다시 실시간으로 확인합니다.'
        else: step = '5. 자세 검증을 누르고 손으로 관절을 움직여 각도를 확인하세요.'
        self.next_step.setText('다음 할 일: ' + step)

    def refresh(self, status):
        self.latest = status or {}; role = self.robot.currentText(); joint = self.joint.currentText()
        robot = self.latest.get('robots', {}).get(role, {}); sample = robot.get('sample') or {}
        connected = bool(robot.get('connected')); active = robot.get('active_profile')
        if connected and getattr(self,'send_selection_pending',False):
            self.send_selection_pending=False
            if self.latest.get('selected_joint')!=joint or self.latest.get('selected_role')!=role:self.send('select')
        self.banner.setText(' · '.join(f"{r}: {'Connected' if v.get('connected') else 'Disconnected'} / Profile: {(v.get('active_profile') or {}).get('id','legacy')}" for r,v in self.latest.get('robots',{}).items()) or '장치 연결 대기 · 탭 전환은 장치를 연결하지 않습니다.')
        self.identity.setText('Device: '+json.dumps(robot.get('identity'),ensure_ascii=False))
        spec = self.latest.get('model',{}).get(joint)
        if spec:
            self.axis.setText(f"URDF: {spec['parent']} → {spec['child']} / local axis {spec['axis']} / origin xyz {spec['origin_xyz']} rpy {spec['origin_rpy']}\n각도는 이 Joint frame의 우수법칙입니다. 0°가 불분명하면 지그/각도계의 알려진 각도를 사용하세요. 그리퍼도 URDF 관절 기준각으로 기록합니다.")
        details={k:sample.get(k) for k in ('raw','legacy_position','urdf_angle_rad','estimated_zero_error_rad','zero_error_scope','torque_status','timestamp_ns','calibration_hash','calibration_status')}
        details['motor'] = sample.get('motor'); details['joint_limit_rad'] = spec.get('limits') if spec else None
        self.values.setPlainText(json.dumps(details,ensure_ascii=False,indent=2)+'\n'+self.latest.get('errors',{}).get(role,''))
        entry = robot.get('candidates',{}).get(joint)
        limits = spec.get('limits') if spec else (entry or (active or {}).get('joints',{}).get(joint) or {}).get('limits')
        self.angle_help.setText('각도는 수동 기준각으로 보정한 모터값이며 외부 각도계 측정값은 아닙니다.'
            + (f'\n각도 표시 범위 (URDF): {self.degrees(limits[0])} ~ {self.degrees(limits[1])}. 기록한 각도에서 끝나지 않습니다.' if limits else ''))
        self.show_summary(robot, sample if sample.get('joint')==joint else {}, entry, joint)
        self.reference_list.setPlainText(json.dumps({'references':robot.get('references',{}).get(joint,[]),
            'excluded_references':robot.get('excluded_references',[]),'candidate':entry,
            'active_profile_id':(active or {}).get('id'),'active_joint':(active or {}).get('joints',{}).get(joint)},ensure_ascii=False,indent=2))
        for name,button in self.buttons.items(): button.setEnabled(connected or name=='connect')
        self.buttons['end_session'].setEnabled(self.window.runtime_enabled and self.window.requested_backend=='calibration')
        self.buttons['fit'].setEnabled(connected and bool(robot.get('references',{}).get(joint)))
        for name in ('review', 'save', 'compare'): self.buttons[name].setEnabled(connected and bool(entry))
        self.buttons['fake_raw'].setEnabled(connected and self.fake.isChecked())
        self.buttons['record'].setEnabled(connected and sample.get('joint')==joint and sample.get('torque')==0)
        self.buttons['activate'].setEnabled(connected and bool(entry and entry.get('status') in ('VALIDATED_MANUAL','REVIEWED_MANUAL')) and bool(self.saved_path.text()))
        self.buttons['restore'].setEnabled(connected and bool(active or robot.get('active_pointer_exists')))
        self.update_exclude_button()
