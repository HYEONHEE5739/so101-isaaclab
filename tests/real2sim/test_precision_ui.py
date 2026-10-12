"""Offscreen UI + actual existing worker subprocess, using only explicit fake devices."""
import json
import math
import sys
import time
from pathlib import Path
import pytest
from PyQt6.QtWidgets import QApplication
from soarm101_lab.real2sim.ui import Window


def pump(app, window, predicate, timeout=15):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        app.processEvents();window.refresh()
        if predicate():return
        time.sleep(.02)
    log=window.runtime_launcher.log_path
    pytest.fail('UI/worker timed out: '+(log.read_text() if log.exists() else '')+'\n'+window.log.toPlainText())


def test_fake_ui_joint_reference_review_save_activate_restore(tmp_path,monkeypatch):
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    monkeypatch.setattr(ui,'Registry',lambda:Registry(tmp_path/'registry'))
    app=QApplication.instance() or QApplication([]);w=Window(tmp_path/'real2sim');p=w.precision_panel
    w.workflow_panel.python.setText(sys.executable);w.show();app.processEvents()
    assert w.tabs.count()==4 and w.runtime_launcher.process is None
    for index in (3,0,1,3):w.tabs.setCurrentIndex(index);app.processEvents()
    assert w.runtime_launcher.process is None  # Switching tabs must do nothing to devices.
    p.fake.setChecked(True);monkeypatch.setattr(p,'approved',lambda *args:True)
    def robot_state():return (w.bus.status().get('precision') or {}).get('robots',{}).get('follower',{})
    def sample_value(key):return (robot_state().get('sample') or {}).get(key)
    try:
        p.connect_device()
        pump(app,w,lambda:(w.bus.status().get('precision') or {}).get('robots',{}).get('follower',{}).get('sample') is not None)
        assert w.bus.status()['backend']=='calibration' and not w.bus.status()['sim_connected']
        p.support.setChecked(True);p.release()
        pump(app,w,lambda:sample_value('torque')==0)
        assert not hasattr(p, 'split') and p.tolerance.value()==.5
        for count,(raw,degree) in enumerate([(1000,-45),(2000,45),(1500,0)],1):
            p.fake_raw.setValue(raw);p.send('fake_raw',raw=raw)
            pump(app,w,lambda:sample_value('raw')==raw)
            p.angle.setValue(degree);p.record()
            pump(app,w,lambda:len(w.bus.status()['precision']['robots']['follower']['references'].get('shoulder_pan',[]))==count)
        p.send('fit');pump(app,w,lambda:bool(w.bus.status()['precision']['robots']['follower']['candidates']))
        assert 'raw 차이' in p.raw_gap.text() and '예상' in p.raw_gap.text()
        assert '미적용' in p.candidate_angle.text() and '+0.00°' in p.candidate_angle.text()
        assert p.reference_table.item(2,0).text()=='+0.00°'
        assert all(r['split']=='train' for r in robot_state()['references']['shoulder_pan'])
        assert 'timestamp_ns' not in w.log.toPlainText()  # Full reply stays in the collapsed detail view.
        p.start_review()
        pump(app,w,lambda:w.bus.status()['precision']['robots']['follower']['candidates']['shoulder_pan']['status']=='REVIEWED_MANUAL')
        assert '허용 범위 안' in p.validation_summary.text() and '0.50°' in p.validation_summary.text()
        # Move measured fake feedback only: no new reference, validation pose, or motor goal.
        p.send('fake_raw',raw=1600)
        pump(app,w,lambda:sample_value('raw')==1600)
        assert '+9.00°' in p.candidate_angle.text() and '1600' in p.raw_value.text()
        assert p.live_reference_error.isVisible() and len(robot_state()['references']['shoulder_pan'])==3
        p.send('fake_raw',raw=1500);pump(app,w,lambda:sample_value('raw')==1500)
        p.send('save');pump(app,w,lambda:bool(p.saved_path.text()))
        profile=json.loads(Path(p.saved_path.text()).read_text());assert list(profile['joints'])==['shoulder_pan']
        p.activate();pump(app,w,lambda:robot_state().get('active_profile') is not None and sample_value('calibration_status')=='REVIEWED_MANUAL')
        sample=w.bus.status()['precision']['robots']['follower']['sample']
        assert sample['urdf_angle_rad']==pytest.approx(0,abs=1e-12)
        assert '적용됨' in p.candidate_angle.text()
        # Actual worker and UI must display the applied angle beyond taught -45/+45.
        p.send('fake_raw',raw=2600);pump(app,w,lambda:sample_value('raw')==2600)
        assert sample_value('urdf_angle_rad')==pytest.approx(math.radians(99))
        assert '+99.00°' in p.current_angle.text() and '+99.00°' in p.applied_angle.text()
        assert '+99.00°' in p.candidate_angle.text() and '+0.00°' in p.difference.text()
        assert '-110.00° ~ +110.00°' in p.angle_help.text()
        assert not w.bus.status()['precision']['errors']
        p.send('fake_raw',raw=1500);pump(app,w,lambda:sample_value('raw')==1500)
        saved_path=p.saved_path.text();active_before=robot_state()['active_profile']
        p.reference_table.setCurrentCell(1,0);p.reference_table.selectRow(1)
        selected=p.selected_reference_id();assert selected
        p.refresh(w.bus.status()['precision']);assert p.selected_reference_id()==selected
        monkeypatch.setattr(p,'approved',lambda *args:False);p.exclude_reference()
        assert len(robot_state()['references']['shoulder_pan'])==3
        monkeypatch.setattr(p,'approved',lambda *args:True);p.exclude_reference()
        pump(app,w,lambda:len(robot_state()['references']['shoulder_pan'])==2 and not p.pending_exclusion)
        assert not robot_state()['candidates'] and not p.saved_path.text()
        assert not p.buttons['activate'].isEnabled() and p.selected_reference_id() is None
        assert robot_state()['active_profile']==active_before
        journal=next((tmp_path/'real2sim/joint_calibration/fake/follower/references').glob('*.json'))
        audit=json.loads(journal.read_text())
        assert len(audit['references']['shoulder_pan'])==3 and audit['exclusions'][0]['reference_id']==selected
        p.send('fake_raw',raw=2000);pump(app,w,lambda:sample_value('raw')==2000)
        p.angle.setValue(45);p.record()
        pump(app,w,lambda:len(robot_state()['references']['shoulder_pan'])==3)
        assert all(r['reference_id']!=selected for r in robot_state()['references']['shoulder_pan'])
        p.restore();pump(app,w,lambda:w.bus.status()['precision']['robots']['follower']['active_profile'] is None)
        assert Path(saved_path).exists()  # Exclusion/restoring never deletes saved profile evidence.
        # Gripper uses exactly the same angle-only controls and worker flow.
        p.joint.setCurrentText('gripper')
        pump(app,w,lambda:sample_value('joint')=='gripper')
        assert not hasattr(p,'kind') and not hasattr(p,'opening')
        p.release();pump(app,w,lambda:sample_value('torque')==0)
        for count,(raw,degree) in enumerate([(1000,0),(2000,45)],1):
            p.send('fake_raw',raw=raw);pump(app,w,lambda:sample_value('raw')==raw)
            p.angle.setValue(degree);p.record()
            pump(app,w,lambda:len(robot_state()['references'].get('gripper',[]))==count)
        assert all(r['reference_kind']=='angle' and 'opening_m' not in r for r in robot_state()['references']['gripper'])
        p.fit();pump(app,w,lambda:'gripper' in robot_state()['candidates'])
        p.start_review();pump(app,w,lambda:robot_state()['candidates']['gripper']['status']=='REVIEWED_MANUAL')
        p.send('save');pump(app,w,lambda:bool(p.saved_path.text()))
        p.activate();pump(app,w,lambda:(robot_state().get('active_profile') or {}).get('joints',{}).get('gripper') is not None)
        assert '+45.00°' in p.candidate_angle.text()
        active_id=robot_state()['active_profile']['id']
        old_session=w.bus.status()['session']
        w.disconnect_sim();pump(app,w,lambda:not w.bus.owner_alive() and not w.runtime_launcher.stopping)
        p.connect_device();pump(app,w,lambda:w.bus.status().get('session')!=old_session and sample_value('joint')=='gripper' and robot_state().get('connected'))
        p.send('fake_raw',raw=1500);pump(app,w,lambda:sample_value('raw')==1500)
        assert not robot_state()['candidates'] and len(robot_state()['references']['gripper'])==2
        assert p.reference_table.rowCount()==2 and p.reference_table.item(0,0).text()=='+0.00°'
        assert robot_state()['active_profile']['id']==active_id
        assert '+22.50°' in p.applied_angle.text() and '저장 보정 적용' in p.applied_angle.text()
        assert '+22.50°' in p.candidate_angle.text() and '보정 후 현재각' in p.candidate_angle.text()
        assert '+22.50°' in p.current_angle.text() and '+0.00°' in p.difference.text()
        assert '다시 기록·계산할 필요 없이' in p.next_step.text()
        assert not w.bus.status()['real_connected'] and not w.bus.status()['sim_connected']
    finally:
        w.disconnect_sim()
        pump(app,w,lambda:not w.bus.owner_alive())
        w.close();app.processEvents()


def test_calibration_refuses_to_steal_live_teleop_bus(tmp_path,monkeypatch):
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    monkeypatch.setattr(ui,'Registry',lambda:Registry(tmp_path/'registry'))
    app=QApplication.instance() or QApplication([]);w=Window(tmp_path)
    monkeypatch.setattr(w.bus,'owner_alive',lambda:True);monkeypatch.setattr(w.bus,'status',lambda:{'backend':'mujoco','state':'RUNNING'})
    try:
        with pytest.raises(ValueError,match='Disconnect'):w.connect_precision({'role':'follower','fake':True})
        assert w.runtime_launcher.process is None
    finally:w.close();app.processEvents()


def test_live_before_after_and_unrecorded_positions_without_hardware(tmp_path, monkeypatch):
    from soarm101_lab.real2sim.precision import FakeCalibrationDevice, fit_joint, review_joint, native_from_raw
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.artifacts import Registry
    monkeypatch.setattr(ui,'Registry',lambda:Registry(tmp_path/'registry'))
    app=QApplication.instance() or QApplication([]);w=Window(tmp_path/'real2sim');p=w.precision_panel
    spec=FakeCalibrationDevice('follower').motor_spec('shoulder_pan')
    refs=[dict(joint='shoulder_pan',raw=raw,legacy_position=native_from_raw(raw,spec),angle_rad=math.radians(deg),split='train',calibration_hash=spec['calibration_hash'])
          for raw,deg in [(2035,0),(3050,90),(1015,-90)]]
    candidate=review_joint(fit_joint(refs,spec,'shoulder_pan',[-2,2]))
    robot=dict(connected=True,references={'shoulder_pan':refs},candidates={'shoulder_pan':candidate},
               sample=dict(joint='shoulder_pan',raw=1015,legacy_position=-94.153846,torque=0))
    try:
        w.tabs.setCurrentIndex(3);w.show();app.processEvents()
        p.refresh({'robots':{'follower':robot}})
        assert '-94.15°' in p.current_angle.text() and '보정 전' in p.current_angle.text()
        assert '미적용' in p.candidate_angle.text() and '보정 후' in p.candidate_angle.text()
        assert p.reference_table.item(2,0).text()=='-90.00°'
        assert float(p.reference_table.item(2,5).text().strip('°')) < .5
        assert '0.50°' in p.validation_summary.text() and '독립 검증 결과가 아닙니다' in p.validation_summary.text()
        assert not p.buttons['activate'].isEnabled() and not p.advanced.isChecked()
        assert not p.values.isVisible() and not hasattr(p,'split')
        p.review_key=('follower','shoulder_pan')
        robot['sample']['raw']=3200  # Outside taught data, still inside the operational URDF range.
        robot['sample']['legacy_position']=native_from_raw(3200,spec)
        p.refresh({'robots':{'follower':robot}})
        assert '기록 범위 밖' in p.candidate_angle.text() and '3200' in p.raw_value.text()
        assert '보정 전·후 차이:' in p.difference.text() and p.live_reference_error.isVisible()
        assert len(robot['references']['shoulder_pan'])==3
        # A fresh session has no candidate; its active saved calibration still displays.
        robot['active_profile']={'id':'saved','joints':{'shoulder_pan':candidate}}
        robot['candidates']={};robot['references']={}
        robot['sample']['raw']=2035
        q,_=p.candidate_value(candidate,2035);robot['sample']['urdf_angle_rad']=q
        p.refresh({'robots':{'follower':robot}})
        assert '저장 보정 적용' in p.applied_angle.text()
        assert p.degrees(q) in p.applied_angle.text() and '보정 후 현재각' in p.candidate_angle.text()
        assert p.degrees(q) in p.current_angle.text() and '+0.00°' in p.difference.text()
        robot['sample']['urdf_angle_rad']=None
        p.refresh({'robots':{'follower':robot}})
        assert '변환 불가' in p.applied_angle.text() and '(적용됨)' not in p.candidate_angle.text()
        p.joint.setCurrentText('elbow_flex')
        assert '측정 없음' in p.current_angle.text() and '3200' not in p.raw_value.text()
        assert not p.live_reference_error.isVisible() and w.runtime_launcher.process is None
    finally:w.close();app.processEvents()
