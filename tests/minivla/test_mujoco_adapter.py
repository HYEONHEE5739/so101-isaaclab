"""Mechanical API smoke fixture, NOT SO-101 geometry or pick-place benchmark."""
import os
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from soarm101_lab.simulation.mujoco_backend import MujocoSO101
from soarm101_lab.data.legacy import JOINTS, UNITS

def test_position_mapping_reset_render_step(tmp_path):
    if importlib.util.find_spec('mujoco') is None:
        pytest.skip('MuJoCo is not installed')
    from soarm101_lab.real2sim.launcher import worker_environment
    from soarm101_lab.real2sim.workspaces.package import ROOT
    code = '''
import runpy, sys
from pathlib import Path
from soarm101_lab.simulation.mujoco_backend import configure_rendering
configure_rendering('software-egl')
runpy.run_path(sys.argv[1])['_position_mapping_reset_render_step'](Path(sys.argv[2]))
'''
    env = worker_environment(ROOT)
    for key in ('MUJOCO_GL', 'PYOPENGL_PLATFORM', 'LIBGL_ALWAYS_SOFTWARE', '__EGL_VENDOR_LIBRARY_FILENAMES'):
        env.pop(key, None)
    result = subprocess.run([sys.executable, '-c', code, str(Path(__file__).resolve()), str(tmp_path)],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def _position_mapping_reset_render_step(tmp_path):
    bodies=''
    for i,name in enumerate(JOINTS):
        bodies+=f'<body pos="{i*.04} 0 .2"><joint name="{name}" type="hinge" range="-2 2"/><geom name="g{i}" type="sphere" size=".015" mass=".1"/>'
        if i==5: bodies+='<site name="eef" size=".005"/>'
        bodies+='</body>'
    xml='<mujoco><compiler angle="radian"/><option timestep="0.0033333333333333335" gravity="0 0 0"/><worldbody><camera name="side" pos="0 -1 1" xyaxes="1 0 0 0 .707 .707"/>'+bodies+'<body name="target" pos=".1 .1 .2"><freejoint name="target_free"/><geom type="sphere" size=".02" mass=".05"/></body><body name="other" pos="-.1 .1 .2"><geom type="sphere" size=".02"/></body></worldbody><actuator>'
    xml+=''.join(f'<position name="a{i}" joint="{name}" kp="10" ctrlrange="-2 2"/>' for i,name in enumerate(JOINTS))+'</actuator></mujoco>'
    (tmp_path/'scene.xml').write_text(xml)
    c=dict(xml='scene.xml',joint_names=JOINTS,units=UNITS,mj_joints=JOINTS,mj_actuators=[f'a{i}' for i in range(6)],
        mj_per_canonical=[1]*5+[.5],mj_offset=[0]*6,lower=[-1]*5+[0],upper=[1]*6,
        eef_site='eef',cameras={'side':'side'},finger_geom_groups=[['g4'],['g5']],render_height=64,render_width=64)
    (tmp_path/'mapping.json').write_text(json.dumps(c))
    env=MujocoSO101(tmp_path/'mapping.json',['side'],30)
    try:
        case=dict(robot_state=[0]*5+[.5],target_body='target',distractor_bodies=['other'],
            destination_center=[.2,.2,.2],destination_radius=.03,destination_z_tolerance=.02)
        obs=env.reset(case); assert obs['images'][0].shape==(64,64,3)
        np.testing.assert_allclose(obs['state'],case['robot_state'])
        row=env.step(case['robot_state']); assert np.isfinite(row['target_distance'])
        assert not row['place'] and env.data.time==pytest.approx(1/30)
    finally: env.close()
