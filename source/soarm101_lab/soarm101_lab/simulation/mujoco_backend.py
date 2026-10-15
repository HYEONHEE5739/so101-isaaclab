"""One SO101 environment, CPU physics, one optional renderer. No Isaac imports."""
import copy
import ctypes.util
import json
from pathlib import Path
import os
import sys
from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .base import SimBackend
from .mjcf import build_scene, transform
from ..so101_joint_mapping import JOINTS, POLICY_UNITS, RobotCoordinates
from ..real2sim.workspaces.reset import WorkspaceRandomizer


def configure_rendering(backend):
    modes = {'none': 'disable', 'egl': 'egl', 'osmesa': 'osmesa', 'software-egl': 'egl'}
    if backend not in modes:
        raise ValueError('render_backend must be none, egl, osmesa or software-egl')
    if backend == 'osmesa' and ctypes.util.find_library('OSMesa') is None:
        raise RuntimeError('OSMesa shared library unavailable; install libosmesa6 or use software-egl')
    desired = modes[backend]
    software = backend == 'software-egl'
    if 'mujoco' in sys.modules:
        if os.environ.get('MUJOCO_GL', 'glfw') != desired or software != (os.environ.get('LIBGL_ALWAYS_SOFTWARE') == '1'):
            raise RuntimeError('Choose renderer before importing MuJoCo; compare modes in separate processes')
    elif backend == 'egl' and os.environ.get('LIBGL_ALWAYS_SOFTWARE') == '1':
        raise ValueError('EGL GPU request conflicts with LIBGL_ALWAYS_SOFTWARE=1')
    os.environ['MUJOCO_GL'] = desired
    if software:
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
        # GLVND otherwise enumerates NVIDIA first, ignoring Mesa's software flag.
        mesa = '/usr/share/glvnd/egl_vendor.d/50_mesa.json'
        if os.path.isfile(mesa):
            os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = mesa


class MujocoBackend(SimBackend):
    def __init__(self, output_dir, workspace=None, *, render_backend='none', width=320, height=240,
                 timestep=.002, decimation=10, task_definition=None, full_workspace=False,
                 physics_profile='isaac_matched', identity_sim=False):
        if any(n.split('.')[0] in ('isaaclab', 'omni', 'pxr') for n in sys.modules):
            raise RuntimeError('Run MuJoCo in an independent worker without Isaac/Omniverse/USD imports')
        configure_rendering(render_backend)
        import mujoco
        self.mj = mujoco
        self.render_backend = render_backend
        self.renderer = None
        self.viewer = None
        self.closed = False
        self.width, self.height = width, height
        self.scene_path, self.workspace, self.metadata = build_scene(
            output_dir, workspace, width, height, timestep, decimation, task_definition, full_workspace, physics_profile, identity_sim)
        if identity_sim: self.metadata['simulation_joint_frame'] = 'urdf'
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.decimation = decimation
        self.limits = np.asarray(self.metadata['joint_limits_rad'])
        self.home = np.array([self.workspace['task']['robot_initial_joint_rad'][n] for n in JOINTS])
        self.joint_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS])
        self.q_indices = self.model.jnt_qposadr[self.joint_ids]
        self.v_indices = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, 'target_' + n) for n in JOINTS])
        self.eef_id = self._id(mujoco.mjtObj.mjOBJ_SITE, 'tool0')
        review = self.workspace['task'].get('mode') == 'geometry_review'
        self.object_joint = None if review else self._id(mujoco.mjtObj.mjOBJ_JOINT, 'object_free')
        self.object_q = None if review else int(self.model.jnt_qposadr[self.object_joint])
        self.object_v = None if review else int(self.model.jnt_dofadr[self.object_joint])
        self.object_indices = {}
        for name in self.metadata['dynamic_objects']:
            joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, 'object_free' if name == self.metadata['pick'] else 'object_free_' + name)
            self.object_indices[name] = (int(self.model.jnt_qposadr[joint]), int(self.model.jnt_dofadr[joint]))
        self.world_from_workspace = np.asarray(self.metadata['workspace_T_world'])
        self.workspace_from_world = np.linalg.inv(self.world_from_workspace)
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.randomizer = WorkspaceRandomizer(self.workspace)
        self.reset()

    def _id(self, kind, name):
        index = self.mj.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError('Missing model entity: ' + name)
        return index

    def _open(self):
        if self.closed:
            raise RuntimeError('MuJoCo environment is closed')

    def _target(self, value):
        if isinstance(value, Mapping):
            if set(value) != set(JOINTS):
                raise ValueError('Expected exact SO101 joint names')
            value = [value[n] for n in JOINTS]
        q = np.asarray(value, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Expected finite model-radian target of shape (6,)')
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            bad = [f'{JOINTS[i]}={q[i]:.6f} rad outside [{lo:.6f}, {hi:.6f}]'
                   for i, (lo, hi) in enumerate(self.limits) if q[i] < lo or q[i] > hi]
            raise ValueError('Joint target exceeds URDF limits: ' + '; '.join(bad))
        return q

    def reset(self, seed=None, initial_state=None):
        self._open()
        if initial_state is not None:
            if seed is not None:
                raise ValueError('Choose initial_state restoration or seeded reset, not both')
            self.restore_state(initial_state)
        else:
            if seed is not None:
                if type(seed) is not int or not 0 <= seed < 2**32:
                    raise ValueError('Seed must be integer in [0, 2**32)')
                self.randomizer = WorkspaceRandomizer(self.workspace, seed)
            sample = self.randomizer.sample()
            self.mj.mj_resetData(self.model, self.data)
            self.data.qpos[self.q_indices] = self._target(self.home)
            self.data.ctrl[self.actuator_ids] = self.home
            for name, point in sample['cube_positions'].items():
                pose = np.eye(4)
                pose[:3, 3] = point
                self.set_object_pose(name, pose)
            self.mj.mj_forward(self.model, self.data)
        self.initial_state = self._snapshot()
        if self.viewer is not None: self.viewer.sync()
        return self.get_state()

    def step(self, joint_target_rad):
        self._open()
        self.data.ctrl[self.actuator_ids] = self._target(joint_target_rad)
        before = float(self.data.time)
        warnings = np.array([w.number for w in self.data.warning])
        for _ in range(self.decimation):
            self.mj.mj_step(self.model, self.data)
        # Refresh kinematics at the integrated state, rather than returning pre-integration poses.
        self.mj.mj_forward(self.model, self.data)
        if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or self.data.time <= before or np.any(np.array([w.number for w in self.data.warning]) > warnings)):
            raise RuntimeError('Nonfinite/unstable MuJoCo step; inspect warning and actuator/collision settings')
        if self.viewer is not None: self.viewer.sync()
        return self.get_state()

    def get_state(self):
        self._open()
        return {'joint_order': list(JOINTS), 'joint_pos': self.data.qpos[self.q_indices].copy(),
                'joint_vel': self.data.qvel[self.v_indices].copy(),
                'joint_targets': self.data.ctrl[self.actuator_ids].copy(),
                'eef_pose': self.get_eef_pose(),
                'object_poses': {n: self.get_object_pose(n) for n in self.metadata['dynamic_objects'] + self.metadata['target_objects']},
                'timestamp_s': float(self.data.time), 'success': self.is_success(),
                'physics_device': 'cpu', 'frame': 'workspace', 'joint_unit': 'rad'}

    def set_joint_state(self, joint_pos):
        """Measured-pose inspection only; no integration or hardware command."""
        self._open()
        q = self._target(joint_pos)
        self.data.qpos[self.q_indices] = q
        self.data.qvel[self.v_indices] = 0
        self.data.ctrl[self.actuator_ids] = q
        self.mj.mj_forward(self.model, self.data)

    def _pose(self, position, rotation):
        t = np.eye(4)
        t[:3, 3] = position
        t[:3, :3] = np.asarray(rotation).reshape(3, 3)
        return self.workspace_from_world @ t

    def get_eef_pose(self):
        self._open()
        return self._pose(self.data.site_xpos[self.eef_id], self.data.site_xmat[self.eef_id])

    def get_object_pose(self, object_name):
        self._open()
        if object_name not in self.metadata['dynamic_objects'] + self.metadata['target_objects']:
            raise ValueError('Unknown task object: ' + object_name)
        index = self._id(self.mj.mjtObj.mjOBJ_BODY, object_name)
        return self._pose(self.data.xpos[index], self.data.xmat[index])

    def set_object_pose(self, object_name, pose):
        self._open()
        if object_name not in self.object_indices:
            raise ValueError('Only dynamic objects can be moved; targets are fixed')
        object_q, object_v = self.object_indices[object_name]
        world = self.world_from_workspace @ transform(pose)
        quat = Rotation.from_matrix(world[:3, :3]).as_quat()
        self.data.qpos[object_q:object_q + 7] = [*world[:3, 3], *quat[[3, 0, 1, 2]]]
        self.data.qvel[object_v:object_v + 6] = 0
        self.data.qacc_warmstart[:] = 0
        self.mj.mj_forward(self.model, self.data)

    def success_details(self):
        self._open()
        task, p = self.workspace['task'], self.workspace['profile']
        if task.get('mode') == 'geometry_review':
            return {'success': None, 'definition': 'not applicable: geometry review'}
        cup = p['objects'][task['targets'][task['place']]]
        relative = np.linalg.inv(self.get_object_pose(task['place'])) @ self.get_object_pose(task['pick'])
        size = np.asarray(p['objects'][task['pick']]['dimensions_m'])
        corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * size / 2
        points = corners @ relative[:3, :3].T + relative[:3, 3]
        if task['success']['relation'] == 'inside_region_and_settled':
            lo, hi = np.asarray(task['success']['bounds_m'])
            body = self._id(self.mj.mjtObj.mjOBJ_BODY, task['pick'])
            velocity = np.empty(6)
            self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY, body, velocity, 0)
            speed = float(np.linalg.norm(velocity[3:]))
            inside = bool(np.all(points >= lo) and np.all(points <= hi))
            return {'height_ok':inside, 'wall_ok':inside, 'speed_ok':speed < task['success']['max_speed_m_s'],
                    'success':inside and speed < task['success']['max_speed_m_s'], 'speed_m_s':speed,
                    'definition':'conservative local bounds inside explicit target-local region'}
        h = cup['dimensions_m'][2]
        top = cup['dimensions_m'][0] / 2
        bottom = cup.get('bottom_diameter_m', top * 2) / 2
        radius = bottom + (top - bottom) * (points[:, 2] + h / 2) / h - cup['wall_m']
        excess = np.linalg.norm(points[:, :2], axis=1) - radius
        body = self._id(self.mj.mjtObj.mjOBJ_BODY, task['pick'])
        velocity = np.empty(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY, body, velocity, 0)
        speed = float(np.linalg.norm(velocity[3:]))
        checks = {'height_ok': bool(points[:, 2].min() > -h / 2 + cup['wall_m'] - .002
                                    and points[:, 2].max() < h / 2),
                  'wall_ok': bool(np.all(excess <= .001)),
                  'speed_ok': bool(speed < task['success']['max_speed_m_s'])}
        return {**checks, 'success': all(checks.values()), 'speed_m_s': speed,
                'radial_excess_m': float(excess.max()),
                'definition': 'all eight corners contained and instantaneous linear speed below threshold'}

    def is_success(self):
        return self.success_details()['success']

    def finger_contacts(self):
        """Target-filtered contacts for the same two finger bodies used by Isaac."""
        target = self._id(self.mj.mjtObj.mjOBJ_BODY, self.metadata['pick'])
        fingers = [self._id(self.mj.mjtObj.mjOBJ_BODY, n) for n in ('robot_gripper', 'robot_jaw')]
        contact = [False, False]
        for c in self.data.contact:
            bodies = {int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])}
            if target in bodies and c.dist <= 0:
                for i, body in enumerate(fingers): contact[i] |= body in bodies
        return contact

    def target_pose(self, joints):
        """FK on scratch data; never modifies the running physical state."""
        scratch = self.mj.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self.q_indices] = self._target(joints)
        self.mj.mj_forward(self.model, scratch)
        return self._pose(scratch.site_xpos[self.eef_id], scratch.site_xmat[self.eef_id])

    def solve_pose(self, pose, seed, position_tolerance=.003, rotation_tolerance=.12):
        """Bounded 5-axis IK. Unreachable 6D poses are rejected, never silently accepted."""
        from scipy.optimize import least_squares
        goal = self.world_from_workspace @ transform(pose)
        scratch = self.mj.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        q = self._target(seed).copy()
        def residual(arm):
            scratch.qpos[self.q_indices] = np.r_[arm, q[5]]
            self.mj.mj_forward(self.model, scratch)
            position = scratch.site_xpos[self.eef_id] - goal[:3, 3]
            rotation = Rotation.from_matrix(goal[:3, :3] @ scratch.site_xmat[self.eef_id].reshape(3, 3).T).as_rotvec()
            return np.r_[position, .04 * rotation]
        fitted = least_squares(residual, q[:5], bounds=(self.limits[:5, 0], self.limits[:5, 1]),
                               max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        error = residual(fitted.x)
        if np.linalg.norm(error[:3]) > position_tolerance or np.linalg.norm(error[3:]) / .04 > rotation_tolerance:
            raise ValueError('Unreachable target pose: position/rotation IK residual exceeded')
        q[:5] = fitted.x
        return self._target(q)

    def _snapshot(self):
        state = np.empty(self.mj.mj_stateSize(self.model, self.state_spec))
        self.mj.mj_getState(self.model, self.data, state, self.state_spec)
        return {'schema': 'so101.mujoco-state/1', 'backend': 'mujoco',
                'model_sha256': self.metadata['model_sha256'], 'mujoco_version': self.mj.__version__,
                'state_spec': int(self.state_spec), 'integration_state': state.tolist()}

    def get_initial_state(self):
        self._open()
        return copy.deepcopy(self.initial_state)

    def restore_state(self, state):
        self._open()
        if (state.get('schema') != 'so101.mujoco-state/1' or state.get('backend') != 'mujoco'
                or state.get('model_sha256') != self.metadata['model_sha256']
                or state.get('mujoco_version') != self.mj.__version__ or state.get('state_spec') != int(self.state_spec)):
            raise ValueError('Incompatible MuJoCo state/model/version')
        values = np.asarray(state['integration_state'], dtype=float)
        if values.shape != (self.mj.mj_stateSize(self.model, self.state_spec),) or not np.isfinite(values).all():
            raise ValueError('Invalid integration state')
        self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_setState(self.model, self.data, values, self.state_spec)
        self.mj.mj_forward(self.model, self.data)
        return self.get_state()

    def get_rgb(self, camera_name=None):
        self._open()
        camera_name = camera_name or self.metadata.get('primary_camera','side')
        if camera_name != self.metadata.get('primary_camera','side') and camera_name not in self.metadata.get('cameras', {}):
            raise ValueError('Unknown scene camera: ' + camera_name)
        if self.render_backend == 'none':
            raise RuntimeError('Renderer disabled; launch with egl, osmesa or software-egl')
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, height=self.height, width=self.width)
            # Refresh after construction before querying context-specific GL metadata.
            self.renderer._gl_context.make_current()
            from OpenGL.GL import GL_RENDERER, GL_VENDOR, glGetString
            vendor, renderer = glGetString(GL_VENDOR), glGetString(GL_RENDERER)
            if vendor is None or renderer is None:
                # Some PyOpenGL wrappers select GLX entry points even with EGL.
                # Query through the platform's GL library for the current context.
                import ctypes
                library = ctypes.CDLL(ctypes.util.find_library('OpenGL') or 'libGL.so.1')
                query = library.glGetString
                query.argtypes = [ctypes.c_uint]
                query.restype = ctypes.c_char_p
                vendor, renderer = query(GL_VENDOR), query(GL_RENDERER)
                if vendor is None and self.render_backend in ('egl', 'software-egl'):
                    egl = ctypes.CDLL(ctypes.util.find_library('EGL') or 'libEGL.so.1')
                    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
                    egl.eglGetProcAddress.restype = ctypes.c_void_p
                    address = egl.eglGetProcAddress(b'glGetString')
                    if address:
                        query = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_uint)(address)
                        vendor, renderer = query(GL_VENDOR), query(GL_RENDERER)
            if vendor is None or renderer is None:
                raise RuntimeError('Active renderer did not provide GL vendor/renderer metadata')
            self.metadata['render_context'] = {
                'requested': self.render_backend,
                'vendor': vendor.decode(), 'renderer': renderer.decode(),
            }
            renderer_name = self.metadata['render_context']['renderer'].lower()
            self.metadata['render_context']['software'] = any(
                name in renderer_name for name in ('llvmpipe', 'softpipe', 'software', 'swrast'))
            if self.render_backend == 'software-egl' and not self.metadata['render_context']['software']:
                raise RuntimeError('Software rendering requested but GL renderer is not recognized as software')
        self.renderer.update_scene(self.data, camera=camera_name)
        rgb = self.renderer.render().copy()
        if self.metadata.get('render_profile') == 'so101.mujoco-rgb/2':
            # Workspace material values were fitted in Isaac's linear-light renderer.
            # Apply one explicit display transfer consistently to UI, records and policy.
            if not hasattr(self, '_display_lut'):
                linear = np.arange(256, dtype=float) / 255
                display = np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)
                self._display_lut = np.rint(display*255).astype(np.uint8)
            rgb = self._display_lut[rgb]
        return rgb

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.closed = True


class PolicyEnvironment:
    def __init__(self, output, workspace, renderer='egl', control_hz=30, width=320, height=240,
                 closed_rad=None, open_rad=None, task_definition=None, full_workspace=True, viewer=False,
                 physics_profile='isaac_matched', identity_sim=False):
        if not np.isfinite(control_hz) or not 10 <= control_hz <= 200:
            raise ValueError('control_hz must be in [10,200]')
        self.hz = float(control_hz)
        self.coordinates = RobotCoordinates(closed_rad, open_rad)
        self.backend = MujocoBackend(output, workspace, render_backend=renderer, width=width,
                                     height=height, timestep=1 / (self.hz * 10), decimation=10,
                                     task_definition=task_definition, full_workspace=full_workspace,
                                     physics_profile=physics_profile, identity_sim=identity_sim)
        limits = self.backend.limits.copy()
        if self.coordinates.closed < limits[5, 0] or self.coordinates.open > limits[5, 1] + 1e-5:
            self.backend.close()
            raise ValueError('Canonical gripper endpoints exceed robot model limits')
        self.lower = np.r_[limits[:5, 0], 0.]
        self.upper = np.r_[limits[:5, 1], 1.]
        if viewer:
            # GLFW viewer startup can invalidate a lazily initialized EGL display.
            # Create the offscreen context first; both contexts then coexist.
            if renderer != 'none':
                self.backend.get_rgb()
            import mujoco.viewer
            self.backend.viewer = mujoco.viewer.launch_passive(self.backend.model, self.backend.data)

    def observe(self):
        state = self.backend.get_state()
        names = self.backend.workspace['task'].get('observation_cameras', ['side'])
        camera_images = {n:self.backend.get_rgb(n) for n in names}
        return {'images': list(camera_images.values()), 'camera_images':camera_images,
                'state': self.coordinates.to_policy(state['joint_pos'], measured=True),
                'model_joint_rad': state['joint_pos'], 'eef_pose': state['eef_pose'],
                'object_poses': state['object_poses'], 'timestamp': state['timestamp_s'],
                'success': state['success']}

    def step(self, canonical):
        return self.backend.step(self.coordinates.to_model(canonical))

    def reset(self, seed):
        return self.backend.reset(seed=seed)

    def close(self):
        self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class MujocoSO101:
    def __init__(self, mapping_file, cameras, hz):
        import mujoco as mj
        self.mj=mj; path=Path(mapping_file); self.cfg=json.loads(path.read_text()); c=self.cfg
        if c['joint_names']!=list(JOINTS) or c['units']!=list(POLICY_UNITS): raise ValueError('Mapping order/units mismatch')
        self.model=mj.MjModel.from_xml_path(str(path.parent/c['xml']))
        self.data=mj.MjData(self.model); self.hz=hz
        ratio=1/(hz*self.model.opt.timestep); self.substeps=round(ratio)
        if self.substeps<1 or abs(ratio-self.substeps)>1e-5: raise ValueError('control_dt must be integer multiple of physics_dt')
        self.joints=[self.id(mj.mjtObj.mjOBJ_JOINT,n) for n in c['mj_joints']]
        self.actuators=[self.id(mj.mjtObj.mjOBJ_ACTUATOR,n) for n in c['mj_actuators']]
        if len(self.joints)!=6 or len(self.actuators)!=6: raise ValueError('Six joint/actuator mapping entries required')
        for jid,aid in zip(self.joints,self.actuators):
            if int(self.model.jnt_type[jid]) not in (int(mj.mjtJoint.mjJNT_HINGE),int(mj.mjtJoint.mjJNT_SLIDE)):
                raise ValueError('Mapped robot joints must have one scalar position')
            if self.model.actuator_trnid[aid,0]!=jid or not np.isclose(self.model.actuator_gear[aid,0],1):
                raise ValueError('Adapter expects direct joint transmission with gear 1')
            gain=self.model.actuator_gainprm[aid,0]
            if gain<=0 or not np.isclose(self.model.actuator_biasprm[aid,1],-gain):
                raise ValueError('Mapped actuator is not a direct position servo')
        self.qadr=np.array([self.model.jnt_qposadr[i] for i in self.joints])
        self.scale=np.asarray(c['mj_per_canonical'],float); self.offset=np.asarray(c['mj_offset'],float)
        if self.scale.shape!=(6,) or self.offset.shape!=(6,) or not np.isfinite(self.scale).all() or np.any(self.scale==0) or not np.isfinite(self.offset).all():
            raise ValueError('Invalid canonical-to-MuJoCo conversion')
        self.low=np.asarray(c['lower'],float); self.high=np.asarray(c['upper'],float)
        if self.low.shape!=(6,) or self.high.shape!=(6,) or not np.isfinite(self.low).all() or not np.isfinite(self.high).all() or np.any(self.low>=self.high):
            raise ValueError('Fill all canonical lower/upper bounds in mapping')
        for j,aid in enumerate(self.actuators):
            limits=np.sort(np.array([self.low[j],self.high[j]])*self.scale[j]+self.offset[j])
            jid=self.joints[j]
            if self.model.jnt_limited[jid] and (limits[0]<self.model.jnt_range[jid,0]-1e-6 or limits[1]>self.model.jnt_range[jid,1]+1e-6):
                raise ValueError('Canonical bounds exceed physical joint range')
            if self.model.actuator_ctrllimited[aid] and (limits[0]<self.model.actuator_ctrlrange[aid,0]-1e-6 or limits[1]>self.model.actuator_ctrlrange[aid,1]+1e-6):
                raise ValueError('Canonical bounds exceed actuator ctrlrange')
        self.eef_id=self.id(mj.mjtObj.mjOBJ_SITE,c['eef_site'])
        self.camera_ids=[self.id(mj.mjtObj.mjOBJ_CAMERA,c['cameras'][n]) for n in cameras]
        self.fingers=[set(self.id(mj.mjtObj.mjOBJ_GEOM,n) for n in names) for names in c['finger_geom_groups']]
        if len(self.fingers)!=2 or not all(self.fingers): raise ValueError('Two finger contact groups required')
        self.renderer=mj.Renderer(self.model,height=c.get('render_height',480),width=c.get('render_width',640))

    def id(self,kind,name):
        value=self.mj.mj_name2id(self.model,kind,name)
        if value<0: raise ValueError(f'MuJoCo name not found: {name}')
        return value
    def state(self): return ((self.data.qpos[self.qadr]-self.offset)/self.scale).astype(np.float32)
    def reset(self, case):
        mj=self.mj; mj.mj_resetData(self.model,self.data)
        initial=np.asarray(case['robot_state'],float)
        if initial.shape!=(6,) or not np.isfinite(initial).all() or np.any(initial<self.low) or np.any(initial>self.high):
            raise ValueError('Invalid case initial robot state')
        self.data.qpos[self.qadr]=initial*self.scale+self.offset
        self.data.ctrl[self.actuators]=initial*self.scale+self.offset
        for name,pose in case.get('free_joint_poses',{}).items():
            j=self.id(mj.mjtObj.mjOBJ_JOINT,name)
            if int(self.model.jnt_type[j])!=int(mj.mjtJoint.mjJNT_FREE): raise ValueError('Object reset expects freejoint')
            pose=np.asarray(pose,float)
            if pose.shape!=(7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]),1,atol=1e-4):
                raise ValueError('Freejoint pose = xyz + unit quaternion wxyz')
            adr=self.model.jnt_qposadr[j]; self.data.qpos[adr:adr+7]=pose
        mj.mj_forward(self.model,self.data)
        self.target=self.id(mj.mjtObj.mjOBJ_BODY,case['target_body'])
        self.distractors=[self.id(mj.mjtObj.mjOBJ_BODY,n) for n in case['distractor_bodies']]
        if self.target in self.distractors: raise ValueError('Target cannot be its own distractor')
        target_bodies={self.target}
        for b in range(self.model.nbody):
            if int(self.model.body_parentid[b]) in target_bodies: target_bodies.add(b)
        self.target_geoms={i for i in range(self.model.ngeom) if int(self.model.geom_bodyid[i]) in target_bodies}
        self.destination=np.asarray(case['destination_center'],float)
        if self.destination.shape!=(3,) or not np.isfinite(self.destination).all(): raise ValueError('Invalid destination')
        self.destination_radius=float(case['destination_radius']); self.z_tolerance=float(case['destination_z_tolerance'])
        if self.destination_radius<=0 or self.z_tolerance<=0: raise ValueError('Destination tolerances must be positive')
        self.start_z=float(self.data.xpos[self.target,2]); self.was_grasped=False; self.transported=False; self.place_dwell=0
        self.initial_xy=self.data.xpos[self.target,:2].copy()
        return self.observe()
    def observe(self):
        images=[]
        for camera in self.camera_ids:
            self.renderer.update_scene(self.data,camera=camera); images.append(self.renderer.render().copy())
        return {'images':images,'state':self.state(),'timestamp':float(self.data.time)}
    def step(self, action):
        self.data.ctrl[self.actuators]=np.asarray(action)*self.scale+self.offset
        for _ in range(self.substeps): self.mj.mj_step(self.model,self.data)
        return self.measure()
    def measure(self):
        q=self.state()
        if not np.isfinite(q).all(): raise FloatingPointError('MuJoCo state diverged')
        eef=self.data.site_xpos[self.eef_id].copy(); target=self.data.xpos[self.target].copy()
        contact=[False,False]
        for i in range(self.data.ncon):
            g1,g2=int(self.data.contact[i].geom1),int(self.data.contact[i].geom2)
            for k,fingers in enumerate(self.fingers):
                contact[k]|=(g1 in fingers and g2 in self.target_geoms) or (g2 in fingers and g1 in self.target_geoms)
        lifted=target[2]>self.start_z+self.cfg.get('lift_threshold_m',.015)
        grasp=all(contact) and lifted
        self.was_grasped|=grasp
        self.transported|=self.was_grasped and lifted and np.linalg.norm(target[:2]-self.initial_xy)>self.cfg.get('transport_threshold_m',.04)
        placed=(self.transported and np.linalg.norm(target[:2]-self.destination[:2])<self.destination_radius
            and abs(target[2]-self.destination[2])<self.z_tolerance and not any(contact))
        self.place_dwell=self.place_dwell+1 if placed else 0
        return {'eef':eef.tolist(),'target_distance':float(np.linalg.norm(eef-target)),
            'distractor_distance':min((float(np.linalg.norm(eef-self.data.xpos[b])) for b in self.distractors),default=float('inf')),
            'grasp':grasp,'transport':self.transported,'place':self.place_dwell>=round(.3*self.hz)}
    def close(self): self.renderer.close()
