"""Shared Real/Sim process state machine. Hardware writes stay in this worker."""

import copy
import hashlib
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import traceback
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from soarm101_lab.so101_joint_mapping import DEFAULT_MAPPER
from .profile import load, render_k, digest, validate
from .storage import Bus, atomic
from .core import JOINTS, CONVENTION, stamp, to_sim, from_sim, joint_error, Stability, assess_pair, values
from .hardware import Leader, Follower
from .cameras import CameraReader, device_key
from .scene import build_usd, configure_environment, apply_profile


def tf(pos, q):
    t = np.eye(4)
    t[:3, 3] = pos
    t[:3, :3] = Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()
    return t


class Runtime:
    def __init__(self, args, app):
        self.args = args
        self.app = app
        self.backend = getattr(args, 'backend', 'isaac')
        self.live_workspace = None
        self.bus = Bus(Path(args.workspace) / "ipc", owner=True)
        for name in ('real_side', 'real_wrist', 'sim_side', 'sim_wrist'):
            (self.bus.root / (name + '.jpg')).unlink(missing_ok=True)
        self.env = None
        self.scene_outputs = []
        self.leader = None
        self.follower = None
        self.cameras = {}
        self.running = False
        self.shutdown_requested = False
        self.activated = False
        self.state = "IDLE"
        self.message = "Choose profile with field provenance and Connect in PyQt"
        self.profile = None
        self.pending = None
        self.writer = ThreadPoolExecutor(max_workers=1)
        self.images = {}
        self.last_preview = 0.0
        self.error = None
        self.step = 0
        self.last_cycle = None
        self.real_history = Stability()
        self.sim_history = Stability()
        self.snapshot_data = None
        self.target_limits = {}
        self.snapshot_store = None
        self.snapshot_session = None
        self.snapshot_finalizers = []
        self.snapshot_requests = []
        self.last_snapshot = None
        self.precision = None
        self.precision_options = {}
        self.snapshot_results = []
        self.publish()

    def publish(self):
        self.bus.publish(
            pid=os.getpid(),
            state=self.state,
            backend=self.backend,
            renderer=getattr(self.args, 'renderer', None) if self.backend == 'mujoco' else None,
            message=self.message,
            error=self.error,
            capture_quality=(self.snapshot_data or {}).get("quality"),
            availability={
                "robot_state": bool(self.follower and self.snapshot_data),
                "sim": self.env is not None,
                **{name: ("error: " + camera.error if camera.error else
                           "frame received" if name in self.images
                           else "waiting for frame") for name, camera in self.cameras.items()},
            },
            real_connected=bool(self.follower),
            real_camera_connected=bool(self.cameras),
            sim_connected=self.env is not None,
            joint_error=(self.snapshot_data or {}).get("real_sim_joint_error"),
            target_limits=self.target_limits,
            snapshot_dataset=self.snapshot_store.status() if getattr(self, "snapshot_store", None) else None,
            snapshot_session=self.snapshot_session,
            snapshot_results=getattr(self,"snapshot_results",[]),
            profile_hash=digest(self.profile) if self.profile else None,
            scene_cameras=list(self.profile['cameras']) if self.profile else [],
            precision=self.precision.status() if getattr(self, 'precision', None) else None,
            precision_run_configuration=getattr(self, 'precision_options', {}),
            precision_teleop_ready=self.precision_teleop_ready(),
        )

    def disconnect_hardware(self):
        if getattr(self, 'precision', None):
            self.precision.close()
            self.precision = None
        for c in getattr(self,"snapshot_requests",[]):self.bus.reply(c["id"],False,"Snapshot cancelled by Disconnect")
        self.snapshot_requests=[]
        # Serialize finalization behind any frozen snapshot currently being written.
        # Never wait for disk I/O in the hardware control/close path.
        if self.snapshot_store is not None:
            future = self.writer.submit(self.snapshot_store.finish, time.time_ns())
            self.snapshot_finalizers.append(future)
        self.snapshot_store = None
        self.snapshot_session = None
        self.running = False
        self.activated = False
        for item in [self.follower, self.leader, *self.cameras.values()]:
            if item:
                try:
                    item.close()
                except Exception:
                    traceback.print_exc()
        self.follower = self.leader = None
        self.cameras = {}
        self.images = {}
        self.snapshot_data = None

    def connect(self, payload):
        if self.backend == 'calibration':
            from .precision import Calibration
            from .workspaces.package import ROOT
            if self.precision is None:
                self.precision = Calibration(Path(self.args.workspace)/'joint_calibration', ROOT/'assets/SO101/urdf/so101_isaaclab.urdf')
            result = self.precision.connect(payload['precision'])
            self.state = 'CALIBRATION'; self.message = result; self.error = None
            return
        if self.pending:
            raise ValueError("Wait for capture save to finish")
        started_at_unix_ns = time.time_ns()
        self.state = "CONNECTING"
        self.message = f"{self.backend} 연결 준비 중 · runtime 잠금으로 생존 확인"
        self.error = None
        self.publish()
        self.disconnect_hardware()
        p = load(payload["profile"])
        self.precision_options = copy.deepcopy(payload.get('precision_options', {}))
        if self.precision_options.get('use_precision'):
            from ..so101_robot_spec import simulation_joint_offsets
            offsets = simulation_joint_offsets(p.get('simulation_joint_offsets_rad'), JOINTS)
            if self.backend!='real' and np.any(abs(offsets) > 1e-12):
                if not self.precision_options.get('identity_sim_for_validation'):
                    raise ValueError('Precision calibration conflicts with Scene simulation_joint_offsets_rad; choose explicit identity-Sim validation configuration')
                p = copy.deepcopy(p)
                p.pop('simulation_joint_offsets_rad', None)
        cfg = payload["hardware"]
        self.config = cfg
        if not payload.get("preview_only") and not cfg.get("follower_id"):
            raise ValueError("Enter the actual follower robot id")
        if (
            not payload.get("preview_only")
            and Path(cfg["leader_port"]).resolve() == Path(cfg["follower_port"]).resolve()
        ):
            raise ValueError("Leader/follower ports must differ")
        if not payload.get("preview_only") and {"side","wrist"} <= p["cameras"].keys() and device_key(cfg["side_camera"]) == device_key(cfg["wrist_camera"]):
            raise ValueError("Side/wrist cameras must differ")
        self.state = "BUILDING_ENVIRONMENT"
        self.message = f"{self.backend} 환경 생성 중 · runtime 잠금으로 생존 확인"
        self.publish()
        if self.env is not None:
            started = time.monotonic()
            print(f'[Real2Sim][{self.backend}] Previous environment close: START', flush=True)
            self.env.close()
            self.env = None
            print(f'[Real2Sim][{self.backend}] Previous environment close: {time.monotonic()-started:.3f}s', flush=True)
        self.live_workspace = payload.get("workspace")
        self.workspace_context = None
        self.task_definition = payload.get('task_definition')
        if self.live_workspace:
            from .workspaces.package import load as read_workspace
            workspace=read_workspace(self.live_workspace)
            bindings=workspace['task'].get('real_camera_bindings',{})
            manifest=workspace['manifest']
            if manifest.get('builder'):
                from .workspaces.package import sha
                self.workspace_context={'workspace_id':manifest['workspace_id'],'version':manifest['version'],
                    'content_sha256':manifest['content_sha256'],'builder':copy.deepcopy(manifest['builder']),
                    'scene_sha256':manifest['files'][manifest['builder']['scene']],
                    'manifest_sha256':sha(workspace['root']/'manifest.json')}
            for name,key in bindings.items(): cfg[name+'_camera']=cfg.get(key,'')
        if self.backend == 'real':
            self.profile=p
        else:
            self.build_environment(p)
        from .workspaces.package import ROOT
        candidate=Path(self.live_workspace)/'assets/SO101/urdf/so101_isaaclab.urdf' if self.live_workspace else ROOT/'assets/SO101/urdf/so101_isaaclab.urdf'
        self.robot_model_path=candidate
        if self.backend == 'real':self.robot_hash=hashlib.sha256(candidate.read_bytes()).hexdigest()
        if payload.get("preview_only"):
            if payload.get('real_camera_preview'):
                self.open_cameras(cfg, p)
            self.state = "PREVIEW"
            self.message = "Simulation preview; robot motor ports remain closed"
            self.error = None
            return
        self.state = "CONNECTING"
        self.message = f"{self.backend} 환경 생성 완료 · 장비 연결 중 (새 follower 목표 전송 없음)"
        self.publish()
        # Constructors may read files; overridden automatic calibration can never run.
        self.leader = Leader(self.env, cfg["leader_port"], cfg["leader_calibration"], False, use_precision=False)
        self.follower = Follower(
            cfg["follower_port"], cfg["follower_id"], cfg.get("follower_calibration_dir") or None, False, use_precision=False
        )
        self.leader.connect()
        self.follower.connect()
        if self.precision_options.get('use_precision'):
            from .precision import ProfileStore
            from .hardware import attach_precision
            store = ProfileStore(Path(self.args.workspace)/'joint_calibration')
            model_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
            found = False
            for role, adapter in [('leader', self.leader), ('follower', self.follower)]:
                active = store.active(role)
                if active:
                    attach_precision(adapter, role, active, model_hash); found = True
            if not found: raise ValueError('Activate a validated precision profile first')
        self.leader.activate()  # Release leader only after both calibrations match.
        self.open_cameras(cfg, p)
        self.hardware_identity = digest(
            {
                "robot": self.robot_hash,
                "leader": self.leader.metadata["calibration"],
                "follower": self.follower.metadata["calibration"],
                'precision_profiles': {role: adapter.metadata.get('precision_mapping') for role, adapter in [('leader',self.leader),('follower',self.follower)]},
                "convention": CONVENTION,
                "cameras": {
                    n: {"device": c.metadata["resolved_device"], "size": c.metadata["requested_size"]}
                    for n, c in self.cameras.items()
                },
            }
        )
        from .capture_dataset import SnapshotDataset
        self.snapshot_session = SnapshotDataset.new_session(started_at_unix_ns)
        self.snapshot_results = []
        self.last_snapshot = None
        self.real_history = Stability()
        self.sim_history = Stability()
        self.error = None
        self.state = "CONNECTED"
        self.message = "Connected: leader torque released; no follower goal sent. Press Start to follow leader targets."
        self.last_cycle = None

    def open_cameras(self, cfg, p):
        if {'side','wrist'} <= p['cameras'].keys() and device_key(cfg['side_camera']) == device_key(cfg['wrist_camera']):
            raise ValueError('Side/wrist cameras must differ')
        for view in p['cameras']:
            if not cfg.get(view + '_camera'): continue
            w, h = p['cameras'][view]['resolution']
            camera = CameraReader(cfg[view + '_camera'], w, h, int(cfg.get('fps', 30)))
            self.cameras['real_' + view] = camera
            camera.start()

    def build_environment(self, p):
        # Engine files are session-local runtime resources, not reconstruction exports.
        output = TemporaryDirectory(prefix=f'scene_{self.backend}_', dir=self.bus.root)
        self.scene_outputs.append(output)
        asset_dir = Path(output.name)
        if self.backend == 'mujoco':
            from ..simulation.mujoco_backend import MujocoBackend
            from .workspaces.package import ROOT, load as load_workspace
            inputs = self.live_workspace or {'profile': p, 'root': ROOT}
            if self.live_workspace:
                w = load_workspace(self.live_workspace)
                if digest(w['profile']) != digest(p):
                    inputs = {**w, 'profile': p,
                              'urdf': w['root'] / 'assets/SO101/urdf/so101_isaaclab.urdf'}
            w, h = next(iter(p['cameras'].values()))['resolution']
            if any(c['resolution'] != [w,h] for c in p['cameras'].values()):
                raise ValueError('Real/Sim MuJoCo currently requires equal side/wrist resolutions')
            self.env = MujocoBackend(asset_dir, inputs, render_backend=self.args.renderer,
                                     width=w, height=h, timestep=1/300, decimation=10,
                                     full_workspace=True, task_definition=getattr(self, 'task_definition', None))
            # Live inspection shows the profile's measured layout, without reset randomization.
            for name in self.env.metadata['dynamic_objects']:
                self.env.set_object_pose(name, p['objects'][name]['T_workspace'])
            self.limits = self.env.limits
            self.robot_hash = self.env.metadata['model_sha256']
            self.profile = p
            return
        from isaaclab.envs import ManagerBasedEnv
        from soarm101_lab.tasks.manager_based.soarm101_lab import SO101TeleopEnvCfg

        started = time.monotonic()
        print('[Real2Sim][isaac] Scene configuration/assets: START', flush=True)
        if self.live_workspace:
            from .workspaces.environment import make_config
            ec = make_config(self.live_workspace, self.args.device,
                             task_definition=getattr(self, 'task_definition', None))
            if getattr(self, 'precision_options', {}).get('use_precision') and self.precision_options.get('identity_sim_for_validation'):
                # Session configuration only. Do not rewrite the immutable calibrated USD.
                ec.scene.robot.spawn.usd_path = str(Path(self.live_workspace)/'assets/SO101/usd/so101_isaaclab.usd')
        else:
            ec = SO101TeleopEnvCfg()
        ec.scene.num_envs = 1
        ec.sim.device = self.args.device
        ec.actions.joint_positions.joint_names = list(JOINTS)
        ec.actions.joint_positions.preserve_order = True
        ec.sim.render_interval = ec.decimation
        if not self.live_workspace:
            usd = build_usd(asset_dir / "workspace.usda", p)
            configure_environment(ec, p, usd, calibration=True)
        if not Path(ec.scene.robot.spawn.usd_path).is_file():
            raise FileNotFoundError("Install the repository SO101 robot USD at " + str(ec.scene.robot.spawn.usd_path))
        print(f'[Real2Sim][isaac] Scene configuration/assets: {time.monotonic()-started:.3f}s', flush=True)
        started = time.monotonic()
        self.message = 'Isaac context / USD scene / physics / camera 초기화 중'
        self.publish()
        print('[Real2Sim][isaac] ManagerBasedEnv/context/scene/cameras: START', flush=True)
        self.env = ManagerBasedEnv(cfg=ec)
        print(f'[Real2Sim][isaac] ManagerBasedEnv/context/scene/cameras: {time.monotonic()-started:.3f}s', flush=True)
        started = time.monotonic()
        self.message = 'Isaac reset / profile / camera warmup 중'
        self.publish()
        print('[Real2Sim][isaac] Reset/profile/camera warmup: START', flush=True)
        self.env.reset()
        if self.live_workspace:
            from .workspaces.package import load as load_workspace
            # Both live backends show the measured profile layout, not a reset sample.
            apply_profile(self.env, p, workspace=load_workspace(self.live_workspace))
        from .scene import apply_robot_appearance
        from isaaclab.sim import get_current_stage
        apply_robot_appearance(get_current_stage(), p)
        for _ in range(5):
            self.env.sim.render()
        print(f'[Real2Sim][isaac] Reset/profile/camera warmup: {time.monotonic()-started:.3f}s', flush=True)
        robot = self.env.scene["robot"]
        self.ids = [robot.joint_names.index(n) for n in JOINTS]
        if list(self.env.action_manager.get_term("joint_positions")._joint_names) != list(JOINTS):
            raise RuntimeError("Unexpected joint action mapping")
        self.limits = robot.data.soft_joint_pos_limits[0, self.ids].detach().cpu().numpy().copy()
        self.robot_hash = hashlib.sha256(Path(ec.scene.robot.spawn.usd_path).read_bytes()).hexdigest()
        self.profile = p

    def start(self):
        if not self.follower:
            raise ValueError("Connect hardware first")
        if not self.precision_teleop_ready():
            raise ValueError('Activate validated Leader/Follower profiles for the same joints before precision Teleop; static follower comparison remains available')
        if getattr(self,'precision_options',{}).get('use_precision'):
            # Read-only preflight precedes even torque-enable/hold configuration.
            leader_native,_=self.leader.read();follower_native,_=self.follower.read()
            self.model_from_native(follower_native)
            self.check_target(self.follower_target(leader_native))
        if not self.activated:
            self.leader.activate()
            self.follower.activate()
            self.activated = True
        self.running = True
        self.state = "RUNNING"
        self.message = "Leader controls Real + Sim"
        self.last_cycle = None

    def stop(self):
        self.running = False
        self.state = "STOPPED" if self.env or self.follower else "IDLE"
        self.message = "Stopped: no new physical targets. Follower holds last target until Disconnect."

    def check_target(self, native):
        q = np.array([self.model_from_native(native)[n] for n in JOINTS])
        if getattr(self, 'backend', 'isaac') == 'mujoco':
            self.env._target(q)
        return q

    def command_target(self, native):
        """Bound a leader command before sending the same target to Real and Sim."""
        self.target_limits = {}
        mapped = self.model_from_native(native)  # Exact joint names and finite values are mandatory.
        if getattr(self, 'backend', 'isaac') != 'mujoco':
            return native
        requested = np.array([mapped[n] for n in JOINTS])
        lower, upper = self.env.limits.T.copy()
        # Physical gripper commands use calibrated percent, with endpoints 0 and 100.
        ends = sorted([DEFAULT_MAPPER.gripper_zero_offset_rad,
                       DEFAULT_MAPPER.gripper_zero_offset_rad + 100 * DEFAULT_MAPPER.gripper_rad_per_percent])
        mapper = self.mapper()
        if hasattr(mapper, 'profile') and 'gripper' in mapper.profile['joints']:
            # References determine the transform, not the command bounds.
            # Precision inverse mapping checks URDF and calibrated motor limits.
            ends = None
        if ends is not None:
            lower[-1], upper[-1] = max(lower[-1], ends[0]), min(upper[-1], ends[1])
        # Leave only a numerical margin for radians -> degrees -> radians round-off.
        applied = np.clip(requested, lower + 1e-9, upper - 1e-9)
        self.env._target(applied)
        self.target_limits = {
            name: {'requested_rad': float(requested[i]), 'applied_rad': float(applied[i]),
                   'range_rad': [float(lower[i]), float(upper[i])]}
            for i, name in enumerate(JOINTS) if abs(requested[i] - applied[i]) > 1e-8}
        return self.native_from_model(dict(zip(JOINTS, applied)))

    def mapper(self, role='follower'):
        from soarm101_lab.so101_joint_mapping import DEFAULT_MAPPER as current_mapper
        if not getattr(self, 'precision_options', {}).get('use_precision'): return current_mapper
        adapter = getattr(self, role, None)
        return getattr(adapter, 'precision_mapper', None) or current_mapper

    def precision_teleop_ready(self):
        if not getattr(self,'precision_options',{}).get('use_precision'):return True
        from .precision import PrecisionMapper
        leader=self.mapper('leader');follower=self.mapper()
        return isinstance(leader,PrecisionMapper) and isinstance(follower,PrecisionMapper) and bool(follower.profile['joints']) and set(leader.profile['joints'])==set(follower.profile['joints'])

    def model_from_native(self, native):
        return self.mapper().calibrated_to_urdf(values(native))

    def observed_model_from_native(self, native):
        mapper = self.mapper()
        convert = getattr(mapper, 'calibrated_to_observation', mapper.calibrated_to_urdf)
        return convert(values(native))

    def native_from_model(self, q):
        return self.mapper().urdf_to_calibrated(values(q))

    def precision_feedback_valid(self, native):
        if not getattr(self,'precision_options',{}).get('use_precision'):return True
        try:self.model_from_native(native);return True
        except ValueError as exc:
            self.stop();self.snapshot_data=None
            self.message=str(exc)+' — calibrated feedback unavailable; check joint limits/mapping, no new goals'
            return False

    def follower_target(self, leader_native):
        if not getattr(self, 'precision_options', {}).get('use_precision'):
            return leader_native
        return self.native_from_model(self.mapper('leader').calibrated_to_urdf(leader_native))

    def tracking(self, leader, real, sent, sim=None, observation_only=False):
        if not getattr(self, 'precision_options', {}).get('use_precision'): return None
        q = self.observed_model_from_native(real) if observation_only else self.model_from_native(real)
        command = self.model_from_native(sent) if sent is not None else None
        leader_error=None
        try:leader_common=self.mapper('leader').calibrated_to_urdf(leader)
        except ValueError as exc:leader_common=None;leader_error=str(exc)
        return {'timestamp_ns': time.monotonic_ns(), 'leader_command_rad': leader_common,'leader_conversion_error':leader_error,
                'follower_command_rad': command, 'follower_measured_rad': q, 'sim_measured_rad': sim,
                'tracking_error_rad': {n:q[n]-command[n] for n in JOINTS} if command else None,
                'scope': 'passive follower observation; no physical goals' if observation_only else 'dynamic tracking, never calibration zero compensation'}

    def set_measured_pose(self, native):
        q = self.check_target(native)
        if getattr(self, 'backend', 'isaac') == 'mujoco':
            self.env.set_joint_state(q)
            return
        import torch

        robot = self.env.scene["robot"]
        full = robot.data.joint_pos.clone()
        full[0, self.ids] = torch.tensor(q, device=self.env.device, dtype=full.dtype)
        robot.write_joint_state_to_sim(full, torch.zeros_like(full))
        robot.set_joint_position_target(full)
        self.env.sim.forward()
        self.env.scene.update(self.env.step_dt)
        for _ in range(2):
            self.env.sim.render()
        for v in self.profile['cameras']:
            self.env.scene["camera_" + v + "view"].update(self.env.step_dt, force_recompute=True)

    def snapshot(self):
        begin = stamp()
        env = self.env
        if getattr(self, 'backend', 'isaac') == 'mujoco':
            images = {'sim_' + view: env.get_rgb(view) for view in self.profile['cameras']}
            base_id = env.model.body('robot_base').id
            base = np.eye(4)
            base[:3, :3] = env.data.xmat[base_id].reshape(3, 3)
            base[:3, 3] = env.data.xpos[base_id]
            links = {}
            for i in range(env.model.nbody):
                name = env.mj.mj_id2name(env.model, env.mj.mjtObj.mjOBJ_BODY, i)
                if name and name.startswith('robot_'):
                    pose = np.eye(4)
                    pose[:3, :3] = env.data.xmat[i].reshape(3, 3)
                    pose[:3, 3] = env.data.xpos[i]
                    links[name[6:]] = (np.linalg.inv(base) @ pose).tolist()
            # MuJoCo preserves the URDF's massless fixed tool frame as a site.
            tool = np.eye(4)
            tool[:3, :3] = env.data.site_xmat[env.eef_id].reshape(3, 3)
            tool[:3, 3] = env.data.site_xpos[env.eef_id]
            links['tool0'] = (np.linalg.inv(base) @ tool).tolist()
            cameras = {}
            for view in self.profile['cameras']:
                i = env.model.camera(view).id
                optical = np.eye(4)
                optical[:3, :3] = env.data.cam_xmat[i].reshape(3, 3) @ np.diag([1., -1., -1.])
                optical[:3, 3] = env.data.cam_xpos[i]
                cameras[view] = {'K': env.metadata['cameras'][view]['K'],
                                 'T_world_optical': optical.tolist(),
                                 'resolution': [env.width, env.height], 'frame': self.step,
                                 'distortion': [0] * 5}
            q = dict(zip(JOINTS, env.data.qpos[env.q_indices].tolist()))
            return images, {'start': begin, 'end': stamp(), 'sim_time': float(env.data.time)}, q, cameras, links, base.tolist()
        robot = env.scene["robot"]
        images = {}
        cameras = {}
        base = tf(robot.data.root_pos_w[0].cpu().numpy(), robot.data.root_quat_w[0].cpu().numpy())
        links = {
            n: (
                np.linalg.inv(base)
                @ tf(robot.data.body_pos_w[0, i].cpu().numpy(), robot.data.body_quat_w[0, i].cpu().numpy())
            ).tolist()
            for i, n in enumerate(robot.body_names)
        }
        for v in self.profile['cameras']:
            sensor = env.scene["camera_" + v + "view"]
            d = sensor.data
            rgb = d.output["rgb"][0, :, :, :3].cpu().numpy().copy()
            if rgb.dtype != np.uint8:
                raise TypeError("Unexpected render dtype")
            images["sim_" + v] = rgb
            k = d.intrinsic_matrices[0].cpu().numpy()
            if not np.allclose(k, render_k(self.profile["cameras"][v]), atol=0.5):
                raise RuntimeError("Isaac rendered intrinsics do not match rectified target K")
            cameras[v] = {
                "K": k.tolist(),
                "T_world_optical": tf(d.pos_w[0].cpu().numpy(), d.quat_w_ros[0].cpu().numpy()).tolist(),
                "resolution": [rgb.shape[1], rgb.shape[0]],
                "frame": int(sensor.frame[0].item()),
                "distortion": [0] * 5,
            }
        q = dict(zip(JOINTS, robot.data.joint_pos[0, self.ids].cpu().tolist()))
        end = stamp()
        return (
            images,
            {"start": begin, "end": end, "sim_time": float(env.sim.current_time)},
            q,
            cameras,
            links,
            base.tolist(),
        )

    def render(self):
        if getattr(self, 'backend', 'isaac') == 'isaac':
            self.env.sim.render()
            for view in self.profile.get('cameras', {}):
                self.env.scene['camera_' + view + 'view'].update(self.env.step_dt, force_recompute=True)

    def tick(self):
        if getattr(self, 'precision', None): self.precision.tick(); return
        if not self.env:
            if self.follower and self.backend == 'real':self.tick_real()
            return
        if not self.follower:
            self.render()
            images, *_ = self.snapshot()
            for name, camera in self.cameras.items():
                packet = camera.nearest(time.monotonic_ns())
                if packet is not None: images[name] = packet['rgb']
            self.images = images
            for n, rgb in images.items():
                path = self.bus.root / (n + ".pending")
                Image.fromarray(rgb).save(path, format="JPEG")
                path.replace(self.bus.root / (n + ".jpg"))
            return
        native, leader_interval = self.leader.read()
        real, real_interval = self.follower.read()
        if not self.precision_feedback_valid(real):return
        sent = None
        send_interval = None
        pose_error = None
        if self.running:
            try:
                target = self.command_target(self.follower_target(native))
                sent, send_interval = self.follower.send(target)
                q = self.check_target(sent)
                if getattr(self, 'backend', 'isaac') == 'mujoco':
                    self.env.step(q)
                else:
                    import torch
                    self.env.step(torch.tensor(q, device=self.env.device, dtype=torch.float32)[None])
                if self.target_limits:
                    details = ', '.join(f"{n}: {d['requested_rad']:.4f} → {d['applied_rad']:.4f} rad"
                                        for n, d in self.target_limits.items())
                    self.message = 'RUNNING · Real/Sim 관절 목표 제한 · ' + details
                else:
                    self.message = 'Leader controls Real + Sim'
            except ValueError as exc:
                self.stop()
                self.message = str(exc) + " — paused; correct pose then Start"
        else:
            try:
                self.set_measured_pose(real)
            except ValueError as exc:
                pose_error = str(exc)
                self.message = pose_error + " — Sim shows last valid pose"
                self.render()
        self.step += 1
        sim_images, sim_interval, sim_q, sim_cameras, links, base = self.snapshot()
        real, real_interval = self.follower.read()
        if not self.precision_feedback_valid(real):return
        packets = {n: c.nearest(sim_interval["end"]["monotonic_ns"]) for n, c in self.cameras.items()}
        self.real_history.add(real_interval["end"]["monotonic_ns"], real)
        self.sim_history.add(sim_interval["end"]["monotonic_ns"], from_sim(sim_q))
        common_real = self.model_from_native(real)
        error = joint_error(from_sim(common_real), sim_q)
        quality = assess_pair(
            packets,
            sim_interval["end"],
            real_interval,
            self.real_history.assess(),
            self.sim_history.assess(),
            error,
            time.monotonic_ns(),
        )
        if pose_error:
            quality["eligible_for_calibration"] = False
            quality["reasons"].append("Measured pose cannot be represented in Sim: " + pose_error)
        self.images = {**sim_images, **{n: p["rgb"] for n, p in packets.items() if p is not None}}
        self.snapshot_data = {
            "leader_target": native,
            "leader_target_limits": getattr(self, 'target_limits', {}),
            "physical_follower_measured_joint_state": real,
            "sim_measured_joint_state_rad": sim_q,
            "real_sim_joint_error": error,
            "joint_names": list(JOINTS),
            "joint_convention": CONVENTION,
            "sim_joint_mapping": self.mapper().metadata(),
            'precision_run_configuration': copy.deepcopy(getattr(self, 'precision_options', {})),
            'tracking': self.tracking(native, real, sent, sim_q),
            'precision_raw_feedback': {'leader':getattr(self.leader,'last_raw',None),'follower':getattr(self.follower,'last_raw',None)},
            "backend": getattr(self, 'backend', 'isaac'),
            "active_profile": copy.deepcopy(self.profile),
            "profile_hash": digest(self.profile),
            "sim_cameras": sim_cameras,
            "real_camera_metadata": {n: c.metadata for n, c in self.cameras.items()},
            "links_base": links,
            "robot_base_transform": base,
            "workspace_transform": self.profile["workspace"]["T_world"],
            "objects": self.profile["objects"],
            "quality": quality,
            "hardware_identity": self.hardware_identity,
            "hardware": {"leader": self.leader.metadata, "follower": self.follower.metadata},
            "timestamps": {
                "leader": leader_interval,
                "follower": real_interval,
                "send": send_interval,
                "sim": sim_interval,
                "real_cameras": {n: {k: v for k, v in p.items() if k != "rgb"} for n, p in packets.items() if p},
            },
            "sent_target": sent,
        }
        self.snapshot_data['measured_samples']=[{'timestamp_ns':int(t),'native':dict(zip(JOINTS,q.tolist()))} for t,q,_ in self.real_history.history]
        self.snapshot_data['real_stability']=self.real_history.assess()
        self.save_snapshot_requests()
        if time.monotonic() - self.last_preview > 0.1:
            self.last_preview = time.monotonic()
            for n, rgb in self.images.items():
                path = self.bus.root / (n + ".pending")
                Image.fromarray(rgb).save(path, format="JPEG", quality=85)
                path.replace(self.bus.root / (n + ".jpg"))

    def save_snapshot_requests(self):
        if not getattr(self, 'snapshot_requests', []) or self.pending is not None:return
        command=self.snapshot_requests.pop(0)
        payload=command['payload']
        try:
            from .capture_dataset import SnapshotDataset
            output=Path(payload['output']).resolve()
            if self.snapshot_store is None:
                if (output/'capture_manifest.yaml').exists():
                    raise ValueError('기존 세션 폴더 대신 저장 상위 폴더를 선택하세요. Start마다 새 세션을 만듭니다.')
                if getattr(self, 'snapshot_session', None) is None:
                    self.snapshot_session = SnapshotDataset.new_session()
                self.snapshot_store=SnapshotDataset(output,payload.get('name','Scene'),self.profile,
                    self.robot_model_path,session=self.snapshot_session)
            elif output not in (self.snapshot_store.root.parent, self.snapshot_store.root):
                raise ValueError('현재 세션 저장 위치는 '+str(self.snapshot_store.root)+
                    '입니다. 저장 위치를 바꾸려면 Disconnect 후 다시 Start하세요.')
            state=command['_snapshot_state']
            images=command['_snapshot_images']
            self.pending=(self.writer.submit(self.snapshot_store.save,state,images,payload.get('scene_state','static_001'),
                command['command'] == 'capture_pair'),command['id'])
        except Exception as exc:
            self.bus.reply(command['id'],False,str(exc));self.error=str(exc)

    def collect_snapshot_finalizers(self):
        for future in list(self.snapshot_finalizers):
            if not future.done():
                continue
            try:
                future.result()
            except Exception as exc:
                self.error = 'Snapshot session close failed: ' + str(exc)
                traceback.print_exc()
            self.snapshot_finalizers.remove(future)

    def tick_real(self):
        """The same devices/teleop and quality history, without any engine import."""
        native,leader_interval=self.leader.read()
        sent=send_interval=None
        if self.running and getattr(self,'precision_options',{}).get('use_precision'):
            before,_=self.follower.read()
            if not self.precision_feedback_valid(before):return
        if self.running:
            to_sim(native)  # Validate all names and finite native measurements before send.
            try:target=self.follower_target(native)
            except ValueError as exc:
                self.stop();self.message=str(exc)+' — paused before any physical goal'
            else:sent,send_interval=self.follower.send(target)
        real,interval=self.follower.read()
        if self.running and not self.precision_feedback_valid(real):return
        observation_only = not self.running and sent is None
        try:
            q = self.observed_model_from_native(real) if observation_only else self.model_from_native(real)
        except ValueError as exc:
            self.stop(); self.snapshot_data = None; self.message = str(exc)+' — capture measurement cannot be decoded'
            return
        joint_limits = getattr(self.mapper(), 'profile', {}).get('joints', {})
        limit_warnings = {n: {'measured_rad':q[n], 'limits_rad':entry['limits']}
                          for n,entry in joint_limits.items()
                          if not entry['limits'][0]-1e-6 <= q[n] <= entry['limits'][1]+1e-6}
        if observation_only:
            self.message = ('Capture · 범위 밖 측정도 원본 저장 · Teleop은 범위 안에서 시작하세요: '
                            + ', '.join(f'{n}={np.degrees(d["measured_rad"]):.2f}°' for n,d in limit_warnings.items())
                            if limit_warnings else 'Capture · 카메라/관절 측정 중 · 로봇 목표 전송 없음')
        t=interval['end']['monotonic_ns']
        self.real_history.add(t,real)
        packets={n:c.nearest(t) for n,c in self.cameras.items()}
        self.images={n:p['rgb'] for n,p in packets.items() if p is not None}
        self.snapshot_data={'joint_names':list(JOINTS),'joint_convention':CONVENTION,
            'sim_joint_mapping':self.mapper().metadata(),'physical_follower_measured_joint_state':real,
            'tracking':self.tracking(native,real,sent,observation_only=observation_only),'precision_run_configuration':copy.deepcopy(getattr(self,'precision_options',{})),
            'precision_observation_only':observation_only, 'joint_limit_warnings':limit_warnings,
            'precision_raw_feedback':{'leader':getattr(self.leader,'last_raw',None),'follower':getattr(self.follower,'last_raw',None)},
            'sent_target':sent,'leader_target':native,'active_profile':copy.deepcopy(self.profile),
            'profile_hash':digest(self.profile),'hardware_identity':self.hardware_identity,
            'hardware':{'leader':self.leader.metadata,'follower':self.follower.metadata},
            'real_camera_metadata':{n:c.metadata for n,c in self.cameras.items()},
            'real_stability':self.real_history.assess(),
            'measured_samples':[{'timestamp_ns':int(stamp),'native':dict(zip(JOINTS,q.tolist()))} for stamp,q,_ in self.real_history.history],
            'timestamps':{'leader':leader_interval,'follower':interval,'send':send_interval,
                'real_cameras':{n:{k:v for k,v in p.items() if k!='rgb'} for n,p in packets.items() if p is not None}}}
        self.save_snapshot_requests()
        if time.monotonic()-self.last_preview>.1:
            self.last_preview=time.monotonic()
            for name,rgb in self.images.items():
                path=self.bus.root/(name+'.pending');Image.fromarray(rgb).save(path,format='JPEG')
                path.replace(self.bus.root/(name+'.jpg'))

    def apply(self, p):
        validate(p)
        if self.running:
            raise ValueError("Stop before updating scene")
        if getattr(self, 'backend', 'isaac') == 'real' and self.follower:
            if any(p['cameras'][n]['resolution'] != self.profile['cameras'][n]['resolution'] for n in self.profile['cameras']):
                raise ValueError('Resolution changed; reconnect before applying')
            self.profile = p
            self.snapshot_data = None
            self.real_history = Stability()
            return
        if not self.env:
            raise ValueError("Connect scene first")
        # Changes in topology/dimensions need rebuild; optimizer only adjusts transforms.
        old = self.profile
        if p.get("wrist_mount") != old.get("wrist_mount"):
            raise ValueError("Wrist mount changed; reconnect")
        if set(p["objects"]) != set(old["objects"]):
            raise ValueError("Object list changed; reconnect")
        for a, b in [(p["table"], old["table"]), *[(p["objects"][n], old["objects"][n]) for n in p["objects"]]]:
            if any(a.get(k) != b.get(k) for k in ("shape", "dimensions_m", "wall_m", "render_enabled", "bottom_diameter_m", "print_texture")):
                raise ValueError("Geometry dimensions changed; reconnect")
        for view in self.profile['cameras']:
            if p["cameras"][view]["resolution"] != old["cameras"][view]["resolution"]:
                raise ValueError("Resolution changed; reconnect before applying")
        try:
            if getattr(self, 'backend', 'isaac') == 'mujoco':
                previous = self.env
                q = previous.data.qpos[previous.q_indices].copy()
                self.env = None
                try:
                    self.build_environment(p)
                    self.env.set_joint_state(q)
                except Exception:
                    if self.env is not None: self.env.close()
                    self.env = previous
                    self.profile = old
                    self.robot_hash = previous.metadata['model_sha256']
                    raise
                previous.close()
            else:
                if getattr(self, 'live_workspace', None):
                    from .workspaces.package import load as load_workspace
                    apply_profile(self.env, p, workspace=load_workspace(self.live_workspace))
                else:
                    apply_profile(self.env, p)
        except Exception:
            if getattr(self, 'backend', 'isaac') == 'isaac':
                if getattr(self, 'live_workspace', None):
                    apply_profile(self.env, old, workspace=load_workspace(self.live_workspace))
                else:
                    apply_profile(self.env, old)
            raise
        self.profile = p
        self.snapshot_data = None
        self.real_history = Stability()
        self.sim_history = Stability()


    def command(self, c):
        name = c["command"]
        p = c["payload"]
        if name.startswith('precision.'):
            if self.backend != 'calibration' or self.running or not self.precision:
                raise ValueError('Calibration requires its exclusive non-teleop worker')
            return self.precision.command(name.split('.',1)[1],p)
        if name == "connect":
            self.connect(p)
            return "Connected; follower not commanded"
        if name == "start":
            self.start()
            return "Started"
        if name == "stop":
            self.stop()
            return "Stopped"
        if name == "disconnect":
            self.stop()
            self.disconnect_hardware()
            self.state = "CLOSING"
            self.shutdown_requested = True
            return "Disconnected; simulation runtime shutting down"
        if name in ('capture_snapshot', 'capture_pair'):
            if not self.follower or set(self.cameras) != {'real_side','real_wrist'}:
                raise ValueError('Connect physical follower and both real cameras first')
            if len(self.snapshot_requests)>=8:raise ValueError('Snapshot save queue full; wait for Save result')
            if not self.snapshot_data:raise ValueError('Measured feedback is warming up; retry Capture')
            state=copy.deepcopy(self.snapshot_data)
            if name == 'capture_pair':
                if not self.env or not self.live_workspace:
                    raise ValueError('Pair capture requires connected Real/Sim with Builder Workspace')
                if not getattr(self, 'workspace_context', None):
                    raise ValueError('Pair capture requires Builder provenance')
                state['workspace_context'] = copy.deepcopy(self.workspace_context)
            state['captured_host_ns']=time.monotonic_ns()
            state['timestamps']['request']=c.get('time')
            self.snapshot_requests.append({**c,'_snapshot_state':state,
                '_snapshot_images':{k:v.copy() for k,v in self.images.items() if k in
                    (('real_side','real_wrist','sim_side','sim_wrist') if name == 'capture_pair' else ('real_side','real_wrist'))}})
            return None
        if name == "apply_profile":
            self.apply(load(p["profile"]))
            return "Profile applied"
        raise ValueError("Unsupported command: " + name)

    def run(self):
        try:
            while self.app is None or self.app.is_running():
                begin = time.monotonic()
                if self.running:
                    heartbeat = self.bus.root / "ui_heartbeat.json"
                    if not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 3:
                        self.stop()
                        self.message = "UI heartbeat lost; physical commands stopped"
                    if self.last_cycle and begin - self.last_cycle > 0.75:
                        self.stop()
                        self.message = "Control loop stalled; press Start after inspection"
                self.last_cycle = begin
                for c in self.bus.commands():
                    try:
                        result = self.command(c)
                        if result is not None:
                            self.error = None
                            self.bus.reply(c["id"], True, result)
                    except Exception as exc:
                        if c["command"] == "connect":
                            self.disconnect_hardware()
                            self.state = "ERROR"
                        self.error = str(exc)
                        self.bus.reply(c["id"], False, str(exc))
                        traceback.print_exc()
                    finally:
                        # Publish completion/error before the first potentially slow render.
                        self.publish()
                    if self.shutdown_requested:
                        break
                if self.shutdown_requested:
                    break
                try:
                    self.tick()
                except Exception as exc:
                    self.disconnect_hardware()
                    self.state = "ERROR"
                    self.error = str(exc)
                    traceback.print_exc()
                if self.pending and self.pending[0].done():
                    future, identifier = self.pending
                    try:
                        result=future.result()
                        self.last_snapshot=result
                        self.snapshot_results=(self.snapshot_results+[result])[-100:]
                        self.bus.reply(identifier,True,result)
                    except Exception as exc:
                        self.bus.reply(identifier, False, str(exc))
                    self.pending = None
                self.collect_snapshot_finalizers()
                self.publish()
                if not self.env and self.app is not None:
                    self.app.update()
                time.sleep(max(0.0, 1 / 30 - (time.monotonic() - begin)))
        finally:
            self.disconnect_hardware()
            self.writer.shutdown(wait=True)
            self.collect_snapshot_finalizers()
            if self.env:
                self.env.close()
            for output in self.scene_outputs:
                output.cleanup()
            self.scene_outputs.clear()
            self.bus.close()
