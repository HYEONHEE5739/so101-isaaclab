"""Atomic, user-triggered real sensor snapshots. No robot commands or recording loop."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid
import xml.etree.ElementTree as ET
import fcntl
import numpy as np
from PIL import Image
from .storage import atomic, sync_directory
from .core import JOINTS, NATIVE_UNITS, to_sim, digest
from soarm101_lab.so101_joint_mapping import DEFAULT_MAPPER, require_capture_mapping

SCHEMA = 'real2sim.snapshot/1'


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_robot(urdf, root):
    urdf = Path(urdf).resolve()
    tree = ET.parse(urdf)
    dest = root / 'robot_model'
    dest.mkdir(exist_ok=True)
    copied = {}
    for mesh in tree.findall('.//mesh'):
        name = mesh.get('filename')
        if name.startswith('package://'):
            raise ValueError('Resolve package:// robot assets before capture export')
        source = (urdf.parent / name).resolve()
        if source not in copied:
            target = dest / (digest_file(source) + source.suffix.lower())
            shutil.copy2(source, target)
            copied[source] = target
        target = copied[source]
        mesh.set('filename', target.name)
    tree.write(dest / 'robot.urdf', encoding='unicode')
    return 'robot_model/robot.urdf'


def align_measured(samples, timestamp, max_skew_ns=100_000_000, mapper=None, observation_only=False):
    """Interpolate measured arm values; gripper is held from the last sample."""
    if not samples:
        raise ValueError('Measured joint history missing')
    times = np.asarray([s['timestamp_ns'] for s in samples], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError('Measured timestamps must increase')
    convert = mapper.calibrated_to_urdf if mapper else to_sim
    if observation_only and mapper: convert = getattr(mapper, 'calibrated_to_observation', convert)
    positions = np.asarray([[convert(s['native'])[n] for n in JOINTS] for s in samples])
    right = int(np.searchsorted(times, timestamp))
    if right < len(times) and times[right] == timestamp:
        return positions[right].tolist()
    if right == 0 or right == len(times):
        index = 0 if right == 0 else len(times) - 1
        if abs(int(times[index]) - timestamp) > max_skew_ns:
            raise ValueError('Image/joint timestamp skew exceeds 100 ms')
        return positions[index].tolist()
    left = right - 1
    if max(timestamp - int(times[left]), int(times[right]) - timestamp) > max_skew_ns:
        raise ValueError('Measured interpolation gap exceeds 100 ms')
    alpha = (timestamp - times[left]) / float(times[right] - times[left])
    q = positions[left] * (1 - alpha) + positions[right] * alpha
    q[-1] = positions[left, -1]
    return q.tolist()


def prepare_snapshot(state, images, profile, now_ns=None):
    """Validate and freeze one whole observation before publishing any files."""
    now_ns = state.get('captured_host_ns', time.monotonic_ns()) if now_ns is None else now_ns
    if set(images) != {'real_side', 'real_wrist'}:
        raise ValueError('Both real Side/Wrist frames are required; retry Capture')
    if state.get('joint_names') != list(JOINTS):
        raise ValueError('Measured joint names/order mismatch')
    if not state.get('sim_joint_mapping'):
        raise ValueError('Snapshot requires explicit joint mapping metadata')
    require_capture_mapping(state)
    from .precision import mapper_from_metadata
    mapper = mapper_from_metadata(state['sim_joint_mapping'])
    convert = mapper.calibrated_to_urdf
    observation_only = state.get('precision_observation_only') is True
    if observation_only:
        if state.get('sent_target') is not None: raise ValueError('Passive capture must not contain a motor command')
        convert = getattr(mapper, 'calibrated_to_observation', convert)
    q = convert(state['physical_follower_measured_joint_state'])
    samples = copy.deepcopy(state.get('measured_samples', []))
    feedback_time = state['timestamps']['follower']['end']['monotonic_ns']
    if not samples or samples[-1]['timestamp_ns'] != feedback_time:
        raise ValueError('Latest measured feedback missing from history')
    if not np.allclose([q[n] for n in JOINTS], align_measured(samples, feedback_time, mapper=mapper, observation_only=observation_only), atol=1e-8):
        raise ValueError('Measured feedback/history mismatch')
    if now_ns - feedback_time > 200_000_000 or feedback_time > now_ns:
        raise ValueError('Measured joint feedback is stale')
    stability = state.get('real_stability', {})
    if not stability.get('stable'):
        raise ValueError('Robot pose is moving or feedback is warming up; hold still and retry')
    warnings = []
    joint_limit_warnings = {n: {'measured_rad':q[n], 'limits_rad':entry['limits']}
                           for n,entry in getattr(mapper, 'profile', {}).get('joints', {}).items()
                           if not entry['limits'][0]-1e-6 <= q[n] <= entry['limits'][1]+1e-6}
    for n,detail in joint_limit_warnings.items():
        warnings.append(f'{n}: measured {np.degrees(detail["measured_rad"]):.2f} deg outside URDF limits; original measurement retained')
    observed = []
    import cv2
    for name in ('side', 'wrist'):
        rgb = images['real_' + name]
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Real camera must provide RGB uint8')
        if list(rgb.shape[1::-1]) != profile['cameras'][name]['resolution']:
            raise ValueError('Original image resolution differs from pixel calibration')
        packet = state['timestamps']['real_cameras']['real_' + name]
        t = packet['read_end']['monotonic_ns']
        start = packet['read_start']['monotonic_ns']
        if not start <= t <= now_ns or now_ns - t > 200_000_000:
            raise ValueError(name + ' camera frame is stale; retry Capture')
        if abs(t - feedback_time) > 100_000_000:
            raise ValueError(name + ' image/joint timestamp skew exceeds 100 ms')
        if start < stability['window_start_ns'] or t > stability['window_end_ns']:
            raise ValueError(name + ' frame is not covered by stable measured feedback')
        blur = float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
        if blur < 10:
            warnings.append(name + ': blurred or textureless image; excluded from calibration')
        observed.append({'camera': name, 'path': name + '.png', 'timestamp_ns': t,
                         'read_start_ns': start, 'sequence': packet['sequence'],
                         'resolution': list(rgb.shape[1::-1]), 'blur_variance': blur,
                         'q_measured': align_measured(samples, t, mapper=mapper, observation_only=observation_only),
                         'joint_skew_ns': t - feedback_time, 'eligible': blur >= 10,
                         'timestamp_kind': 'host_read_interval_not_exposure'})
    command = convert(state['sent_target']) if state.get('sent_target') is not None else None
    measured = {'timestamp_ns': feedback_time, 'q_measured': [q[n] for n in JOINTS],
                'q_native': state['physical_follower_measured_joint_state'],
                'q_command': [command[n] for n in JOINTS] if command else None,
                'command_timestamp_ns': state['timestamps']['send']['end']['monotonic_ns'] if command else None,
                'captured_host_ns': now_ns,
                'measured_samples': [{**a,'q_measured':[convert(a['native'])[n] for n in JOINTS]} for a in samples], 'timestamps': state['timestamps'], 'real_stability': stability,
                'precision_raw_feedback': state.get('precision_raw_feedback'), 'tracking': state.get('tracking'),
                'precision_run_configuration': state.get('precision_run_configuration', {})}
    measured.update(precision_observation_only=observation_only, joint_limit_warnings=joint_limit_warnings)
    return measured, observed, warnings


class SnapshotDataset:
    """One portable append-only dataset; common model/calibration copied once."""
    @staticmethod
    def new_session(started_at_unix_ns=None):
        started = time.time_ns() if started_at_unix_ns is None else started_at_unix_ns
        stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(started / 1e9))
        return {'id': f'snapshot_{stamp}_{uuid.uuid4().hex[:8]}',
                'started_at_unix_ns': started, 'ended_at_unix_ns': None}

    def __init__(self, output, name, profile, urdf, session=None):
        self.session = copy.deepcopy(session or self.new_session())
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', self.session['id']):
            raise ValueError('Invalid snapshot session ID')
        self.root = Path(output).resolve() / self.session['id']
        self.root.mkdir(parents=True, exist_ok=False)
        self.profile = copy.deepcopy(profile)
        self.urdf = str(urdf)
        self.robot_source_hash = digest_file(urdf)
        self.sensor_identity = None
        self.profile_hash = digest(profile)
        self.name = name
        self.count = 0
        self.last = None

    @classmethod
    def resume(cls, root, profile, urdf):
        """Reopen a committed dataset. Unindexed crash remnants are never observations."""
        root = Path(root).resolve()
        manifest = json.loads((root / 'capture_manifest.yaml').read_text())
        if manifest.get('schema') != SCHEMA or not (root / 'COMPLETE').is_file():
            raise ValueError('Cannot resume incomplete snapshot dataset')
        if manifest.get('profile_hash') != digest(profile):
            raise ValueError('Calibration differs from dataset; choose a new output folder')
        if manifest.get('source_robot_sha256') != digest_file(urdf):
            raise ValueError('Robot model differs from dataset; choose a new output folder')
        for path, expected in manifest['files'].items():
            if digest_file(root / path) != expected:
                raise ValueError('Snapshot common file hash mismatch')
        for row in manifest['snapshots']:
            folder = root / row['path']
            if not (folder / 'COMPLETE').is_file() or digest_file(folder / 'metadata.json') != row['metadata_sha256']:
                raise ValueError('Committed snapshot is incomplete or changed')
            for path, expected in json.loads((folder / 'metadata.json').read_text())['files'].items():
                if digest_file(folder / path) != expected:
                    raise ValueError('Committed snapshot file changed')
        instance = cls.__new__(cls)
        instance.root = root
        instance.profile = copy.deepcopy(profile)
        instance.urdf = str(urdf)
        instance.robot_source_hash = digest_file(urdf)
        instance.sensor_identity = json.loads((root / 'calibration/sensors.json').read_text()).get('hardware_identity')
        instance.profile_hash = digest(profile)
        instance.name = manifest['scene_name']
        instance.session = copy.deepcopy(manifest.get('session') or
            {'id': manifest['dataset_id'], 'started_at_unix_ns': None, 'ended_at_unix_ns': None})
        instance.count = len(manifest['snapshots'])
        instance.last = None
        return instance

    def status(self):
        return {'root': str(self.root), 'count': self.count, 'last': self.last,
                'session': copy.deepcopy(self.session)}

    def finish(self, ended_at_unix_ns=None):
        """Close a connection session after queued saves; common files remain unchanged."""
        with (self.root / '.capture.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'capture_manifest.yaml'
            if not path.exists():
                return  # No successfully indexed observations.
            manifest = json.loads(path.read_text())
            session = manifest.setdefault('session', copy.deepcopy(self.session))
            if session.get('ended_at_unix_ns') is None:
                session['ended_at_unix_ns'] = time.time_ns() if ended_at_unix_ns is None else ended_at_unix_ns
                atomic(path, manifest)
                sync_directory(self.root)
            self.session = copy.deepcopy(session)

    def save(self, state, images, scene_state='static_001', paired=False):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', scene_state):
            raise ValueError('Scene state uses letters/digits/underscores')
        if digest(state['active_profile']) != self.profile_hash:
            raise ValueError('Calibration changed; Disconnect and Start a new snapshot session')
        if digest_file(self.urdf) != self.robot_source_hash:
            raise ValueError('Robot model changed; Disconnect and Start a new snapshot session')
        measured, cameras, warnings = prepare_snapshot(state,
            {k:v for k,v in images.items() if k.startswith('real_')} if paired else images, self.profile)
        pair = None
        if paired:
            context = state.get('workspace_context')
            if not context or state.get('backend') not in ('isaac', 'mujoco'):
                raise ValueError('Pair capture requires a Builder workspace and simulation runtime')
            sim_state = state.get('sim_measured_joint_state_rad')
            sim_q = np.asarray([sim_state[n] for n in JOINTS] if isinstance(sim_state, dict) else sim_state, dtype=float)
            if sim_q.shape != (len(JOINTS),) or not np.isfinite(sim_q).all():
                raise ValueError('Missing simulation measured joint state')
            interval = state['timestamps']['sim']
            stamp = interval['end']['monotonic_ns']
            reasons = list(state.get('quality', {}).get('reasons', []))
            if state.get('quality', {}).get('eligible_for_calibration') is not True:
                reasons.append('runtime_pair_quality_not_eligible')
            difference = sim_q - np.asarray(measured['q_measured'])
            if np.max(np.abs(difference)) > .02:
                reasons.append('real_sim_joint_difference_exceeds_0.02_rad')
            if abs(stamp - measured['timestamp_ns']) > 100_000_000 or measured['captured_host_ns'] - stamp > 200_000_000:
                reasons.append('sim_frame_timestamp_skew_or_stale')
            for camera in cameras:
                if abs(camera['timestamp_ns'] - stamp) > 100_000_000:
                    reasons.append(camera['camera'] + '_real_sim_frame_skew_exceeds_100_ms')
                rgb = images.get('sim_' + camera['camera'])
                if rgb is None or rgb.dtype != np.uint8 or list(rgb.shape) != [camera['resolution'][1], camera['resolution'][0], 3]:
                    raise ValueError('Pair requires original calibrated simulation RGB for ' + camera['camera'])
                calibration = state.get('sim_cameras', {}).get(camera['camera'], {})
                if calibration.get('resolution') != camera['resolution'] or 'K' not in calibration or 'T_world_optical' not in calibration:
                    raise ValueError('Missing runtime simulation camera calibration')
            pair = {'schema': 'real2sim.paired-observation/1', 'engine': state['backend'],
                'workspace_context': copy.deepcopy(context), 'sim_timestamp': interval,
                'sim_q_measured_rad': sim_q.tolist(), 'real_sim_difference_rad': difference.tolist(),
                'sim_cameras': copy.deepcopy(state['sim_cameras']),
                'runtime_profile': copy.deepcopy(state['active_profile']),
                'real_camera_metadata': copy.deepcopy(state.get('real_camera_metadata', {})),
                'joint_mapping': copy.deepcopy(state['sim_joint_mapping']),
                'quality': {'eligible': not reasons and all(c['eligible'] for c in cameras), 'reasons': reasons},
                'sim_images': {c['camera']: 'sim_' + c['camera'] + '.png' for c in cameras},
                'meaning': 'historical_runtime_baseline_not_ground_truth'}
            warnings.extend(reasons)
        with (self.root / '.capture.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            manifest_path = self.root / 'capture_manifest.yaml'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if (manifest.get('session') or {}).get('ended_at_unix_ns') is not None:
                    raise ValueError('Snapshot session is closed; Start a new session')
                sensors = json.loads((self.root / 'calibration/sensors.json').read_text())
                if sensors.get('hardware_identity') != state.get('hardware_identity'):
                    raise ValueError('Device calibration changed; Disconnect and Start a new snapshot session')
            else:
                model = copy_robot(self.urdf, self.root)
                calibration = []
                for name in ('side', 'wrist'):
                    c = self.profile['cameras'][name]
                    calibration.append({'id': name, **{k: copy.deepcopy(c[k]) for k in
                        ('parent', 'K', 'distortion', 'resolution', 'T_parent_camera')},
                        'provenance': c.get('provenance', {}), 'resize_history': [],
                        'clock_offset_ns': 0, 'offset_provenance': 'same_host_clock_not_hardware_sync'})
                atomic(self.root / 'calibration/cameras.json', calibration)
                atomic(self.root / 'calibration/sensors.json', {k: state.get(k) for k in
                    ('hardware', 'hardware_identity', 'real_camera_metadata', 'profile_hash')})
                manifest = {'schema': SCHEMA, 'dataset_id': self.root.name, 'scene_name': self.name,
                    'session': copy.deepcopy(self.session),
                    'profile_hash': self.profile_hash,
                    'source_robot_sha256': self.robot_source_hash,
                    'clock': {'domain': 'host_monotonic_ns', 'hardware_sync': False},
                    'joint_names': list(JOINTS), 'joint_units': ['rad'] * 6,
                    'native_units': NATIVE_UNITS, 'joint_mapping': copy.deepcopy(state['sim_joint_mapping']),
                    'q_measured_source': 'physical_follower_feedback_converted_to_model',
                    'q_command_source': 'sent_target_not_measurement', 'discrete_joints': ['gripper'],
                    'robot': {'format': 'urdf', 'path': model,
                              'T_world_base': self.profile['robot_base']['T_world'],
                              'base_provenance': self.profile['robot_base'].get('provenance', {}),
                              'mapping_verified': bool(state['sim_joint_mapping'].get('physical_verified',False))},
                    'cameras': calibration, 'calibration': 'calibration/cameras.json',
                    'coordinates': {'units': 'm', 'matrix': 'T_parent_child; column vectors',
                        'world_up': '+Z', 'camera': 'x_right_y_down_z_forward', 'quaternion': 'xyzw'},
                    'physics_status': 'PHYSICS_NOT_VALIDATED', 'scene_states': {}, 'snapshots': [],
                    'files': {p.relative_to(self.root).as_posix(): digest_file(p)
                              for folder in ('robot_model', 'calibration') for p in (self.root / folder).rglob('*') if p.is_file()}}
            index = len(manifest['snapshots']) + 1
            snapshot_id = f'snapshot_{index:06d}'
            folder = self.root / 'snapshots' / snapshot_id
            # A crash before index publication may leave a complete but unindexed folder.
            # Preserve it for diagnosis and reserve a fresh ID rather than overwriting it.
            while folder.exists():
                index += 1
                snapshot_id = f'snapshot_{index:06d}'
                folder = self.root / 'snapshots' / snapshot_id
            pending = folder.with_name('.' + snapshot_id + '.' + uuid.uuid4().hex + '.pending')
            pending.mkdir(parents=True)
            committed = False
            try:
                for name in ('side', 'wrist'):
                    with (pending / (name + '.png')).open('xb') as stream:
                        Image.fromarray(images['real_' + name]).save(stream, format='PNG')
                        stream.flush()
                        os.fsync(stream.fileno())
                if pair:
                    for name, path in pair['sim_images'].items():
                        with (pending / path).open('xb') as stream:
                            Image.fromarray(images['sim_' + name]).save(stream, format='PNG')
                            stream.flush(); os.fsync(stream.fileno())
                    atomic(pending / 'pair.json', pair)
                atomic(pending / 'state.json', measured)
                metadata = {'schema': SCHEMA, 'id': snapshot_id, 'scene_state_id': scene_state,
                    'timestamp_ns': measured['timestamp_ns'], 'images': cameras,
                    'quality': {'warnings': warnings, 'eligible': all(c['eligible'] for c in cameras)},
                    'joint_metadata': '../../capture_manifest.yaml', 'files': {p.name: digest_file(p) for p in pending.iterdir() if p.is_file()}}
                atomic(pending / 'metadata.json', metadata)
                atomic(pending / 'COMPLETE', True)
                sync_directory(pending)
                pending.rename(folder)
                sync_directory(folder.parent)
                manifest['scene_states'].setdefault(scene_state, {'object_init': self.profile['objects'],
                    'assumption': 'user keeps static object layout and robot base fixed within this state'})
                manifest['snapshots'].append({'id': snapshot_id, 'scene_state_id': scene_state,
                    'path': folder.relative_to(self.root).as_posix(),
                    'metadata_sha256': digest_file(folder / 'metadata.json')})
                # COMPLETE alone is insufficient: readers also require the committed index.
                atomic(self.root / 'COMPLETE', True)
                sync_directory(self.root)
                atomic(manifest_path, manifest)
                committed = True
                try:
                    sync_directory(self.root)
                except OSError as exc:
                    warnings.append('Index published; directory durability check failed: ' + str(exc))
            finally:
                if pending.exists(): shutil.rmtree(pending)
                # Before index publication this directory is task-owned, not a valid observation.
                if not committed and folder.exists(): shutil.rmtree(folder)
            self.count = len(manifest['snapshots'])
            self.sensor_identity = state.get('hardware_identity')
            self.last = {'id': snapshot_id, 'path': str(folder), 'timestamp_ns': measured['timestamp_ns'],
                         'status': 'Saved with warnings' if warnings else 'Saved', 'warnings': warnings}
            return {**self.last, 'count': self.count, 'dataset': str(self.root)}
