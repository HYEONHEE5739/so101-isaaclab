"""Adapters for inspected repository leader and LeRobot v0.4.1 SO101Follower.
Hardware imports occur only when requested. Calibration registers are never written.
Start seeds and verifies a hold target before the existing follower configuration.
"""

import ast
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import math
import time
from pathlib import Path
from .core import JOINTS, NATIVE_UNITS, stamp, values


def ast_hash(path):
    tree = ast.parse(Path(path).read_text())
    # Python 3.12 adds empty type_params to function/class ASTs; keep the audit portable to 3.11.
    for node in ast.walk(tree):
        if hasattr(node, "type_params") and not node.type_params:
            node._fields = tuple(f for f in node._fields if f != "type_params")
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def audit_lerobot():
    version = importlib.metadata.version("lerobot")
    if version != "0.4.1":
        raise RuntimeError(
            f"Inspected adapter targets LeRobot 0.4.1, installed {version}. Do not change versions blindly; provide local follower API for review."
        )
    import lerobot

    root = Path(lerobot.__file__).resolve().parent
    manifest = json.loads(Path(__file__).with_name("api_manifest.json").read_text())
    records = {}
    for relative, expected in manifest["ast_sha256"].items():
        file = root / relative
        actual = ast_hash(file)
        if actual != expected:
            raise RuntimeError(
                f"Local LeRobot source differs from audited v0.4.1: {file}. Adapter review required before controlling follower."
            )
        records[relative] = {"path": str(file), "ast_sha256": actual}
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    return (
        SO101Follower,
        SO101FollowerConfig,
        {"version": version, "upstream_commit": manifest["commit"], "sources": records},
    )


def calibration_record(device, path):
    data = {k: dataclasses.asdict(v) for k, v in device.items()}
    p = Path(path)
    return {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "values": data}


def motor_specs(bus, calibration_hash):
    return {n:{'calibration_hash':calibration_hash,'calibration':dataclasses.asdict(bus.calibration[n]),
        'resolution':bus.model_resolution_table[bus.motors[n].model], 'mode':bus.motors[n].norm_mode.name,
        'apply_drive_mode':bool(bus.apply_drive_mode)} for n in JOINTS}


class Leader:
    def __init__(self, env, port, calibration_file_name, recalibrate=False, *, use_precision=True, precision_root=None, model_hash=None):
        self.use_precision = use_precision; self.precision_root = precision_root; self.model_hash = model_hash
        from soarm101_lab.devices import SO101Leader, SO101LeaderCfg

        class StrictLeader(SO101Leader):
            def calibrate(inner):
                raise RuntimeError(
                    "Leader calibration missing or requested implicitly. Select the existing calibration file; automatic calibration is disabled."
                )

        cfg = SO101LeaderCfg(port=port, calibration_file_name=calibration_file_name, recalibrate=recalibrate, use_precision=False)
        params = inspect.signature(SO101Leader.__init__).parameters
        kwargs = {"cfg": cfg}
        if "env" in params:
            kwargs["env"] = env
        self.device = StrictLeader(**kwargs)
        expected = {n: ("RANGE_0_100" if n == "gripper" else "DEGREES") for n in JOINTS}
        actual = {n: m.norm_mode.name for n, m in self.device._bus.motors.items()}
        if actual != expected:
            raise RuntimeError(f"Unsupported local leader normalization: {actual}")
        self.metadata = {
            "class_source": inspect.getfile(SO101Leader),
            "source_sha256": hashlib.sha256(Path(inspect.getfile(SO101Leader)).read_bytes()).hexdigest(),
            "native_units": NATIVE_UNITS,
            "normalization": actual,
            "port": port,
            "strict_read": "Missing/failed reads raise; never replace joints with zero",
        }

    def connect(self):
        self.device._bus.connect()
        if not self.device.is_calibrated:
            raise RuntimeError(
                "Leader hardware calibration does not match its file; use existing leader calibration procedure"
            )
        self.metadata["calibration"] = calibration_record(self.device._bus.calibration, self.device.calibration_path)
        self.motor_specs=motor_specs(self.device._bus,self.metadata['calibration']['sha256'])
        self.metadata['motor_specs']=self.motor_specs
        if self.use_precision:
            try: attach_active_precision(self, 'leader', self.precision_root, self.model_hash)
            except BaseException:
                self.close(); raise

    def activate(self):
        if not self.device.is_calibrated:
            raise RuntimeError("Leader calibration changed")
        self.device._bus.disable_torque()

    def read(self):
        start = stamp()
        bus = self.device._bus
        raw = bus.sync_read('Present_Position', normalize=False)
        normalized = bus._normalize({bus.motors[n].id: v for n, v in raw.items()})
        native = values({n: normalized[bus.motors[n].id] for n in JOINTS})
        self.last_raw = raw
        return native, {'start': start, 'end': stamp()}

    def close(self):
        if self.device.is_connected:
            self.device.disconnect()


class Follower:
    def __init__(
        self, port, identifier, calibration_dir=None, allow_calibration=False, *, use_precision=True, precision_root=None, model_hash=None
    ):
        self.use_precision = use_precision; self.precision_root = precision_root; self.model_hash = model_hash
        cls, cfg_cls, audit = audit_lerobot()

        class CheckedFollower(cls):
            def configure(inner):
                if not inner.is_calibrated:
                    raise RuntimeError(
                        "Follower calibration mismatch. Check the selected follower id/file using the explicit LeRobot calibration workflow; no automatic calibration is allowed."
                    )
                super().configure()

        self.device = CheckedFollower(
            cfg_cls(
                port=port,
                id=identifier,
                calibration_dir=Path(calibration_dir) if calibration_dir else None,
                use_degrees=True,
                cameras={},
                disable_torque_on_disconnect=True,
                max_relative_target=None,
            )
        )
        if allow_calibration:
            raise ValueError("Interactive calibration must be run explicitly outside Real2Sim")
        self.allow_calibration = False
        if not self.device.calibration and not allow_calibration:
            raise FileNotFoundError(
                f"Follower calibration required: {self.device.calibration_fpath}. Select your own follower id and calibration directory in the UI. Leader calibration is never reused."
            )
        self.metadata = {
            "api": audit,
            "port": port,
            "id": identifier,
            "use_degrees": True,
            "native_units": NATIVE_UNITS,
            "max_relative_target": self.device.config.max_relative_target,
        }

    def connect(self):
        try:
            # Read-only connection: do not call LeRobot connect/configure here.
            self.device.bus.connect()
            if not self.device.is_calibrated:
                raise RuntimeError("Follower calibration mismatch; no configuration or motion command sent")
            if set(self.device.action_features) != {n + ".pos" for n in JOINTS}:
                raise RuntimeError("Unexpected follower action schema")
            modes = {n: m.norm_mode.name for n, m in self.device.bus.motors.items()}
            expected = {n: ("RANGE_0_100" if n == "gripper" else "DEGREES") for n in JOINTS}
            if modes != expected:
                raise RuntimeError("Unexpected follower motor normalization")
            self.metadata["calibration"] = calibration_record(self.device.calibration, self.device.calibration_fpath)
            # Derive normalized limits using the inspected bus itself, never handwritten register units.
            cal = self.device.calibration
            bus = self.device.bus
            lo = bus._normalize({c.id: c.range_min for c in cal.values()})
            hi = bus._normalize({c.id: c.range_max for c in cal.values()})
            self.limits = {n: sorted([lo[cal[n].id], hi[cal[n].id]]) for n in JOINTS}
            self.metadata["normalized_calibrated_limits"] = self.limits
            self.motor_specs=motor_specs(bus,self.metadata['calibration']['sha256'])
            self.metadata['motor_specs']=self.motor_specs
            if self.use_precision: attach_active_precision(self, 'follower', self.precision_root, self.model_hash)
            self.read_motion_settings('connect')
        except BaseException:
            self.close()
            raise

    def read(self):
        start = stamp()
        bus = self.device.bus
        raw = bus.sync_read('Present_Position', normalize=False)
        normalized = bus._normalize({bus.motors[n].id: v for n, v in raw.items()})
        self.last_raw = raw
        return values({n: normalized[bus.motors[n].id] for n in JOINTS}), {'start':start,'end':stamp()}

    def activate(self):
        if not self.device.is_calibrated:
            raise RuntimeError("Follower calibration changed")
        bus = self.device.bus
        bus.disable_torque()
        present = bus.sync_read("Present_Position", normalize=False)
        bus.sync_write("Goal_Position", present, normalize=False)
        readback = bus.sync_read("Goal_Position", normalize=False)
        if readback != present:
            raise RuntimeError("Follower hold target readback failed; torque remains disabled")
        self.device.configure()
        self.read_motion_settings('activated')

    def read_motion_settings(self, stage):
        """Diagnostic reads only; never reset an unknown speed/phase configuration."""
        settings = {}
        for name in ('Operating_Mode', 'Goal_Velocity', 'Acceleration', 'Phase'):
            try:
                data = self.device.bus.sync_read(name, normalize=False)
                settings[name] = {n: int(data[n]) for n in JOINTS}
            except Exception as exc:
                settings[name] = {'read_error': str(exc)}
        self.metadata['motion_settings'] = {'stage': stage, 'registers': settings}
        print('[Real2Sim][follower-motion] ' + json.dumps(self.metadata['motion_settings'], ensure_ascii=False), flush=True)

    def stop(self):
        # Hold at the last target; no new goal. Disconnect releases torque via audited API.
        self._last_goal_ns = None

    def encode_goal(self, target):
        """Read-only SDK encoding preflight, including integer truncation."""
        from .precision import raw_from_native, native_from_raw
        target=values(target).copy(); bus=self.device.bus
        if any(target[n] < self.limits[n][0]-1e-6 or target[n] > self.limits[n][1]+1e-6 for n in JOINTS):
            raise ValueError("Follower native goal outside calibrated motor range")
        entries=getattr(getattr(self,'precision_mapper',None),'profile',{}).get('joints',{})
        def encode(): return bus._unnormalize({bus.motors[n].id:target[n] for n in JOINTS})
        raw=encode()
        for name in JOINTS:
            spec=entries[name]['motor'] if name in entries else self.motor_specs[name]
            identifier=bus.motors[name].id
            expected=round(raw_from_native(target[name],spec))
            if name not in entries:
                target[name]=native_from_raw(expected,spec);raw=encode()
            # The strict Precision inverse has already selected this integer goal.
            # Move only a few floating point ULPs so SDK int() encodes that goal.
            direction=-math.inf if spec['mode']=='RANGE_0_100' and spec['apply_drive_mode'] and spec['calibration']['drive_mode'] else math.inf
            for _ in range(16):
                if raw[identifier]==expected: break
                if raw[identifier]!=expected-1: raise ValueError(name+': SDK encoder disagrees with Precision inverse')
                target[name]=math.nextafter(target[name],direction);raw=encode()
            if raw[identifier]!=expected: raise ValueError(name+': SDK encoder cannot roundtrip calibrated goal')
        for n in JOINTS:
            cal=bus.calibration[n]; goal=raw[bus.motors[n].id]
            if not cal.range_min <= goal <= cal.range_max:
                raise ValueError(n+": SDK goal outside calibrated motor range")
        normalized=bus._normalize(raw)
        self.last_encoded_goal_raw=dict(raw)
        return target,values({n:normalized[bus.motors[n].id] for n in JOINTS})

    def send(self, target):
        target = values(target)
        if any(target[n] < self.limits[n][0]-1e-6 or target[n] > self.limits[n][1]+1e-6 for n in JOINTS):
            raise ValueError('Follower native goal outside calibrated motor range')
        if any(dataclasses.asdict(self.device.bus.calibration[n])!=self.motor_specs[n]['calibration'] for n in JOINTS):
            raise ValueError('Follower motor calibration identity changed; no goal sent')
        record = self.metadata['calibration']
        if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Follower calibration file identity changed; no goal sent')
        start = stamp()
        sent = self.device.send_action({n + ".pos": v for n, v in target.items()})
        end = stamp()
        previous = getattr(self, '_last_goal_ns', None)
        self._last_goal_ns = end['monotonic_ns']
        self.metadata['command_timing'] = {
            'send_ms': (end['monotonic_ns']-start['monotonic_ns'])/1e6,
            'interval_ms': None if previous is None else (end['monotonic_ns']-previous)/1e6}
        return values({n: sent[n + ".pos"] for n in JOINTS}), {"start": start, "end": end}

    def close(self):
        if self.device.is_connected:
            self.device.disconnect()
        elif self.device.bus.is_connected:
            self.device.bus.disconnect(disable_torque=True)


def audit_camera_sources():
    version = importlib.metadata.version("lerobot")
    if version != "0.4.1":
        raise RuntimeError(f"Camera adapter inspected for LeRobot 0.4.1, installed {version}")
    import lerobot

    root = Path(lerobot.__file__).resolve().parent
    manifest = json.loads(Path(__file__).with_name("api_manifest.json").read_text())
    records = {}
    for relative, expected in manifest["camera_ast_sha256"].items():
        actual = ast_hash(root / relative)
        if actual != expected:
            raise RuntimeError("Local camera implementation differs from audited source: " + str(root / relative))
        records[relative] = actual
    return {"version": version, "source_hashes": records}


class CalibrationDevice:
    """Read-only calibration connection; torque release is the sole explicit write API."""
    def __init__(self, role, hardware):
        self.role = role
        self.adapter = (Leader(None, hardware['leader_port'], hardware['leader_calibration'], False)
                        if role == 'leader' else Follower(hardware['follower_port'], hardware['follower_id'],
                                                         hardware.get('follower_calibration_dir') or None, False))
        self.bus = self.adapter.device._bus if role == 'leader' else self.adapter.device.bus
        self.exclusive = False
    def connect(self):
        # Direct read-only bus connect avoids Follower.connect's torque-releasing
        # failure cleanup. Never call configure/calibrate/activate here.
        self.bus.connect()
        import fcntl, termios
        fcntl.ioctl(self.bus.port_handler.ser.fileno(), termios.TIOCEXCL)
        self.exclusive = True
        if not self.bus.is_calibrated: raise ValueError('Calibration registers differ; no automatic rewrite or torque release')
        device = self.adapter.device
        path = device.calibration_path if self.role=='leader' else device.calibration_fpath
        record = calibration_record(self.bus.calibration, path)
        self.adapter.metadata['calibration'] = record
        self.identity = {'role': self.role, 'port': str(Path(self.adapter.metadata['port']).resolve()),
                         'id': self.adapter.metadata.get('id', Path(record['path']).name),
                         'calibration_sha256': record['sha256'],
                         'motors': {n: {'id': m.id, 'model': m.model} for n, m in self.bus.motors.items()}}
    def motor_spec(self, joint):
        motor = self.bus.motors[joint]
        return {'calibration_hash': self.identity['calibration_sha256'],
                'calibration': dataclasses.asdict(self.bus.calibration[joint]),
                'resolution': self.bus.model_resolution_table[motor.model], 'mode': motor.norm_mode.name,
                'apply_drive_mode': bool(self.bus.apply_drive_mode)}
    def unchanged(self):
        device = self.adapter.device
        path = device.calibration_path if self.role=='leader' else device.calibration_fpath
        return self.bus.is_calibrated and hashlib.sha256(Path(path).read_bytes()).hexdigest()==self.identity['calibration_sha256']
    def sample(self, joint, live=False):
        started = time.monotonic_ns()
        raw = self.bus.read('Present_Position', joint, normalize=False)
        raw_timestamp = time.monotonic_ns()
        if live:
            # One sensor packet for both displays: hand motion between two reads
            # must not turn a valid live observation into a missing measurement.
            native = self.bus._normalize({self.bus.motors[joint].id:raw})[self.bus.motors[joint].id]
            native_timestamp = raw_timestamp
        else:
            native = self.bus.read('Present_Position', joint, normalize=True)
            native_timestamp = time.monotonic_ns()
        normalized_raw = self.bus._unnormalize({self.bus.motors[joint].id: native})[self.bus.motors[joint].id]
        if abs(normalized_raw-raw)>3: raise ValueError('Raw/normalized reads moved or normalization clipped the sensor')
        torque = self.bus.read('Torque_Enable', joint, normalize=False)
        return {'raw': raw, 'legacy_position': native, 'torque': int(torque), 'timestamp_ns': time.monotonic_ns(),
                'read_start_ns':started,'raw_timestamp_ns':raw_timestamp,'native_timestamp_ns':native_timestamp,
                'clock':'host_monotonic_ns; not hardware synchronized',
                'native_source':'normalized_same_raw_packet' if live else 'separate_normalized_read',
                'joint': joint, 'device_identity': self.identity, 'calibration_hash': self.identity['calibration_sha256'],
                'motor': self.motor_spec(joint)}
    def release(self, joint): self.bus.disable_torque([joint])
    def close(self):
        # Preserve existing torque state on a read-only calibration disconnect.
        if self.bus.is_connected:
            if self.exclusive:
                import fcntl, termios
                try: fcntl.ioctl(self.bus.port_handler.ser.fileno(), termios.TIOCNXCL)
                finally: self.bus.disconnect(disable_torque=False); self.exclusive=False
            else:self.bus.disconnect(disable_torque=False)


def attach_active_precision(adapter, role, root=None, model_hash=None):
    from .precision import active_precision
    profile = active_precision(role, root)
    if profile is None: return None
    if model_hash is None:
        urdf = Path(__file__).resolve().parents[4] / 'assets/SO101/urdf/so101_isaaclab.urdf'
        model_hash = hashlib.sha256(urdf.read_bytes()).hexdigest()
    attach_precision(adapter, role, profile, model_hash)
    return adapter.precision_mapper


def attach_precision(adapter, role, profile, model_hash):
    """Bind software mapping to the exact audited motor calibration, never write it."""
    from .precision import PrecisionMapper, profile_for_model
    bus = adapter.device._bus if role == 'leader' else adapter.device.bus
    record = adapter.metadata['calibration']
    identity = {'role': role, 'port': str(Path(adapter.metadata['port']).resolve()),
                'id': adapter.metadata.get('id', Path(record['path']).name),
                'calibration_sha256': record['sha256'],
                'motors': {n: {'id': m.id, 'model': m.model} for n, m in bus.motors.items()}}
    if profile['identity'] != identity:
        changed = sorted(k for k in set(profile['identity']) | set(identity)
                         if profile['identity'].get(k) != identity.get(k))
        raise ValueError(
            f'Precision profile device/calibration identity conflict ({role}: {", ".join(changed)}). '
            '저장 보정과 현재 장치 설정이 다릅니다. 포트·장치 ID·LeRobot 보정 파일을 확인하세요.'
        )
    effective = profile_for_model(profile, model_hash)
    for joint,entry in profile['joints'].items():
        spec=entry['motor'];motor=bus.motors[joint]
        if spec['calibration']!=dataclasses.asdict(bus.calibration[joint]) or spec['resolution']!=bus.model_resolution_table[motor.model] or spec['mode']!=motor.norm_mode.name or spec['apply_drive_mode']!=bool(bus.apply_drive_mode):
            raise ValueError('Precision profile normalization/register contract differs')
    adapter.precision_mapper = PrecisionMapper(effective)
    adapter.metadata['precision_mapping'] = adapter.precision_mapper.metadata()
