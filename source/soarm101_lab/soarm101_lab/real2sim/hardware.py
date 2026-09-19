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
from pathlib import Path
import numpy as np
from .core import JOINTS, NATIVE_UNITS, stamp, values, to_sim


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


class Leader:
    def __init__(self, env, port, calibration_file_name, recalibrate=False):
        from soarm101_lab.devices import SO101Leader, SO101LeaderCfg

        class StrictLeader(SO101Leader):
            def calibrate(inner):
                raise RuntimeError(
                    "Leader calibration missing or requested implicitly. Select the existing calibration file; automatic calibration is disabled."
                )

            def _get_raw_data(inner):
                # Present_Position and norm modes are verified in the repository's own implementation.
                native = values(inner._bus.sync_read("Present_Position"))
                inner.last_native = native
                return native

        cfg = SO101LeaderCfg(port=port, calibration_file_name=calibration_file_name, recalibrate=recalibrate)
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

    def activate(self):
        if not self.device.is_calibrated:
            raise RuntimeError("Leader calibration changed")
        self.device._bus.disable_torque()

    def read(self):
        start = stamp()
        action = self.device.advance()
        end = stamp()
        native = values(self.device.last_native)
        got = action.detach().cpu().numpy()
        expected = np.array([to_sim(native)[n] for n in JOINTS])
        if got.shape != (6,) or not np.allclose(got, expected, atol=1e-6):
            raise RuntimeError(
                "Local leader.advance mapping differs from inspected repo V2. Refusing to guess signs/gripper offsets."
            )
        return native, {"start": start, "end": end}

    def close(self):
        if self.device.is_connected:
            self.device.disconnect()


class Follower:
    def __init__(
        self, port, identifier, calibration_dir=None, allow_calibration=False
    ):
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
        except BaseException:
            self.close()
            raise

    def read(self):
        start = stamp()
        obs = self.device.get_observation()
        end = stamp()
        return values({n: obs[n + ".pos"] for n in JOINTS}), {"start": start, "end": end}

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

    def stop(self):
        # Hold at the last target; no new goal. Disconnect releases torque via audited API.
        pass

    def send(self, target):
        target = values(target)
        start = stamp()
        sent = self.device.send_action({n + ".pos": v for n, v in target.items()})
        end = stamp()
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
